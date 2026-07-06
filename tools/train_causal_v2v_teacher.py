"""Stage-0: 因果 AR v2v teacher 短训(路线B 前缀 token, 16 通道)。

产出 checkpoints/bernini_causal_v2v_teacher.pt, 供:
  - Causal ODE 离线采样 (generate_v2v_ode_lmdb.py --causal_teacher)
  - Causal CD 在线蒸馏 (self_forcing_v2v_causal_cd_routeb.yaml)

数据: ReCo 配对 src + tar + 指令。监督 tar 的 flow target, 条件为 src 前缀 + 文本。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils.dataset import V2VPairedVideoDataset, cycle
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def maybe_init_dist():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    torch.cuda.set_device(0)
    return 0, 1, 0


def load_v2v_generator(model: WanDiffusionWrapper, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("generator", sd)
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith(".freqs") and not m.endswith(".visual_id_freqs")]
    assert not missing and not unexpected, f"加载不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", default="checkpoints/bernini_v2v_prefix_init.pt")
    ap.add_argument("--data_path", default="/apdcephfs/private_huitinglu/ReCo-Data/replace/replace_data_configs.json")
    ap.add_argument("--base_video_folder", default="/apdcephfs/private_huitinglu/ReCo-Data")
    ap.add_argument("--out", default="checkpoints/bernini_causal_v2v_teacher.pt")
    ap.add_argument("--num_raw_frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_frame_per_block", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--condition_dropout", type=float, default=0.1)
    ap.add_argument("--timestep_shift", type=float, default=3.0)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    rank, world_size, local_rank = maybe_init_dist()
    device = torch.cuda.current_device()
    is_main = rank == 0
    dtype = torch.bfloat16

    model = WanDiffusionWrapper(
        model_kwargs={"timestep_shift": args.timestep_shift}, is_causal=True)
    load_v2v_generator(model, args.init_ckpt)
    model.model.num_frame_per_block = args.num_frame_per_block
    model = model.to(device=device, dtype=dtype)
    model.train()
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if world_size > 1 else model

    text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)

    scheduler = FlowMatchScheduler(shift=args.timestep_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)

    dataset = V2VPairedVideoDataset(
        args.data_path, base_video_folder=args.base_video_folder,
        num_frames=args.num_raw_frames, height=args.height, width=args.width)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, drop_last=True) if world_size > 1 else None
    dataloader = cycle(torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=4, drop_last=True))

    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.0, 0.999), weight_decay=0.01)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        load_v2v_generator(raw_model, args.resume)
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))
        if is_main:
            print(f"续训 from {args.resume}, start_step={start_step}")

    if is_main:
        print(f"[Causal AR v2v teacher] dataset={len(dataset)} world_size={world_size} "
              f"max_steps={args.max_steps}")

    for step in range(start_step, args.max_steps):
        batch = next(dataloader)
        src = batch["src_video"].to(device=device, dtype=dtype)
        tar = batch["tar_video"].to(device=device, dtype=dtype)
        prompts = batch["prompts"]

        with torch.no_grad():
            cond_latent = vae.encode_to_latent(src).to(dtype)
            x0 = vae.encode_to_latent(tar).to(dtype)
            conditional_dict = text_encoder(text_prompts=prompts)

        if args.condition_dropout > 0 and torch.rand(1).item() < args.condition_dropout:
            cond_latent = torch.zeros_like(cond_latent)
        conditional_dict = {**conditional_dict, "cond_latent": cond_latent}

        b, fl = x0.shape[:2]
        timestep = torch.randint(0, 1000, (b, fl), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = scheduler.add_noise(
            x0.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (b, fl)).to(dtype)
        flow_target = noise - x0

        flow_pred, _ = raw_model(xt, conditional_dict, timestep.to(dtype))
        loss = F.mse_loss(flow_pred.float(), flow_target.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if is_main and step % 20 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

        if is_main and (step + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "generator": raw_model.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
            }, args.out)
            print(f"saved {args.out}")

    if is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({
            "generator": raw_model.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.max_steps,
        }, args.out)
        print(f"done -> {args.out}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
