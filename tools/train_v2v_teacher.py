"""Bernini -> 通道拼接 v2v teacher 的短训(双向 WanModel, flow-matching 微调)。

目的: 让转换后的 WanModel(patch_embedding 已扩到 32 通道, 新增通道零初始化)学会
从通道里读取源视频条件, 产出可作为 DMD real_score 的 v2v teacher。

数据: ReCo-Data 配对 (src_video, tar_video, 指令)。监督目标为 tar 的 flow target。
说明: 依赖 Self-Forcing 既有的 wan_models/Wan2.1-T2V-1.3B/ (VAE + UMT5, 与整套蒸馏一致)。

单卡:   python tools/train_v2v_teacher.py --init_ckpt checkpoints/bernini_wanmodel_v2v_init.pt ...
多卡:   torchrun --nproc_per_node 8 tools/train_v2v_teacher.py ...
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from wan.modules.model import WanModel
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
from utils.dataset import V2VPairedVideoDataset, cycle


def maybe_init_dist():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    torch.cuda.set_device(0)
    return 0, 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", required=True, help="convert_bernini_to_wanmodel.py 产出的 32 通道初始权重")
    ap.add_argument("--data_path", default="/apdcephfs/private_huitinglu/ReCo-Data/replace/replace_data_configs.json")
    ap.add_argument("--base_video_folder", default="/apdcephfs/private_huitinglu/ReCo-Data")
    ap.add_argument("--out", default="checkpoints/bernini_v2v_teacher.pt")
    ap.add_argument("--cond_channels", type=int, default=16)
    ap.add_argument("--num_raw_frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--timestep_shift", type=float, default=3.0)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=32760)
    ap.add_argument("--grad_ckpt", action="store_true")
    args = ap.parse_args()

    rank, world_size, local_rank = maybe_init_dist()
    device = torch.cuda.current_device()
    is_main = rank == 0
    dtype = torch.bfloat16

    # 模型: 32 通道 WanModel, 加载转换后的初始权重
    model = WanModel(model_type="t2v", in_dim=16 + args.cond_channels, dim=1536,
                     ffn_dim=8960, freq_dim=256, out_dim=16, num_heads=12,
                     num_layers=30, text_len=512, eps=1e-6)
    sd = torch.load(args.init_ckpt, map_location="cpu")
    sd = sd.get("generator", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith(".freqs")]
    assert not missing and not unexpected, f"加载不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"
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
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
    else:
        sampler = None
    dataloader = cycle(torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=4, drop_last=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    if is_main:
        print(f"teacher 短训开始: dataset={len(dataset)} world_size={world_size} max_steps={args.max_steps}")

    for step in range(args.max_steps):
        batch = next(dataloader)
        src = batch["src_video"].to(device=device, dtype=dtype)   # [B,C,F,H,W]
        tar = batch["tar_video"].to(device=device, dtype=dtype)
        prompts = batch["prompts"]

        with torch.no_grad():
            cond_lat = vae.encode_to_latent(src).to(dtype)        # [B,Fl,16,h,w]
            x0 = vae.encode_to_latent(tar).to(dtype)
            text = text_encoder(text_prompts=prompts)["prompt_embeds"]

        b, fl = x0.shape[:2]
        timestep = torch.randint(0, 1000, (b,), device=device, dtype=torch.long)
        t_perframe = timestep[:, None].expand(b, fl)
        noise = torch.randn_like(x0)
        xt = scheduler.add_noise(
            x0.flatten(0, 1), noise.flatten(0, 1), t_perframe.flatten(0, 1)
        ).unflatten(0, (b, fl)).to(dtype)

        # flow target = noise - x0 (与 WanDiffusionWrapper 的 x0<->flow 约定一致)
        flow_target = (noise - x0)

        flow_pred = raw_model(
            xt.permute(0, 2, 1, 3, 4),
            t=timestep, context=text, seq_len=args.seq_len,
            y=cond_lat.permute(0, 2, 1, 3, 4),
        ).permute(0, 2, 1, 3, 4)

        loss = F.mse_loss(flow_pred.float(), flow_target.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_main and step % 20 == 0:
            print(f"step {step} loss {loss.item():.4f}")

        if is_main and (step + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"generator": raw_model.state_dict()}, args.out)
            print(f"已保存 teacher: {args.out} (step {step + 1})")

    if is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"generator": raw_model.state_dict()}, args.out)
        print(f"训练完成, 已保存 teacher: {args.out}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
