"""Bernini v2v ODE 轨迹离线生成 -> ODERegressionLMDBDataset 可读的 LMDB。

Stage-1 初始化三种方案中的两种需要本脚本(第三种 Causal CD 免采样):

  A) init_method=ode (默认)
     双向 Bernini teacher 采样, 解决前缀读法适配; 轨迹与 AR student 不完全对齐。
     torchrun ... sample --prefix_v2v --teacher_ckpt checkpoints/bernini_v2v_prefix_init.pt

  B) init_method=causal_ode (Causal Forcing)
     因果 AR v2v teacher 采样, 轨迹与 student 结构对齐(injectivity)。
     torchrun ... sample --prefix_v2v --causal_teacher \
       --teacher_ckpt checkpoints/bernini_causal_v2v_teacher.pt

  C) init_method=causal_cd → 不使用本脚本

teacher 范式(路线B 前缀 token, 16 通道):
  - --prefix_v2v: 源 latent patchify 成前缀 token 注入。
  - --causal_teacher: 同上, 但 teacher/student 均为因果 WanDiffusionWrapper。
  - 路线A(--cond_channels>0, 已弃用): 通道拼接 32 通道, 不再推荐。

LMDB 字段(与 utils/dataset.py:ODERegressionLMDBDataset 对齐):
  - latents      [N, 5, F, 16, h, w]  每条 5 个关键帧, 对齐 denoising_step_list[1000,750,500,250] + 干净目标
  - cond_latent  [N, F, 16, h, w]     源视频 VAE latent(前缀 token 条件)
  - prompts      [N]                  指令文本
  关键帧索引按 num_steps 自动对齐到 denoising_step_list[1000,750,500,250,0] 的噪声比例。

两阶段(多卡采样 + 单进程聚合):
  采样:  torchrun --nproc_per_node 8 tools/generate_v2v_ode_lmdb.py sample --prefix_v2v \
            --teacher_ckpt checkpoints/bernini_v2v_prefix_init.pt --shard_dir ode_data/v2v_routeb_shards
  聚合:  python tools/generate_v2v_ode_lmdb.py build \
            --shard_dir ode_data/v2v_routeb_shards --lmdb_path ode_data/bernini_v2v_ode_routeb_lmdb
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmdb
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.dataset import V2VVideoDataset
from utils.lmdb import store_arrays_to_lmdb
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

NEG_PROMPT = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，'
              '最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，'
              '画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，'
              '杂乱的背景，三条腿，背景人很多，倒着走')
def _keyframe_indices(num_steps: int) -> list:
    """与 denoising_step_list [1000,750,500,250,0] 成比例; 48 步 -> [0,12,24,36,-1], 24 步 -> [0,6,12,18,-1]。"""
    n = num_steps
    return [0, n // 4, n // 2, (3 * n) // 4, -1]


def _maybe_init_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(0)
    return 0, 1


@torch.no_grad()
def sample(args):
    rank, world_size = _maybe_init_dist()
    device = torch.cuda.current_device()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 路线B(前缀 token): 不扩通道, in_channels 仍 16, 源前缀由 cond_latents 注入(model.py 内部).
    # 路线A(已弃用): cond_channels>0 时通道拼接扩到 16+cond_channels.
    prefix_v2v = args.prefix_v2v or args.cond_channels == 0
    model = WanDiffusionWrapper(
        model_name=args.model_name,
        is_causal=args.causal_teacher,
        timestep_shift=args.timestep_shift,
    ).to(device).to(torch.bfloat16)
    if args.causal_teacher and args.num_frame_per_block > 1:
        model.model.num_frame_per_block = args.num_frame_per_block
    if not prefix_v2v:
        model.expand_in_channels(16 + args.cond_channels)
    sd = torch.load(args.teacher_ckpt, map_location="cpu")
    sd = sd.get("generator", sd)
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith(".freqs") and not m.endswith(".visual_id_freqs")]
    assert not missing and not unexpected, f"teacher 加载不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"
    teacher_type = "因果AR" if args.causal_teacher else "双向"
    if rank == 0:
        print(f"[ODE sample] init={'causal_ode' if args.causal_teacher else 'ode'}, "
              f"teacher={teacher_type}, 范式={'前缀token(路线B)' if prefix_v2v else f'通道拼接(路线A)'}, "
              f"ckpt={args.teacher_ckpt}")
    model.model.requires_grad_(False)
    model.eval()
    # 静态形状(所有样本同分辨率/帧数), torch.compile 可融合 norm/rope/激活并降低调度开销。
    # 首样本会触发编译预热(数十秒), 之后稳定提速; 与 FA2 共存(FA 处会图断但其余仍融合)。
    if args.compile:
        model.model = torch.compile(model.model, mode="default", dynamic=False)

    encoder = WanTextEncoder().to(device).to(torch.float32).eval()
    vae = WanVAEWrapper().to(device).to(torch.bfloat16).eval()

    scheduler = FlowMatchScheduler(shift=args.timestep_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=args.num_steps, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    keyframe_indices = _keyframe_indices(args.num_steps)
    if rank == 0:
        print(f"[ODE sample] num_steps={args.num_steps}, keyframe_indices={keyframe_indices}")

    uncond = encoder(text_prompts=[NEG_PROMPT])

    dataset = V2VVideoDataset(
        args.data_path, base_video_folder=args.base_video_folder,
        num_frames=args.num_raw_frames, height=args.height, width=args.width,
        prompt_key=args.prompt_key, src_key=args.src_key, target_fps=args.target_fps,
        max_pair=args.max_pair)

    os.makedirs(args.shard_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    n_total = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)

    # 多机无通信分片: 机内用 rank 错开, 各 worker 以 total_workers 为步长领取绝对索引;
    # 配合 skip-existing 可安全并发写同一共享盘目录, 任一机重启可续跑。
    #   - 同构(每机卡数相同): 用 shard_id/num_shards, total_workers = num_shards × world_size。
    #   - 异构(每机卡数不同): 显式 --total_workers(=各机卡数之和) 与 --rank_offset, 本机占 [rank_offset, rank_offset+world_size)。
    if args.total_workers > 0:
        total_workers = args.total_workers
        global_rank = args.rank_offset + rank
        assert 0 <= global_rank < total_workers, \
            f"global_rank({global_rank})=rank_offset({args.rank_offset})+rank({rank}) 越界 [0, total_workers={total_workers})"
    else:
        assert 0 <= args.shard_id < args.num_shards, \
            f"shard_id({args.shard_id}) 必须在 [0, num_shards={args.num_shards}) 内"
        global_rank = args.shard_id * world_size + rank
        total_workers = args.num_shards * world_size
    if rank == 0:
        print(f"[ODE sample] rank_offset={args.rank_offset}, world_size={world_size}, "
              f"total_workers={total_workers}, global_rank(rank0)={global_rank}, n_total={n_total}")

    # 本 worker 待采绝对索引(预先过滤越界/已存在), 便于预取下一条。
    todo = [i * total_workers + global_rank for i in range(int(np.ceil(n_total / total_workers)))]
    todo = [i for i in todo if i < n_total and not os.path.exists(os.path.join(args.shard_dir, f"{i:07d}.pt"))]

    # 异步预取(CPU 视频解码/缓存读) + 异步写盘(cephfs IO): 与 GPU 去噪重叠, 填掉样本间 GPU 空隙。
    # 仅改变调度顺序, 数值与原同步实现完全一致。load 线程串行调用 dataset(decord 非线程安全故只 1 个)。
    load_pool = ThreadPoolExecutor(max_workers=1)
    io_pool = ThreadPoolExecutor(max_workers=2)

    def _load_cpu(sample_index):
        """仅做 CPU 侧工作: 缓存命中读缓存, 否则 decord 解码源视频; 不碰 GPU。"""
        cache_path = os.path.join(args.cache_dir, f"{sample_index:07d}.pt") if args.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            c = torch.load(cache_path, map_location="cpu")
            return {"hit": True, "prompt": c["prompt"],
                    "cond_latent": c["cond_latent"], "cond_pe": c["prompt_embeds"]}
        item = dataset[sample_index]
        return {"hit": False, "prompt": item["prompts"], "src_video": item["src_video"]}

    future = load_pool.submit(_load_cpu, todo[0]) if todo else None
    for i in tqdm(range(len(todo)), disable=rank != 0):
        sample_index = todo[i]
        loaded = future.result()
        if i + 1 < len(todo):  # 提前解码下一条, 与本条 GPU 计算重叠
            future = load_pool.submit(_load_cpu, todo[i + 1])

        if loaded["hit"]:
            prompt = loaded["prompt"]
            cond_latent = loaded["cond_latent"].to(device=device, dtype=torch.float32).unsqueeze(0)
            cond_pe = loaded["cond_pe"].to(device=device, dtype=torch.float32)
        else:
            prompt = loaded["prompt"]
            src_video = loaded["src_video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            cond_latent = vae.encode_to_latent(src_video).to(torch.float32)  # [1, F, 16, h, w]
            cond_pe = encoder(text_prompts=[prompt])["prompt_embeds"][0]  # [seq, dim]
            if args.cache_dir:
                cache_path = os.path.join(args.cache_dir, f"{sample_index:07d}.pt")
                io_pool.submit(torch.save, {
                    "cond_latent": cond_latent.squeeze(0).half().cpu(),
                    "prompt_embeds": cond_pe.half().cpu(),
                    "prompt": prompt,
                }, cache_path)

        # CFG 合批: 把 cond / uncond 拼成 batch=2 一次前向, 数值与两次独立前向等价。
        # prompt_embeds 以 list 传入, model 内部逐条 pad 到 text_len, 故两者文本长度可不同。
        cond_latent2 = cond_latent.repeat(2, 1, 1, 1, 1)
        both_dict = {
            "prompt_embeds": [cond_pe, uncond["prompt_embeds"][0]],
            "cond_latent": cond_latent2,
        }

        b, fl, _, h, w = cond_latent.shape
        latents = torch.randn([1, fl, 16, h, w], dtype=torch.float32, device=device)

        traj = []
        for progress_id, t in enumerate(scheduler.timesteps):
            timestep = t * torch.ones([1, fl], device=device, dtype=torch.float32)
            traj.append(latents)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, x0_both = model(latents.repeat(2, 1, 1, 1, 1), both_dict, timestep.repeat(2, 1))
            x0_cond = x0_both[:1].float()
            x0_uncond = x0_both[1:2].float()
            x0_pred = x0_uncond + args.guidance_scale * (x0_cond - x0_uncond)
            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler, x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1), timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])
            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones([1, fl], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(0, flow_pred.shape[:2])
        traj.append(latents)

        traj = torch.stack(traj, dim=1)[:, keyframe_indices]  # [1, 5, F, 16, h, w]
        out_path = os.path.join(args.shard_dir, f"{sample_index:07d}.pt")
        io_pool.submit(torch.save, {
            "latents": traj.squeeze(0).half().cpu(),       # [5, F, 16, h, w]
            "cond_latent": cond_latent.squeeze(0).half().cpu(),  # [F, 16, h, w]
            "prompt": prompt,
        }, out_path)

    load_pool.shutdown(wait=True)
    io_pool.shutdown(wait=True)  # 等所有后台写盘落地后再退出

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def cache(args):
    """批量预计算 cond_latent(VAE) + prompt_embeds(UMT5) 写入 cache_dir, 供 sample --cache_dir 复用。
    缓存键为数据集绝对索引(与 world_size 无关), 批量编码摊薄 VAE/文本编码开销, 采样阶段仅做去噪。"""
    rank, world_size = _maybe_init_dist()
    device = torch.cuda.current_device()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    encoder = WanTextEncoder().to(device).to(torch.float32).eval()
    vae = WanVAEWrapper().to(device).to(torch.bfloat16).eval()

    dataset = V2VVideoDataset(
        args.data_path, base_video_folder=args.base_video_folder,
        num_frames=args.num_raw_frames, height=args.height, width=args.width,
        prompt_key=args.prompt_key, src_key=args.src_key, target_fps=args.target_fps,
        max_pair=args.max_pair)

    os.makedirs(args.cache_dir, exist_ok=True)
    n_total = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)

    my_indices = list(range(rank, n_total, world_size))
    bs = args.cache_batch_size
    for bstart in tqdm(range(0, len(my_indices), bs), disable=rank != 0):
        batch = [i for i in my_indices[bstart:bstart + bs]
                 if not os.path.exists(os.path.join(args.cache_dir, f"{i:07d}.pt"))]
        if not batch:
            continue

        videos, prompts, valid = [], [], []
        for i in batch:
            item = dataset[i]
            videos.append(item["src_video"])
            prompts.append(item["prompts"])
            valid.append(i)

        video = torch.stack(videos, dim=0).to(device=device, dtype=torch.bfloat16)  # [B, C, F, H, W]
        cond_latent = vae.encode_to_latent(video).to(torch.float16)  # [B, F, 16, h, w]
        ctx = encoder(text_prompts=prompts)["prompt_embeds"]  # list[B] of [seq, dim]

        for j, i in enumerate(valid):
            torch.save({
                "cond_latent": cond_latent[j].cpu(),     # [F, 16, h, w] fp16
                "prompt_embeds": ctx[j].half().cpu(),    # [seq, dim] fp16
                "prompt": prompts[j],
            }, os.path.join(args.cache_dir, f"{i:07d}.pt"))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def build(args):
    files = sorted(glob.glob(os.path.join(args.shard_dir, "*.pt")))
    assert files, f"未找到 .pt 分片: {args.shard_dir}"
    env = lmdb.open(args.lmdb_path, map_size=args.map_size_tb * (1024 ** 4))

    counter = 0
    seen = set()
    last_shapes = {}
    for f in tqdm(files):
        d = torch.load(f, map_location="cpu")
        prompt = d["prompt"]
        if prompt in seen:
            continue
        seen.add(prompt)
        arrays = {
            "latents": d["latents"].numpy()[None],        # [1, 5, F, 16, h, w]
            "cond_latent": d["cond_latent"].numpy()[None],  # [1, F, 16, h, w]
            "prompts": np.array([prompt]),
        }
        store_arrays_to_lmdb(env, arrays, start_index=counter)
        for k, v in arrays.items():
            last_shapes[k] = v.shape
        counter += 1

    with env.begin(write=True) as txn:
        for key, shape in last_shapes.items():
            arr_shape = np.array(shape)
            arr_shape[0] = counter
            txn.put(f"{key}_shape".encode(), " ".join(map(str, arr_shape)).encode())
    print(f"LMDB 写入完成: {args.lmdb_path}, 样本数={counter}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--teacher_ckpt", required=True)
    s.add_argument("--shard_dir", default="ode_data/v2v_shards")
    s.add_argument("--model_name", default="Wan2.1-T2V-1.3B")
    s.add_argument("--data_path", default="/apdcephfs/private_huitinglu/ReCo-Data/replace/replace_data_configs.json")
    s.add_argument("--base_video_folder", default="/apdcephfs/private_huitinglu/ReCo-Data")
    s.add_argument("--prompt_key", default="instruction_final_refine")
    s.add_argument("--src_key", default="src_video")
    s.add_argument("--cond_channels", type=int, default=0,
                   help="路线A通道拼接的源通道数; 0=路线B前缀token(默认, 不扩通道)")
    s.add_argument("--prefix_v2v", action="store_true",
                   help="路线B: 前缀 token v2v, 不扩通道(16ch teacher, 如 bernini_v2v_prefix_init.pt)")
    s.add_argument("--causal_teacher", action="store_true",
                   help="Causal Forcing 因果 ODE: 用因果 AR teacher 采样(init_method=causal_ode)")
    s.add_argument("--num_frame_per_block", type=int, default=3,
                   help="因果 teacher 的 block 大小, 与 DMD/ODE 训练 config 对齐")
    s.add_argument("--num_raw_frames", type=int, default=81)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--width", type=int, default=832)
    s.add_argument("--target_fps", type=int, default=16)
    s.add_argument("--num_steps", type=int, default=24)
    s.add_argument("--guidance_scale", type=float, default=6.0)
    s.add_argument("--timestep_shift", type=float, default=3.0)
    s.add_argument("--max_samples", type=int, default=-1)
    s.add_argument("--max_pair", type=int, default=int(1e8))
    s.add_argument("--compile", action="store_true", help="torch.compile teacher 以提高 MFU")
    s.add_argument("--cache_dir", default="", help="cond_latent/prompt_embeds 预缓存目录, 命中则跳过 VAE/文本编码")
    s.add_argument("--num_shards", type=int, default=1, help="同构多机分片总数(每台机器一个 shard_id), 无跨机通信")
    s.add_argument("--shard_id", type=int, default=0, help="本机分片编号, 取值 [0, num_shards)")
    s.add_argument("--total_workers", type=int, default=0,
                   help="异构分片: 全局 worker 总数(=各机卡数之和); >0 时启用并忽略 num_shards/shard_id")
    s.add_argument("--rank_offset", type=int, default=0,
                   help="异构分片: 本机首卡的全局 worker 编号偏移(本机占 [rank_offset, rank_offset+本机卡数))")

    c = sub.add_parser("cache")
    c.add_argument("--cache_dir", required=True)
    c.add_argument("--data_path", default="/apdcephfs/private_huitinglu/ReCo-Data/replace/replace_data_configs.json")
    c.add_argument("--base_video_folder", default="/apdcephfs/private_huitinglu/ReCo-Data")
    c.add_argument("--prompt_key", default="instruction_final_refine")
    c.add_argument("--src_key", default="src_video")
    c.add_argument("--num_raw_frames", type=int, default=81)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--width", type=int, default=832)
    c.add_argument("--target_fps", type=int, default=16)
    c.add_argument("--max_samples", type=int, default=-1)
    c.add_argument("--max_pair", type=int, default=int(1e8))
    c.add_argument("--cache_batch_size", type=int, default=4)

    b = sub.add_parser("build")
    b.add_argument("--shard_dir", default="ode_data/v2v_shards")
    b.add_argument("--lmdb_path", default="ode_data/bernini_v2v_ode_lmdb")
    b.add_argument("--map_size_tb", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args)
    elif args.cmd == "cache":
        cache(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
