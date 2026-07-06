"""Bernini 1.3B 双向教师模型(多步扩散, 非蒸馏)v2v 推理, 单卡。

复用 tools/generate_v2v_ode_lmdb.py 的采样循环(路线B 前缀 token, --prefix_v2v),
默认 24 步 + guidance_scale=6.0, 对齐离线 ODE teacher 采样设置。
支持 --no_crop: 源视频等比缩放+黑边填充(不裁剪内容), 输出按 pad_box 裁掉黑边。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.io import write_video

from utils.dataset import V2VVideoDataset
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

NEG_PROMPT = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，'
              '最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，'
              '画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，'
              '杂乱的背景，三条腿，背景人很多，倒着走')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", default="checkpoints/bernini_v2v_prefix_init.pt")
    ap.add_argument("--model_name", default="Wan2.1-T2V-1.3B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--base_video_folder", default=None)
    ap.add_argument("--prompt_key", default="instruction_final_refine")
    ap.add_argument("--src_key", default="src_video")
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--num_raw_frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--target_fps", type=int, default=16)
    ap.add_argument("--num_steps", type=int, default=24)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--timestep_shift", type=float, default=3.0)
    ap.add_argument("--no_crop", action="store_true",
                    help="等比缩放+黑边填充而非居中裁剪, 兼容任意源视频尺寸且不丢内容; 输出自动裁掉黑边")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[bidir teacher] ckpt={args.teacher_ckpt}, num_steps={args.num_steps}, "
          f"guidance_scale={args.guidance_scale}, no_crop={args.no_crop}")

    model = WanDiffusionWrapper(
        model_name=args.model_name, is_causal=False, timestep_shift=args.timestep_shift,
    ).to(device).to(torch.bfloat16)
    sd = torch.load(args.teacher_ckpt, map_location="cpu")
    sd = sd.get("generator", sd)
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith(".freqs") and not m.endswith(".visual_id_freqs")]
    assert not missing and not unexpected, f"teacher 加载不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"
    model.model.requires_grad_(False)
    model.eval()

    encoder = WanTextEncoder().to(device).to(torch.float32).eval()
    vae = WanVAEWrapper().to(device).to(torch.bfloat16).eval()

    scheduler = FlowMatchScheduler(shift=args.timestep_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=args.num_steps, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    base_video_folder = args.base_video_folder or os.path.dirname(args.data_path)
    dataset = V2VVideoDataset(
        args.data_path, base_video_folder=base_video_folder,
        num_frames=args.num_raw_frames, height=args.height, width=args.width,
        prompt_key=args.prompt_key, src_key=args.src_key, target_fps=args.target_fps,
        fit_mode="pad" if args.no_crop else "crop",
    )
    item = dataset[0]
    prompt = item["prompts"]
    pad_box = item.get("pad_box", None)
    print(f"prompt: {prompt}")

    src_video = item["src_video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    cond_latent = vae.encode_to_latent(src_video).to(torch.float32)  # [1, F, 16, h, w]
    cond_pe = encoder(text_prompts=[prompt])["prompt_embeds"][0]
    uncond = encoder(text_prompts=[NEG_PROMPT])

    cond_latent2 = cond_latent.repeat(2, 1, 1, 1, 1)
    both_dict = {
        "prompt_embeds": [cond_pe, uncond["prompt_embeds"][0]],
        "cond_latent": cond_latent2,
    }

    b, fl, _, h, w = cond_latent.shape
    latents = torch.randn([1, fl, 16, h, w], dtype=torch.float32, device=device)

    os.makedirs(args.output_folder, exist_ok=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for progress_id, t in enumerate(scheduler.timesteps):
        timestep = t * torch.ones([1, fl], device=device, dtype=torch.float32)
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

    video = vae.decode_to_pixel(latents.to(torch.bfloat16), use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_frames = video.shape[1]
    fps = n_frames / dt
    print(f"[realtime] 生成 {n_frames} 帧, 耗时 {dt:.2f}s, 等效 {fps:.1f} FPS ({args.num_steps} 步双向教师)")

    from einops import rearrange
    thwc = rearrange(video, 'b t c h w -> b t h w c')[0].float().mul_(255.0).clamp_(0, 255).to(torch.uint8).cpu()
    if pad_box is not None:
        top, left, ch, cw = pad_box
        thwc = thwc[:, top:top + ch, left:left + cw, :]

    out_path = os.path.join(args.output_folder, "0-0_bidir24.mp4")
    write_video(out_path, thwc, fps=16)
    print(f"saved {out_path}")

    import json
    import datetime
    json.dump({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "teacher_ckpt": args.teacher_ckpt,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "no_crop": args.no_crop,
        "output_video": out_path,
        "frames": n_frames,
        "seconds": dt,
        "fps_exact": fps,
    }, open(os.path.join(args.output_folder, "fps_metrics.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
