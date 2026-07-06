import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb


def load_src_video(path, width, height, num_frames, target_fps):
    import decord
    vr = decord.VideoReader(path, width=width, height=height)
    step = max(vr.get_avg_fps() / float(target_fps), 1e-6)
    idx = np.arange(0, len(vr), step).astype(int)
    if len(idx) < num_frames:
        idx = np.concatenate([idx, np.array([idx[-1]] * (num_frames - len(idx)))])
    idx = np.clip(idx[:num_frames], 0, len(vr) - 1)
    frames = vr.get_batch(list(idx)).asnumpy()
    video = torch.from_numpy(frames).float().div_(255.0).mul_(2.0).sub_(1.0)
    return video.permute(3, 0, 1, 2).contiguous().unsqueeze(0)  # [1, C, F, H, W]


@torch.no_grad()
def generate_prompt_switch(pipeline, cond_latent, embeds1, embeds2, switch_block, device, dtype, seed):
    set_seed(seed)
    num_source_frames = cond_latent.shape[1]
    num_frames = cond_latent.shape[1]
    fsl = pipeline.frame_seq_length
    npb = pipeline.num_frame_per_block
    num_blocks = num_frames // npb

    noise = torch.randn([1, num_frames, 16, 60, 104], device=device, dtype=dtype)

    pipeline.kv_cache1 = None
    pipeline._initialize_kv_cache(batch_size=1, dtype=dtype, device=device, num_source_frames=num_source_frames)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=dtype, device=device)

    cd1 = {"prompt_embeds": embeds1}
    cd2 = {"prompt_embeds": embeds2}

    # 源前缀用 prompt1 预填(编辑指令的公共部分一致, 源 self-attn KV 与文本无关的部分已入 cache)
    pipeline.generator.prefill_source(
        source_latent=cond_latent, conditional_dict=cd1,
        kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache, current_start=0,
    )

    output = torch.zeros([1, num_frames, 16, 60, 104], device=device, dtype=dtype)
    current_start_frame = 0
    denoising_step_list = pipeline.denoising_step_list

    for block_index in range(num_blocks):
        cond = cd2 if block_index >= switch_block else cd1
        # 切换点: 清空各层 cross-attn 文本缓存, 使后续 block 改用 prompt2 的 K/V
        if block_index == switch_block:
            for c in pipeline.crossattn_cache:
                c["is_init"] = False

        noisy_input = noise[:, current_start_frame:current_start_frame + npb]
        for index, current_timestep in enumerate(denoising_step_list):
            timestep = torch.ones([1, npb], device=device, dtype=torch.int64) * current_timestep
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input, conditional_dict=cond, timestep=timestep,
                kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                current_start=(num_source_frames + current_start_frame) * fsl,
            )
            if index < len(denoising_step_list) - 1:
                next_timestep = denoising_step_list[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1), torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([npb], device=device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])

        output[:, current_start_frame:current_start_frame + npb] = denoised_pred

        context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
        pipeline.generator(
            noisy_image_or_video=denoised_pred, conditional_dict=cond, timestep=context_timestep,
            kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
            current_start=(num_source_frames + current_start_frame) * fsl,
        )
        current_start_frame += npb

    video = pipeline.vae.decode_to_pixel(output, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video  # [1, F, C, H, W] in [0,1]


def build_compare(src_mp4, gen_video_thwc, out_png, switch_frame, p1, p2, fps, n=6, H=270, W=468):
    import decord
    vr = decord.VideoReader(src_mp4, width=W, height=H)
    total = len(vr)
    sidx = np.linspace(0, total - 1, n).astype(int)
    sf = vr.get_batch(list(sidx)).asnumpy()

    gv = gen_video_thwc  # [F, H, W, C] uint8
    gtotal = gv.shape[0]
    gidx = np.linspace(0, gtotal - 1, n).astype(int)

    pad, lab = 8, 30
    gw = W * n + pad * (n + 1)
    gh = lab + H * 2 + pad * 3
    canvas = Image.new("RGB", (gw, gh), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 6), f"[SRC] top   vs   [OUT] ckpt400 prompt-switch @frame~{switch_frame} ({fps:.1f} FPS)  P1(red long) -> P2(golden short)", fill=(255, 255, 255))
    for i in range(n):
        x = pad + i * (W + pad)
        canvas.paste(Image.fromarray(sf[i]), (x, lab + pad))
        thumb = Image.fromarray(gv[gidx[i]]).resize((W, H))
        canvas.paste(thumb, (x, lab + pad * 2 + H))
        tag = "P2" if gidx[i] >= switch_frame else "P1"
        col = (80, 200, 120) if tag == "P1" else (240, 180, 60)
        d.rectangle([x, lab + pad * 2 + H, x + 34, lab + pad * 2 + H + 18], fill=col)
        d.text((x + 4, lab + pad * 2 + H + 3), tag, fill=(0, 0, 0))
    canvas.save(out_png)
    return canvas.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", required=True, help="v2v json (含 src_video / instruction_final_refine)")
    parser.add_argument("--prompt2_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--switch_frac", type=float, default=0.5)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")

    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), config)

    pipeline = CausalInferencePipeline(config, device=device)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    gsd = state_dict['generator' if not args.use_ema else 'generator_ema']
    gsd = {k.replace('_fsdp_wrapped_module.', ''): v for k, v in gsd.items()}
    pipeline.generator.load_state_dict(gsd)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)

    with open(args.data_path, encoding="utf-8") as f:
        item = json.load(f)[0]
    with open(args.prompt2_path, encoding="utf-8") as f:
        prompt2 = f.read().strip()
    prompt1 = item[getattr(config, "prompt_key", "instruction_final_refine")]
    src_path = item[getattr(config, "src_key", "src_video")]

    src_video = load_src_video(
        src_path, getattr(config, "width", 832), getattr(config, "height", 480),
        getattr(config, "num_raw_frames", 81), getattr(config, "target_fps", 16),
    ).to(device=device, dtype=torch.bfloat16)
    cond_latent = pipeline.vae.encode_to_latent(src_video).to(device=device, dtype=torch.bfloat16)

    embeds1 = pipeline.text_encoder(text_prompts=[prompt1])["prompt_embeds"]
    embeds2 = pipeline.text_encoder(text_prompts=[prompt2])["prompt_embeds"]

    num_blocks = cond_latent.shape[1] // pipeline.num_frame_per_block
    switch_block = max(1, min(num_blocks - 1, round(num_blocks * args.switch_frac)))
    print(f"num_blocks={num_blocks}, switch_block={switch_block} (前{switch_block}块 P1, 其余 P2)")
    print(f"P1: {prompt1}")
    print(f"P2: {prompt2}")

    os.makedirs(args.output_folder, exist_ok=True)
    torch.cuda.synchronize()
    t0 = time.time()
    video = generate_prompt_switch(
        pipeline, cond_latent, embeds1, embeds2, switch_block, device, torch.bfloat16, args.seed)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_frames = video.shape[1]
    fps = n_frames / dt
    print(f"[realtime] 生成 {n_frames} 帧, 耗时 {dt:.2f}s, 等效 {fps:.1f} FPS")

    thwc = rearrange(video, 'b t c h w -> b t h w c')[0].float().mul_(255.0).clamp_(0, 255).to(torch.uint8).cpu()
    mp4_path = os.path.join(args.output_folder, "0-0_promptswitch.mp4")
    write_video(mp4_path, thwc, fps=16)

    switch_frame = int(round(n_frames * switch_block / num_blocks))
    png_path = os.path.join(args.output_folder, "compare_src_vs_ckpt400_promptswitch.png")
    size = build_compare(src_path, thwc.numpy(), png_path, switch_frame, prompt1, prompt2, fps)
    print(f"saved {mp4_path}")
    print(f"saved {png_path} {size}")


if __name__ == "__main__":
    main()
