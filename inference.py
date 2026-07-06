import argparse
import time
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    BidirectionalInferencePipeline,
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset, V2VVideoDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--v2v", action="store_true", help="Whether to perform streaming V2V (source video as channel-concat condition)")
parser.add_argument("--no_crop", action="store_true",
                    help="v2v: 等比缩放+黑边填充而非居中裁剪, 兼容任意源视频尺寸且不丢内容; 输出会裁掉黑边还原真实内容区域")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
# generator_causal=False(不因果化蒸馏的双向模型)走双向 few-step 推理, 否则保持原因果流式推理。
if not getattr(config, "generator_causal", True):
    assert hasattr(config, 'denoising_step_list'), "双向推理仅支持 few-step (需 denoising_step_list)"
    pipeline = BidirectionalInferencePipeline(config, device=device)
elif hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

# v2v(方案B): 前缀 token 注入, in_channels 仍 16, 无需扩通道; 源 latent 由 pipeline 预填进 KV cache

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    generator_state_dict = state_dict['generator' if not args.use_ema else 'generator_ema']
    generator_state_dict = {k.replace('_fsdp_wrapped_module.', ''): v for k, v in generator_state_dict.items()}
    pipeline.generator.load_state_dict(generator_state_dict)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
elif args.v2v:
    dataset = V2VVideoDataset(
        args.data_path,
        base_video_folder=getattr(config, "base_video_folder", os.path.dirname(args.data_path)),
        num_frames=getattr(config, "num_raw_frames", 81),
        height=getattr(config, "height", 480),
        width=getattr(config, "width", 832),
        prompt_key=getattr(config, "prompt_key", "instruction_final_refine"),
        src_key=getattr(config, "src_key", "src_video"),
        target_fps=getattr(config, "target_fps", 16),
        fit_mode="pad" if args.no_crop else "crop",
    )
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    cond_latent = None

    if args.v2v:
        prompt = batch['prompts'][0]
        prompts = [prompt] * args.num_samples
        src_video = batch['src_video'].to(device=device, dtype=torch.bfloat16)  # [1, C, F, H, W]
        cond_latent = pipeline.vae.encode_to_latent(src_video).to(device=device, dtype=torch.bfloat16)
        cond_latent = cond_latent.repeat(args.num_samples, 1, 1, 1, 1)
        num_cond_frames = cond_latent.shape[1]
        initial_latent = None
        sampled_noise = torch.randn(
            [args.num_samples, num_cond_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    elif args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate 81 frames
    torch.cuda.synchronize()
    _t0 = time.time()
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
        cond_latent=cond_latent,
    )
    torch.cuda.synchronize()
    _gen_t = time.time() - _t0
    _n_frames = video.shape[1]
    if local_rank == 0:
        print(f"[realtime] 样本{idx}: 生成 {_n_frames} 帧, 耗时 {_gen_t:.2f}s, 等效 {(_n_frames / _gen_t):.1f} FPS "
              f"({args.num_samples} samples/prompt)", flush=True)
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # v2v --no_crop: 源视频是等比缩放+黑边填充进来的, 这里按记录的 pad_box 裁掉黑边, 只保留真实内容区域
    if args.v2v and "pad_box" in batch:
        pb_top, pb_left, pb_h, pb_w = [int(x[0]) for x in batch["pad_box"]]
        video = video[:, :, pb_top:pb_top + pb_h, pb_left:pb_left + pb_w, :]

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
