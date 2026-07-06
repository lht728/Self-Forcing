from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BidirectionalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=False) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        cond_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        双向(非因果) few-step 推理。与 CausalInferencePipeline.inference 接口对齐
        (供 inference.py 复用)。v2v 时把源 latent 作前缀 token(source_id=1)整段注入,
        与训练态一致(无 KV cache, 整段双向去噪)。
        Inputs:
            noise (torch.Tensor): [B, F, C, H, W] 初始噪声。
            text_prompts (List[str]): 文本提示。
            cond_latent (torch.Tensor, optional): v2v 源视频 latent [B, F, C, H, W]。
        Outputs:
            video [B, F, C, H, W] 归一化到 [0,1]; return_latents 时额外返回 pred latent。
        """
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        if cond_latent is not None:
            conditional_dict = {**conditional_dict, "cond_latent": cond_latent}

        # initial point
        noisy_image_or_video = noise
        pred_image_or_video = noise

        # 逐步去噪: 每步预测 x0, 再加噪到下一 timestep(最后一步即输出)
        for index, current_timestep in enumerate(self.denoising_step_list):
            _, pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            )  # [B, F, C, H, W]

            if index == len(self.denoising_step_list) - 1:
                break

            next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                noise.shape[:2], dtype=torch.long, device=noise.device)

            noisy_image_or_video = self.scheduler.add_noise(
                pred_image_or_video.flatten(0, 1),
                torch.randn_like(pred_image_or_video.flatten(0, 1)),
                next_timestep.flatten(0, 1)
            ).unflatten(0, noise.shape[:2])

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, pred_image_or_video
        return video
