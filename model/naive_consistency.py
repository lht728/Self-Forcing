import random
from typing import Tuple

import torch
import torch.nn.functional as F

from model.base import BaseModel
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def _load_v2v_state_dict(ckpt_path: str) -> dict:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    elif "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    fixed = {}
    for k, v in state_dict.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[k] = v
    return fixed


def _load_into_causal_wrapper(wrapper: WanDiffusionWrapper, ckpt_path: str, v2v: bool) -> None:
    state_dict = _load_v2v_state_dict(ckpt_path)
    if v2v:
        missing, unexpected = wrapper.model.load_state_dict(state_dict, strict=False)
        missing = [
            m for m in missing
            if not m.endswith(".freqs") and not m.endswith(".visual_id_freqs")
        ]
        assert not missing and not unexpected, (
            f"v2v causal 载入不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    else:
        wrapper.load_state_dict(state_dict, strict=True)


class NaiveConsistency(BaseModel):
    """Causal Consistency Distillation (Causal Forcing++ Stage-1 init).

    v2v(路线B): teacher/student 均为因果模型, 源条件走 cond_latent 前缀注入;
    clean_latent 为编辑后目标视频 latent。DMD 阶段仍用双向 Bernini teacher(见 distillation)。
    """

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.generator_ema = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator_ema.model.requires_grad_(False)

        self.teacher = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True)
        self.teacher.model.requires_grad_(False)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def __init__(self, args, device):
        super().__init__(args, device)

        self.v2v = getattr(args, "v2v", False)
        self.teacher_forcing = getattr(args, "teacher_forcing", False) and not self.v2v

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            self.generator_ema.model.num_frame_per_block = self.num_frame_per_block
            self.teacher.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.guidance_scale = args.guidance_scale
        self.discrete_cd_N = getattr(args, "discrete_cd_N", 48)
        cd_shift = getattr(args, "cd_timestep_shift", 5.0)
        self.cd_scheduler = FlowMatchScheduler(
            shift=cd_shift, sigma_min=0.0, extra_one_step=True)
        self.cd_scheduler.set_timesteps(
            num_inference_steps=self.discrete_cd_N, denoising_strength=1.0)
        self.cd_scheduler.sigmas = self.cd_scheduler.sigmas.to(device)

        if getattr(args, "generator_ckpt", False):
            _load_into_causal_wrapper(self.generator, args.generator_ckpt, self.v2v)
            _load_into_causal_wrapper(self.generator_ema, args.generator_ckpt, self.v2v)

        teacher_ckpt = getattr(args, "teacher_ckpt", None) or getattr(args, "generator_ckpt", False)
        if teacher_ckpt:
            _load_into_causal_wrapper(self.teacher, teacher_ckpt, self.v2v)

    def generator_loss(
        self,
        conditional_dict,
        unconditional_dict,
        clean_latent,
        ema_model,
    ) -> Tuple[torch.Tensor, dict]:
        clean_latent = clean_latent.to(self.device).to(torch.bfloat16)
        B, num_frames = clean_latent.shape[:2]
        timestep_idx = random.randrange(self.discrete_cd_N - 1)

        t = self.cd_scheduler.timesteps[timestep_idx]
        timestep = t * torch.ones([B, num_frames], device=self.device, dtype=torch.bfloat16)
        t_next = self.cd_scheduler.timesteps[timestep_idx + 1]
        timestep_next = t_next * torch.ones(
            [B, num_frames], device=self.device, dtype=torch.bfloat16)

        noise = torch.randn_like(clean_latent)
        latent_t = self.cd_scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise=noise.flatten(0, 1),
            timestep=t * torch.ones([1], device=self.device),
        ).unflatten(0, (B, num_frames)).to(torch.bfloat16)

        clean_x = clean_latent if self.teacher_forcing else None

        with torch.no_grad():
            v_cond, _ = self.teacher(
                latent_t, conditional_dict, timestep, clean_x=clean_x)
            v_uncond, _ = self.teacher(
                latent_t, unconditional_dict, timestep, clean_x=clean_x)
            v_pred = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            dt = (timestep - timestep_next).reshape(B, num_frames, 1, 1, 1) / 1000
            latent_t_next = latent_t - dt * v_pred

        if self.generator.model.block_mask is None and self.teacher.model.block_mask is not None:
            self.generator.model.block_mask = self.teacher.model.block_mask
            self.generator_ema.model.block_mask = self.teacher.model.block_mask

        _, cm_pred_t = self.generator(
            latent_t, conditional_dict, timestep, clean_x=clean_x)

        with torch.no_grad():
            ema_model.copy_to(self.generator_ema)
            _, cm_pred_t_next = self.generator_ema(
                latent_t_next, conditional_dict, timestep_next, clean_x=clean_x)

        loss = F.mse_loss(cm_pred_t, cm_pred_t_next, reduction="mean")
        log_dict = {
            "unnormalized_loss": F.mse_loss(
                cm_pred_t, cm_pred_t_next, reduction="none"
            ).mean(dim=[1, 2, 3, 4]).detach(),
            "cd_timestep": t.detach(),
        }
        return loss, log_dict
