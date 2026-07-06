import gc
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

from utils.dataset import V2VVideoDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from utils.misc import set_seed
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def _unwrap_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    if "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    return {
        k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", ""): v
        for k, v in state_dict.items()
    }


def _load_bidirectional_v2v(wrapper, ckpt_path, name):
    state_dict = _unwrap_checkpoint(ckpt_path)
    if state_dict and next(iter(state_dict)).startswith("model."):
        missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpected = wrapper.model.load_state_dict(state_dict, strict=False)
    missing = [
        key for key in missing
        if not key.endswith(".freqs") and not key.endswith(".visual_id_freqs")
    ]
    assert not missing and not unexpected, (
        f"{name} 载入 v2v 权重不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"
    )


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = getattr(config, "ckpt_step", 0)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )

        self.output_path = config.logdir
        self.teacher_num_steps = int(config.teacher_num_steps)
        self.student_num_steps = int(config.student_num_steps)
        assert self.teacher_num_steps > self.student_num_steps >= 2
        self.boundaries = [
            int(round(i * self.teacher_num_steps / self.student_num_steps))
            for i in range(self.student_num_steps + 1)
        ]
        self.boundaries[-1] = self.teacher_num_steps

        model_kwargs = getattr(config, "model_kwargs", {})
        self.student = WanDiffusionWrapper(**model_kwargs, is_causal=False)
        self.teacher = WanDiffusionWrapper(**model_kwargs, is_causal=False)
        self.text_encoder = WanTextEncoder()
        self.vae = WanVAEWrapper()

        _load_bidirectional_v2v(self.student, config.generator_ckpt, "student")
        _load_bidirectional_v2v(self.teacher, getattr(config, "teacher_ckpt", config.generator_ckpt), "teacher")

        self.student.model.requires_grad_(True)
        self.teacher.model.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        if config.gradient_checkpointing:
            self.student.enable_gradient_checkpointing()

        self.teacher_scheduler = FlowMatchScheduler(
            shift=getattr(config, "timestep_shift", 3.0),
            extra_one_step=True,
        )
        self.teacher_scheduler.set_timesteps(self.teacher_num_steps, denoising_strength=1.0)
        self.teacher_scheduler.sigmas = self.teacher_scheduler.sigmas.to(self.device)
        self.teacher_scheduler.timesteps = self.teacher_scheduler.timesteps.to(self.device)

        self.student = fsdp_wrap(
            self.student,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "generator_cpu_offload", False),
        )
        self.teacher = fsdp_wrap(
            self.teacher,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=True,
        )
        self.text_encoder = fsdp_wrap(
            self.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True,
        )
        self.vae = self.vae.to(device=self.device, dtype=self.dtype)
        self.teacher.eval()
        self.text_encoder.eval()
        self.student.train()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.generator_ema = None
        ema_weight = getattr(config, "ema_weight", 0.0)
        if ema_weight is not None and ema_weight > 0:
            self.generator_ema = EMA_FSDP(self.student, decay=ema_weight)

        dataset = V2VVideoDataset(
            config.data_path,
            base_video_folder=config.base_video_folder,
            num_frames=getattr(config, "num_raw_frames", 81),
            height=config.height,
            width=config.width,
            prompt_key=getattr(config, "prompt_key", "instruction_final_refine"),
            src_key=getattr(config, "src_key", "src_video"),
            target_fps=getattr(config, "target_fps", 16),
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        if self.is_main_process:
            print(f"[Progressive CD v2v] dataset size {len(dataset)}")
            print(f"[Progressive CD v2v] teacher_steps={self.teacher_num_steps}, "
                  f"student_steps={self.student_num_steps}, boundaries={self.boundaries}")
        self.dataloader = cycle(dataloader)

        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            assert total_batch_size == config.batch_size * self.world_size

        self.guidance_scale = config.guidance_scale
        self.max_grad_norm = getattr(config, "max_grad_norm_generator", 10.0)
        self.previous_time = None

    def _sigma_at(self, boundary_idx):
        if boundary_idx >= self.teacher_num_steps:
            return torch.zeros([], device=self.device, dtype=torch.float32)
        return self.teacher_scheduler.sigmas[boundary_idx].float()

    def _timestep_at(self, boundary_idx):
        if boundary_idx >= self.teacher_num_steps:
            return torch.zeros([], device=self.device, dtype=torch.float32)
        return self.teacher_scheduler.timesteps[boundary_idx].float()

    @torch.no_grad()
    def _teacher_step(self, latents, conditional_dict, unconditional_dict, step_idx):
        batch_size, num_frames = latents.shape[:2]
        t = self.teacher_scheduler.timesteps[step_idx].float()
        timestep = t * torch.ones([batch_size, num_frames], device=self.device, dtype=torch.float32)
        both_dict = {
            "prompt_embeds": torch.cat([
                conditional_dict["prompt_embeds"],
                unconditional_dict["prompt_embeds"],
            ], dim=0),
            "cond_latent": torch.cat([
                conditional_dict["cond_latent"],
                unconditional_dict["cond_latent"],
            ], dim=0),
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.mixed_precision):
            _, x0_both = self.teacher(
                latents.repeat(2, 1, 1, 1, 1),
                both_dict,
                timestep.repeat(2, 1),
            )
        x0_cond, x0_uncond = x0_both[:batch_size].float(), x0_both[batch_size:].float()
        x0 = x0_uncond + self.guidance_scale * (x0_cond - x0_uncond)
        flow = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.teacher_scheduler,
            x0_pred=x0.flatten(0, 1),
            xt=latents.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, x0.shape[:2])
        return self.teacher_scheduler.step(
            flow.flatten(0, 1),
            timestep.flatten(0, 1),
            latents.flatten(0, 1),
        ).unflatten(0, latents.shape[:2]).detach()

    @torch.no_grad()
    def _teacher_interval(self, noise, conditional_dict, unconditional_dict, interval_idx):
        start = self.boundaries[interval_idx]
        end = self.boundaries[interval_idx + 1]
        latents = noise
        for step_idx in range(start):
            latents = self._teacher_step(latents, conditional_dict, unconditional_dict, step_idx)
        latent_start = latents.detach()
        for step_idx in range(start, end):
            latents = self._teacher_step(latents, conditional_dict, unconditional_dict, step_idx)
        return latent_start, latents.detach(), start, end

    def save(self):
        state_dict = {
            "generator": fsdp_state_dict(self.student),
        }
        if self.generator_ema is not None and self.step >= getattr(self.config, "ema_start_step", 0):
            state_dict["generator_ema"] = self.generator_ema.state_dict()
        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            out_path = os.path.join(ckpt_dir, "model.pt")
            torch.save(state_dict, out_path)
            print(f"Model saved to {out_path}")

    def fwdbwd_one_step(self, batch):
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)

        with torch.no_grad():
            conditional_dict = self.text_encoder(text_prompts=text_prompts)
            unconditional_dict = self.text_encoder(
                text_prompts=[self.config.negative_prompt] * batch_size)

            src_video = batch["src_video"].to(device=self.device, dtype=self.dtype)
            cond_latent = self.vae.encode_to_latent(src_video).to(self.dtype)
            conditional_dict = {**conditional_dict, "cond_latent": cond_latent}
            unconditional_dict = {**unconditional_dict, "cond_latent": cond_latent}

            noise = torch.randn(
                [batch_size, cond_latent.shape[1], 16, cond_latent.shape[3], cond_latent.shape[4]],
                device=self.device,
                dtype=self.dtype,
            )
            interval_idx = random.randrange(self.student_num_steps)
            latent_t, latent_next, start, end = self._teacher_interval(
                noise.float(), conditional_dict, unconditional_dict, interval_idx)

        t = self._timestep_at(start)
        sigma_t = self._sigma_at(start)
        sigma_next = self._sigma_at(end)
        timestep = t * torch.ones(
            [batch_size, latent_t.shape[1]], device=self.device, dtype=torch.float32)

        flow_pred, _ = self.student(
            latent_t.to(self.dtype),
            conditional_dict,
            timestep,
        )
        pred_next = latent_t.float() + flow_pred.float() * (sigma_next - sigma_t)
        loss = F.mse_loss(pred_next, latent_next.float(), reduction="mean")
        loss.backward()
        grad_norm = self.student.clip_grad_norm_(self.max_grad_norm)
        return {
            "generator_loss": loss.detach(),
            "generator_grad_norm": grad_norm.detach(),
            "interval": torch.tensor(interval_idx, device=self.device),
            "timestep": t.detach(),
        }

    def train(self):
        start_step = self.step
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            log_dict = self.fwdbwd_one_step(next(self.dataloader))
            self.optimizer.step()
            if self.generator_ema is not None:
                self.generator_ema.update(self.student)

            self.step += 1

            if (not self.config.no_save) and self.step > start_step and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.is_main_process and not self.disable_wandb:
                wandb.log({
                    "generator_loss": log_dict["generator_loss"].mean().item(),
                    "generator_grad_norm": log_dict["generator_grad_norm"].mean().item(),
                    "pcd_interval": log_dict["interval"].item(),
                    "pcd_timestep": log_dict["timestep"].item(),
                }, step=self.step)

            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None and not self.disable_wandb:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                self.previous_time = current_time
