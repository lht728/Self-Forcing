import gc
import logging
import os
import time

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from model import NaiveConsistency
from utils.dataset import V2VPairedVideoDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from utils.misc import set_seed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = getattr(config, "ckpt_step", 0)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.v2v = getattr(config, "v2v", False)

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

        assert config.distribution_loss in ("causal_cd", "consistency_distillation"), (
            "Causal CD trainer 需 distribution_loss: causal_cd"
        )
        self.model = NaiveConsistency(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True,
        )
        self.model.generator_ema = fsdp_wrap(
            self.model.generator_ema,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True,
        )
        self.model.teacher = fsdp_wrap(
            self.model.teacher,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=True,
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True,
        )

        if self.v2v or getattr(config, "load_raw_video", False):
            self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        ema_weight = getattr(config, "ema_weight", 0.99)
        self.generator_ema = None
        if ema_weight is not None and ema_weight > 0.0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        if self.v2v:
            dataset = V2VPairedVideoDataset(
                config.data_path,
                base_video_folder=getattr(config, "base_video_folder", ""),
                prompt_key=getattr(config, "prompt_key", "instruction_final_refine"),
                src_key=getattr(config, "src_key", "src_video"),
                tar_key=getattr(config, "tar_key", "tar_video"),
                num_frames=getattr(config, "num_raw_frames", 81),
                target_fps=getattr(config, "target_fps", 16),
                height=getattr(config, "height", 480),
                width=getattr(config, "width", 832),
            )
        else:
            raise NotImplementedError(
                "当前仅实现 v2v 路线B 的 Causal CD; T2V 请用 LatentLMDBDataset 扩展"
            )

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        if dist.get_rank() == 0:
            print(f"[Causal CD] dataset size {len(dataset)}")
        self.dataloader = cycle(dataloader)

        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            assert total_batch_size == config.batch_size * self.world_size, (
                "Causal CD 不支持梯度累积"
            )

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.previous_time = None

    def save(self):
        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            state_dict = {
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": fsdp_state_dict(self.model.generator),
            }

        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            out_path = os.path.join(ckpt_dir, "model.pt")
            torch.save(state_dict, out_path)
            print(f"Model saved to {out_path}")

    def fwdbwd_one_step(self, batch):
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            if self.v2v:
                src_video = batch["src_video"].to(device=self.device, dtype=self.dtype)
                tar_video = batch["tar_video"].to(device=self.device, dtype=self.dtype)
                clean_latent = self.model.vae.encode_to_latent(tar_video).to(self.dtype)
                cond_latent = self.model.vae.encode_to_latent(src_video).to(self.dtype)
                conditional_dict = {**conditional_dict, "cond_latent": cond_latent}
                unconditional_dict = {**unconditional_dict, "cond_latent": cond_latent}
            else:
                clean_latent = batch["clean_latent"].to(self.device, dtype=self.dtype)

        generator_loss, generator_log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            ema_model=self.generator_ema,
        )
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator)
        generator_log_dict.update({
            "generator_loss": generator_loss,
            "generator_grad_norm": generator_grad_norm,
        })
        return generator_log_dict

    def train(self):
        start_step = self.step
        while True:
            self.generator_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            generator_log_dict = self.fwdbwd_one_step(batch)
            self.generator_optimizer.step()
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)

            self.step += 1

            if (not self.config.no_save) and self.step > start_step and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.is_main_process and not self.disable_wandb:
                wandb.log({
                    "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                    "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                    "cd_timestep": generator_log_dict["cd_timestep"].float().item(),
                }, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None and not self.disable_wandb:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                self.previous_time = current_time
