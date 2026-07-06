import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.dataset import V2VVideoDataset
from utils.dataset import V2VPairedVideoDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, fsdp_optim_state_dict, fsdp_load_optim_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, BidirectionalDMD, SiD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
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
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # 本地 tensorboard(可选, enable_tensorboard 开关, 默认关; 与 wandb 互不影响)
        self.tensorboard_writer = None
        if self.is_main_process and getattr(config, "enable_tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = getattr(config, "tensorboard_dir", None) or os.path.join(self.output_path, "tensorboard")
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
            self.tensorboard_writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(config)}\n```", 0)
            if getattr(config, "guidance_scale", None) is not None:
                self.tensorboard_writer.add_scalar("cfg/guidance_scale", float(config.guidance_scale), 0)

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "bidirectional_dmd":
            self.model = BidirectionalDMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # v2v(方案B): 复用 Bernini 原生前缀 token v2v 能力, in_channels 仍 16, 不扩通道。
        # real_score(teacher,冻结)/fake_score/generator 统一用 Bernini v2v 16通道权重初始化,
        # 源视频走 source_id 旋转的前缀 token 注入(见 model/causal_model)。
        self.v2v = getattr(config, "v2v", False)
        if self.v2v:
            self.condition_dropout = getattr(config, "condition_dropout", 0.0)

            def _load_prefix_v2v(module, ckpt_path, name):
                sd = torch.load(ckpt_path, map_location="cpu")
                sd = sd.get("generator", sd)
                missing, unexpected = module.model.load_state_dict(sd, strict=False)
                missing = [m for m in missing if not m.endswith(".freqs") and not m.endswith(".visual_id_freqs")]
                assert not missing and not unexpected, \
                    f"{name} 载入 prefix v2v 权重不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"
                if self.is_main_process:
                    print(f"[v2v] {name} 已载入 Bernini prefix v2v 权重: {ckpt_path}")

            teacher_ckpt = getattr(config, "real_score_v2v_ckpt", None)
            if teacher_ckpt:
                _load_prefix_v2v(self.model.real_score, teacher_ckpt, "real_score")
                fake_ckpt = getattr(config, "fake_score_v2v_ckpt", teacher_ckpt)
                _load_prefix_v2v(self.model.fake_score, fake_ckpt, "fake_score")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        elif getattr(config, "v2v", False):
            use_bg_preservation = (
                config.distribution_loss in ("dmd", "bidirectional_dmd")
                and float(getattr(config, "background_preservation_weight", 0.0)) > 0
            )
            dataset_cls = V2VPairedVideoDataset if use_bg_preservation else V2VVideoDataset
            dataset = dataset_cls(
                config.data_path,
                base_video_folder=config.base_video_folder,
                num_frames=getattr(config, "num_raw_frames", 81),
                height=config.height,
                width=config.width,
                prompt_key=getattr(config, "prompt_key", "instruction_final_refine"),
                src_key=getattr(config, "src_key", "src_video"),
                target_fps=getattr(config, "target_fps", 16),
            )
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            full_state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            gen_sd = full_state_dict
            if isinstance(full_state_dict, dict) and "generator" in full_state_dict:
                gen_sd = full_state_dict["generator"]
            elif isinstance(full_state_dict, dict) and "model" in full_state_dict:
                gen_sd = full_state_dict["model"]
            # v2v(方案B): bernini_v2v_prefix_init.pt 是裸 WanModel 键(无 'model.' 前缀),
            # 而 generator(WanDiffusionWrapper) 期望 'model.' 前缀; 缺前缀则补上以对齐。
            if gen_sd and not next(iter(gen_sd)).startswith("model."):
                gen_sd = {f"model.{k}": v for k, v in gen_sd.items()}
            self.model.generator.load_state_dict(
                gen_sd, strict=True
            )

            # 续训: checkpoint 含 critic/ema 时一并恢复, 并从目录名续上 step,
            # 否则判别器(fake_score)、EMA、计步都会从头开始, 浪费已蒸馏进度。
            if isinstance(full_state_dict, dict) and "critic" in full_state_dict:
                self.model.fake_score.load_state_dict(full_state_dict["critic"], strict=False)
                if self.is_main_process:
                    print("[resume] critic(fake_score) 已从 checkpoint 恢复")
            if isinstance(full_state_dict, dict) and "generator_ema" in full_state_dict \
                    and self.generator_ema is not None:
                self.generator_ema.load_state_dict(full_state_dict["generator_ema"])
                if self.is_main_process:
                    print("[resume] generator_ema 已从 checkpoint 恢复")
            # 无损续训: 恢复优化器动量(AdamW 一阶/二阶矩), 否则续训优化器冷启动会抖动。
            # 必须所有 rank 同时调用(FSDP 集合通信)。
            if isinstance(full_state_dict, dict) and "generator_optimizer" in full_state_dict:
                fsdp_load_optim_state_dict(
                    self.model.generator, self.generator_optimizer,
                    full_state_dict["generator_optimizer"])
                if self.is_main_process:
                    print("[resume] generator_optimizer 已从 checkpoint 恢复")
            if isinstance(full_state_dict, dict) and "critic_optimizer" in full_state_dict:
                fsdp_load_optim_state_dict(
                    self.model.fake_score, self.critic_optimizer,
                    full_state_dict["critic_optimizer"])
                if self.is_main_process:
                    print("[resume] critic_optimizer 已从 checkpoint 恢复")
            import re as _re
            _m = _re.search(r"checkpoint_model_(\d+)", str(config.generator_ckpt))
            if _m:
                self.step = int(_m.group(1))
                if self.is_main_process:
                    print(f"[resume] 续训起始 step={self.step}")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def _write_tensorboard_scalars(self, scalars, step=None):
        if self.tensorboard_writer is None:
            return
        step = self.step if step is None else step
        for key, value in scalars.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            elif not isinstance(value, (int, float)):
                continue
            self.tensorboard_writer.add_scalar(key, value, step)
        self.tensorboard_writer.flush()

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)
        # 无损续训: 聚合优化器分片状态(全部 rank 参与集合通信, rank0 得到全量)
        generator_optim_state_dict = fsdp_optim_state_dict(
            self.model.generator, self.generator_optimizer)
        critic_optim_state_dict = fsdp_optim_state_dict(
            self.model.fake_score, self.critic_optimizer)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
                "generator_optimizer": generator_optim_state_dict,
                "critic_optimizer": critic_optim_state_dict,
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_optimizer": generator_optim_state_dict,
                "critic_optimizer": critic_optim_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            # 无损 checkpoint 含 optimizer 体积大(~38G), 自动只保留最近 N 个, 防磁盘写满崩溃。
            keep_last = getattr(self.config, "checkpoint_keep_last", 3)
            if keep_last and keep_last > 0:
                import glob as _glob
                import re as _re
                import shutil as _shutil
                ckpt_dirs = [d for d in _glob.glob(os.path.join(self.output_path, "checkpoint_model_*"))
                             if os.path.isdir(d)]

                def _step_of(p):
                    m = _re.search(r"checkpoint_model_(\d+)", p)
                    return int(m.group(1)) if m else -1
                ckpt_dirs = sorted(ckpt_dirs, key=_step_of)
                for old in ckpt_dirs[:-keep_last]:
                    _shutil.rmtree(old, ignore_errors=True)
                    print(f"[cleanup] 删除旧 checkpoint(仅保留最近 {keep_last} 个): {old}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            # v2v: 源视频 -> VAE 编码为条件 latent，注入 cond / uncond(源在两侧都保留, CFG 只作用于文本)
            if self.v2v:
                src_video = batch["src_video"].to(device=self.device, dtype=self.dtype)
                cond_latent = self.model.vae.encode_to_latent(src_video).to(self.dtype)
                target_latent = None
                if "tar_video" in batch:
                    tar_video = batch["tar_video"].to(device=self.device, dtype=self.dtype)
                    target_latent = self.model.vae.encode_to_latent(tar_video).to(self.dtype)
                # condition dropout: 按样本概率整体置零, 防止 4 步少步生成直接拷贝源(欠编辑)
                if self.condition_dropout > 0 and torch.rand(1).item() < self.condition_dropout:
                    cond_latent = torch.zeros_like(cond_latent)
                conditional_dict = {**conditional_dict, "cond_latent": cond_latent}
                unconditional_dict = {**unconditional_dict, "cond_latent": cond_latent}
            else:
                cond_latent = None
                target_latent = None

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss_kwargs = dict(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
            )
            if self.config.distribution_loss in ("dmd", "bidirectional_dmd"):
                generator_loss_kwargs.update({
                    "preservation_source_latent": cond_latent,
                    "preservation_target_latent": target_latent,
                })
            generator_loss, generator_log_dict = self.model.generator_loss(**generator_loss_kwargs)

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update({
                        "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                        "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                        "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                    })
                    if "background_preservation_loss" in generator_log_dict:
                        wandb_loss_dict.update({
                            "loss/background_preservation": generator_log_dict[
                                "background_preservation_loss"].mean().item(),
                            "mask/background_preservation_bg_ratio": generator_log_dict[
                                "background_preservation_bg_ratio"].mean().item(),
                            "mask/background_preservation_edit_ratio": generator_log_dict[
                                "background_preservation_edit_ratio"].mean().item(),
                        })

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                tb_dict = dict(wandb_loss_dict)
                tb_dict["cfg/guidance_scale"] = float(getattr(self.config, "guidance_scale", 0.0))
                self._write_tensorboard_scalars(tb_dict)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
