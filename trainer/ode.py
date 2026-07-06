import gc
import logging
from utils.dataset import ODERegressionLMDBDataset, cycle
from model import ODERegression
from collections import defaultdict
from utils.misc import (
    set_seed
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job


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
        self.tensorboard_writer = None
        if self.is_main_process and getattr(config, "enable_tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir = getattr(config, "tensorboard_dir", None) or os.path.join(self.output_path, "tensorboard")
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
            self.tensorboard_writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(config)}\n```", 0)
            if getattr(config, "guidance_scale", None) is not None:
                # ODE 回归使用离线 CFG teacher 轨迹；这里把采样 CFG 强度写入本地日志，避免训练记录丢失。
                self.tensorboard_writer.add_scalar("cfg/guidance_scale", float(config.guidance_scale), 0)

        # Step 2: Initialize the model and optimizer

        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegression(config, device=self.device)

        # v2v(方案B): in_channels 仍 16, 不扩通道; 源前缀由 prefix token 机制注入。
        # 因果 generator 的 ODE(_forward_train) 前缀注入已实现(cond_latents -> 源前缀 token),
        # cond_latent 经 conditional_dict 在 train_one_step 中注入(见下方 Step 2)。
        # 该 ODE 阶段为可选的双向->因果适配; DMD 默认直接用 Bernini prefix v2v 权重初始化, 不依赖 ODE。
        self.v2v = getattr(config, "v2v", False)
        if self.v2v:
            # v2v 续训: 从 checkpoint 暖启动 generator(优化器状态不恢复)
            if getattr(config, "resume_ckpt", None):
                print(f"Resuming generator from {config.resume_ckpt}")
                resume_sd = torch.load(config.resume_ckpt, map_location="cpu")["generator"]
                self.model.generator.load_state_dict(resume_sd, strict=True)

        # generator_cpu_offload: 因果版(40GB A100)backward 峰值越界 OOM 时开启,
        # 把 generator 的 params/grads/AdamW 优化器状态常驻 CPU, 显著削减常驻 GPU 显存,
        # 给 backward 激活峰值腾空间(速度有损, init 阶段可接受)。默认 False。
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "generator_cpu_offload", False)
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

        # Step 3: Initialize the dataloader
        dataset = ODERegressionLMDBDataset(
            config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        # num_workers 写死 8 在 8 卡下 = 64 个 dataloader 子进程, 叠加多训练并发易把主机内存吃爆触发 OOM。
        # 改为可配置, 默认降到 2; 配合 prefetch_factor 限制每 worker 预取队列, 避免 anon 内存累积。
        _num_workers = getattr(config, "num_workers", 2)
        _dl_kwargs = dict(
            batch_size=config.batch_size, sampler=sampler, num_workers=_num_workers)
        if _num_workers > 0:
            _dl_kwargs["prefetch_factor"] = getattr(config, "prefetch_factor", 2)
            _dl_kwargs["persistent_workers"] = False
        dataloader = torch.utils.data.DataLoader(dataset, **_dl_kwargs)
        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            assert total_batch_size == config.batch_size * self.world_size, "Gradient accumulation is not supported for ODE training"
        self.dataloader = cycle(dataloader)

        self.step = getattr(config, "ckpt_step", 0)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        # v2v: base 16 通道权重已在 ODERegression.__init__ 中加载, Trainer 再扩通道, 故跳过此处重复加载
        if getattr(config, "generator_ckpt", False) and not self.v2v:
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")[
                'generator']
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        self.max_grad_norm = 10.0
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

    def _save_background_mask_visualization(self, log_dict, step=None):
        if not self.is_main_process or not getattr(self.config, "background_preservation_visualize_masks", False):
            return
        interval = int(getattr(self.config, "background_preservation_mask_visualize_interval", 200))
        step = self.step if step is None else step
        if interval > 0 and step % interval != 0:
            return
        required = (
            "background_preservation_edit_weight",
            "background_preservation_bg_weight",
            "background_preservation_diff",
        )
        if any(key not in log_dict for key in required):
            return

        from PIL import Image, ImageDraw, ImageFont

        vis_dir = getattr(self.config, "background_preservation_mask_visualize_dir", None)
        if not vis_dir:
            vis_dir = os.path.join(self.output_path, "background_preservation_masks")
        os.makedirs(vis_dir, exist_ok=True)

        def to_image(tensor, normalize=False):
            tensor = tensor.detach().float().cpu()
            frame_idx = tensor.shape[1] // 2
            image = tensor[0, frame_idx, 0]
            if normalize:
                image = (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)
            image = image.clamp(0.0, 1.0)
            image = (image * 255.0).to(torch.uint8).numpy()
            return Image.fromarray(image, mode="L").resize((416, 240), Image.BILINEAR).convert("RGB")

        panels = [
            ("edit_weight", to_image(log_dict["background_preservation_edit_weight"])),
            ("bg_weight", to_image(log_dict["background_preservation_bg_weight"])),
            ("diff_norm", to_image(log_dict["background_preservation_diff"], normalize=True)),
        ]
        title_h = 18
        width, height = panels[0][1].size
        canvas = Image.new("RGB", (width * len(panels), height + title_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
        except OSError:
            font = None
        for idx, (title, image) in enumerate(panels):
            x0 = idx * width
            draw.text((x0 + 4, 3), title, fill=(0, 0, 0), font=font)
            canvas.paste(image, (x0, title_h))
        output = os.path.join(vis_dir, f"step_{step:06d}_background_mask.jpg")
        canvas.save(output, quality=92)
        print("Background preservation mask saved to", output)

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self):
        VISUALIZE = self.step % 100 == 0
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(
            device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)
            # v2v: 注入源条件 latent(lmdb 内由 Bernini teacher 离线生成的成对源)
            if self.v2v and "cond_latent" in batch:
                conditional_dict["cond_latent"] = batch["cond_latent"].to(
                    device=self.device, dtype=self.dtype)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict
        )
        self._save_background_mask_visualization(log_dict)

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        flat_loss = gathered_unnormalized_loss.flatten()
        flat_timestep = gathered_timestep.flatten()
        stats.update({
            "loss/unnormalized_mean": flat_loss.mean().item(),
            "loss/unnormalized_std": flat_loss.std(unbiased=False).item(),
            "loss/unnormalized_min": flat_loss.min().item(),
            "loss/unnormalized_max": flat_loss.max().item(),
            "timestep/mean": flat_timestep.float().mean().item(),
            "timestep/min": flat_timestep.min().item(),
            "timestep/max": flat_timestep.max().item(),
        })

        for index, t in enumerate(flat_timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                flat_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss/by_timestep_bucket_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 4: Visualization
        if VISUALIZE and not self.config.no_visualize and not self.config.disable_wandb and self.is_main_process:
            # Visualize the input, output, and ground truth
            input = log_dict["input"]
            output = log_dict["output"]
            ground_truth = ode_latent[:, -1]

            input_video = self.model.vae.decode_to_pixel(input)
            output_video = self.model.vae.decode_to_pixel(output)
            ground_truth_video = self.model.vae.decode_to_pixel(ground_truth)
            input_video = 255.0 * (input_video.cpu().numpy() * 0.5 + 0.5)
            output_video = 255.0 * (output_video.cpu().numpy() * 0.5 + 0.5)
            ground_truth_video = 255.0 * (ground_truth_video.cpu().numpy() * 0.5 + 0.5)

            # Visualize the input, output, and ground truth
            wandb.log({
                "input": wandb.Video(input_video, caption="Input", fps=16, format="mp4"),
                "output": wandb.Video(output_video, caption="Output", fps=16, format="mp4"),
                "ground_truth": wandb.Video(ground_truth_video, caption="Ground Truth", fps=16, format="mp4"),
            }, step=self.step)

        # Step 5: Logging
        train_stats = {
            "train/generator_loss": generator_loss.item(),
            "train/generator_grad_norm": generator_grad_norm.item(),
            "optim/lr": self.generator_optimizer.param_groups[0]["lr"],
            "optim/weight_decay": self.generator_optimizer.param_groups[0]["weight_decay"],
            "cfg/guidance_scale": float(getattr(self.config, "guidance_scale", 0.0)),
            "runtime/cuda_memory_allocated_gb": torch.cuda.memory_allocated(self.device) / (1024 ** 3),
            "runtime/cuda_memory_reserved_gb": torch.cuda.memory_reserved(self.device) / (1024 ** 3),
            "runtime/world_size": self.world_size,
            "runtime/batch_size_per_gpu": self.config.batch_size,
            **stats,
        }
        if "background_preservation_loss" in log_dict:
            train_stats.update({
                "loss/background_preservation": log_dict["background_preservation_loss"].item(),
                "mask/background_preservation_bg_ratio": log_dict["background_preservation_bg_ratio"].item(),
                "mask/background_preservation_edit_ratio": log_dict["background_preservation_edit_ratio"].item(),
            })

        if self.is_main_process and not self.disable_wandb:
            wandb.log(train_stats, step=self.step)

        self._write_tensorboard_scalars(train_stats)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def train(self):
        # 迭代上限 (CausVid ODE init = 3000 iters); 未配置则无限训练直到手动停止
        max_iter = getattr(self.config, "max_iter", None)
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self._write_tensorboard_scalars({
                        "runtime/per_iteration_time": current_time - self.previous_time,
                    })
                    self.previous_time = current_time

            self.step += 1

            # 达到 max_iter 后存最终 checkpoint 并退出
            if max_iter is not None and self.step > max_iter:
                if (not self.config.no_save) and (self.step - 1) % self.config.log_iters != 0:
                    self.save()
                    torch.cuda.empty_cache()
                if self.is_main_process:
                    print(f"Reached max_iter={max_iter}, stopping training.")
                break
