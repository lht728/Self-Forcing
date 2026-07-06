import torch.nn.functional as F
from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.loss import background_preservation_loss


class ODERegression(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__(args, device)

        # Step 1: Initialize all models

        # generator_causal=False -> 双向(非因果)学生, 用于"不因果化"ODE 回归(复用双向 teacher 采样的 LMDB)。
        self.generator_causal = getattr(args, "generator_causal", True)
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=self.generator_causal)
        self.generator.model.requires_grad_(True)
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
                'generator']
            if getattr(args, "v2v", False):
                # v2v(方案B) base 为 Bernini 原生前缀 v2v 16通道权重(convert --no_expand 产出, 裸 WanModel 键),
                # 直接载入 self.generator.model; freqs/visual_id_freqs 为非持久缓冲, 容忍缺失。
                missing, unexpected = self.generator.model.load_state_dict(state_dict, strict=False)
                missing = [m for m in missing if not m.endswith(".freqs") and not m.endswith(".visual_id_freqs")]
                assert not missing and not unexpected, \
                    f"v2v base 载入不匹配 missing={missing[:5]} unexpected={unexpected[:5]}"
            else:
                self.generator.load_state_dict(
                    state_dict, strict=True
                )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.background_preservation_weight = float(getattr(args, "background_preservation_weight", 0.0))
        self.background_preservation_mask_quantile = float(
            getattr(args, "background_preservation_mask_quantile", 0.85))
        self.background_preservation_mask_threshold = float(
            getattr(args, "background_preservation_mask_threshold", 0.0))
        self.background_preservation_dilation = int(getattr(args, "background_preservation_dilation", 3))
        self.background_preservation_temporal_smoothing = getattr(
            args, "background_preservation_temporal_smoothing", "none")
        self.background_preservation_temporal_kernel = int(
            getattr(args, "background_preservation_temporal_kernel", 3))
        self.background_preservation_soft_mask = bool(
            getattr(args, "background_preservation_soft_mask", False))
        self.background_preservation_soft_temperature = float(
            getattr(args, "background_preservation_soft_temperature", 0.01))
        self.background_preservation_edit_ratio_min = float(
            getattr(args, "background_preservation_edit_ratio_min", 0.0))
        self.background_preservation_edit_ratio_max = float(
            getattr(args, "background_preservation_edit_ratio_max", 1.0))
        self.background_preservation_visualize_masks = bool(
            getattr(args, "background_preservation_visualize_masks", False))

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=getattr(args, "generator_causal", True))
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        # 双向(非因果)学生所有帧共用同一 timestep, 与 wrapper.uniform_timestep 一致, 保证 gather 的 noisy 帧一致。
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=not self.generator_causal
        )
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        timestep = self.denoising_step_list.to(index.device)[index].to(self.device)

        # if self.extra_noise_step > 0:
        #     random_timestep = torch.randint(0, self.extra_noise_step, [
        #                                     batch_size, num_frames], device=self.device, dtype=torch.long)
        #     perturbed_noisy_input = self.scheduler.add_noise(
        #         noisy_input.flatten(0, 1),
        #         torch.randn_like(noisy_input.flatten(0, 1)),
        #         random_timestep.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frames)).type_as(noisy_input)

        #     noisy_input[timestep == 0] = perturbed_noisy_input[timestep == 0]

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)

        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        if self.background_preservation_weight > 0 and "cond_latent" in conditional_dict:
            bg_result = background_preservation_loss(
                pred=pred_image_or_video,
                source=conditional_dict["cond_latent"],
                target=target_latent,
                mask_quantile=self.background_preservation_mask_quantile,
                mask_threshold=self.background_preservation_mask_threshold,
                dilation=self.background_preservation_dilation,
                temporal_smoothing=self.background_preservation_temporal_smoothing,
                temporal_kernel=self.background_preservation_temporal_kernel,
                soft_mask=self.background_preservation_soft_mask,
                soft_temperature=self.background_preservation_soft_temperature,
                edit_ratio_min=self.background_preservation_edit_ratio_min,
                edit_ratio_max=self.background_preservation_edit_ratio_max,
                return_details=self.background_preservation_visualize_masks,
            )
            bg_loss, bg_ratio, edit_ratio = bg_result[:3]
            loss = loss + self.background_preservation_weight * bg_loss
            log_dict.update({
                "background_preservation_loss": bg_loss.detach(),
                "background_preservation_bg_ratio": bg_ratio,
                "background_preservation_edit_ratio": edit_ratio,
            })
            if self.background_preservation_visualize_masks:
                log_dict.update(bg_result[3])

        return loss, log_dict
