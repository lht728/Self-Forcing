import torch
import torch.distributed as dist
from typing import Tuple

from model.dmd import DMD


class BidirectionalDMD(DMD):
    """双向(非因果) few-step DMD 蒸馏。

    复用 DMD 的 real/fake score、KL 梯度与 critic 逻辑, 仅把"学生生成"从因果
    self-forcing 流式 rollout(需 KV cache)替换为整段双向多步去噪 rollout:
    从纯噪声出发逐步去噪, 仅在随机选中的退出步保留梯度(DMD2 backward simulation)。

    源条件(v2v)通过 conditional_dict['cond_latent'] 走 wrapper 的前缀 token 注入,
    与双向 real/fake score teacher 接口完全一致。

    原因果 DMD(model/dmd.py)保持不变, 二者可在不同 GPU 并行训练。
    """

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        device, dtype = self.device, self.dtype
        denoising_step_list = self.denoising_step_list.to(device)
        num_steps = len(denoising_step_list)

        # 选退出步(全 rank 同步): 该步带梯度, 之前的步无梯度仅用于模拟学生输入
        if dist.is_initialized():
            exit_idx = torch.randint(0, num_steps, (1,), device=device)
            dist.broadcast(exit_idx, src=0)
            exit_idx = int(exit_idx.item())
        else:
            exit_idx = int(torch.randint(0, num_steps, (1,)).item())

        batch_size, num_frame = image_or_video_shape[0], image_or_video_shape[1]
        noisy = torch.randn(image_or_video_shape, device=device, dtype=dtype)

        denoised_pred = None
        for index in range(num_steps):
            current_timestep = denoising_step_list[index]
            timestep = torch.ones(
                [batch_size, num_frame], device=device, dtype=torch.int64) * current_timestep

            if index < exit_idx:
                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                    )
                next_timestep = denoising_step_list[index + 1]
                noisy = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size * num_frame], device=device, dtype=torch.long)
                ).unflatten(0, (batch_size, num_frame))
            else:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                break

        # denoised_timestep_from/to: 与 self-forcing pipeline 口径一致(供 ts_schedule 用)
        timesteps = self.scheduler.timesteps.to(device)
        if exit_idx == num_steps - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (timesteps - denoising_step_list[exit_idx]).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (timesteps - denoising_step_list[exit_idx + 1]).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (timesteps - denoising_step_list[exit_idx]).abs(), dim=0).item()

        gradient_mask = None
        return denoised_pred.type_as(noisy), gradient_mask, denoised_timestep_from, denoised_timestep_to
