from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class DenoisingLoss(ABC):
    @abstractmethod
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Base class for denoising loss.
        Input:
            - x: the clean data with shape [B, F, C, H, W]
            - x_pred: the predicted clean data with shape [B, F, C, H, W]
            - noise: the noise with shape [B, F, C, H, W]
            - noise_pred: the predicted noise with shape [B, F, C, H, W]
            - alphas_cumprod: the cumulative product of alphas (defining the noise schedule) with shape [T]
            - timestep: the current timestep with shape [B, F]
        """
        pass


class X0PredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((x - x_pred) ** 2)


class VPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        weights = 1 / (1 - alphas_cumprod[timestep].reshape(*timestep.shape, 1, 1, 1))
        return torch.mean(weights * (x - x_pred) ** 2)


class NoisePredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((noise - noise_pred) ** 2)


class FlowPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((kwargs["flow_pred"] - (noise - x)) ** 2)


NAME_TO_CLASS = {
    "x0": X0PredLoss,
    "v": VPredLoss,
    "noise": NoisePredLoss,
    "flow": FlowPredLoss
}


def get_denoising_loss(loss_type: str) -> DenoisingLoss:
    return NAME_TO_CLASS[loss_type]


def _temporal_smooth_mask_score(
    score: torch.Tensor,
    mode: str = "none",
    kernel_size: int = 3,
) -> torch.Tensor:
    mode = str(mode or "none").lower()
    kernel_size = int(kernel_size or 1)
    if mode in ("none", "off", "false") or kernel_size <= 1 or score.shape[1] <= 1:
        return score
    if mode not in ("max", "median"):
        raise ValueError(f"Unsupported background_preservation_temporal_smoothing={mode}")
    if kernel_size % 2 == 0:
        kernel_size += 1

    bsz, frames, channels, height, width = score.shape
    assert channels == 1, "background preservation mask score must have one channel"
    pad = kernel_size // 2
    x = score.squeeze(2).permute(0, 2, 3, 1).reshape(-1, 1, frames).float()
    if mode == "max":
        y = F.max_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)
    else:
        y = F.pad(x, (pad, pad), mode="replicate").unfold(2, kernel_size, 1).median(dim=-1).values
    return y.reshape(bsz, height, width, frames).permute(0, 3, 1, 2).unsqueeze(2).to(score.dtype)


def _spatial_dilate_mask_score(score: torch.Tensor, dilation: int) -> torch.Tensor:
    dilation = int(dilation or 0)
    if dilation <= 1:
        return score
    if dilation % 2 == 0:
        dilation += 1
    pad = dilation // 2
    score_2d = score.flatten(0, 1).float()
    score_2d = F.max_pool2d(score_2d, kernel_size=dilation, stride=1, padding=pad)
    return score_2d.unflatten(0, score.shape[:2]).to(score.dtype)


def _clamp_hard_edit_ratio(
    edit_mask: torch.Tensor,
    priority: torch.Tensor,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    ratio_min = max(0.0, min(1.0, float(ratio_min)))
    ratio_max = max(ratio_min, min(1.0, float(ratio_max)))
    if ratio_min <= 0.0 and ratio_max >= 1.0:
        return edit_mask

    bsz, frames, _, height, width = edit_mask.shape
    numel = height * width
    mask_flat = edit_mask.flatten(start_dim=2)
    score_flat = priority.flatten(start_dim=2).float()
    ratio = mask_flat.float().mean(dim=2)
    out = mask_flat.clone()

    for bidx in range(bsz):
        for tidx in range(frames):
            target_k = None
            if ratio[bidx, tidx] > ratio_max:
                target_k = int(round(ratio_max * numel))
            elif ratio[bidx, tidx] < ratio_min:
                target_k = int(round(ratio_min * numel))
            if target_k is None:
                continue
            out[bidx, tidx].zero_()
            if target_k > 0:
                target_k = min(target_k, numel)
                indices = torch.topk(score_flat[bidx, tidx], k=target_k, largest=True).indices
                out[bidx, tidx, indices] = True

    return out.reshape_as(edit_mask)


def _clamp_soft_edit_ratio(
    edit_weight: torch.Tensor,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    ratio_min = max(0.0, min(1.0, float(ratio_min)))
    ratio_max = max(ratio_min, min(1.0, float(ratio_max)))
    if ratio_min <= 0.0 and ratio_max >= 1.0:
        return edit_weight

    edit_weight = edit_weight.float()
    mean = edit_weight.mean(dim=(2, 3, 4), keepdim=True)
    if ratio_max < 1.0:
        scale = torch.where(mean > ratio_max, ratio_max / mean.clamp_min(1e-6), torch.ones_like(mean))
        edit_weight = edit_weight * scale
        mean = edit_weight.mean(dim=(2, 3, 4), keepdim=True)
    if ratio_min > 0.0:
        alpha = (ratio_min - mean) / (1.0 - mean).clamp_min(1e-6)
        alpha = torch.where(mean < ratio_min, alpha.clamp(0.0, 1.0), torch.zeros_like(mean))
        edit_weight = edit_weight + (1.0 - edit_weight) * alpha
    return edit_weight.clamp(0.0, 1.0)


def background_preservation_loss(
    pred: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    mask_quantile: float = 0.85,
    mask_threshold: float = 0.0,
    dilation: int = 3,
    temporal_smoothing: str = "none",
    temporal_kernel: int = 3,
    soft_mask: bool = False,
    soft_temperature: float = 0.01,
    edit_ratio_min: float = 0.0,
    edit_ratio_max: float = 1.0,
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep non-edited regions close to source using a pseudo edit mask from target-source latent diff."""
    source = source.detach().to(device=pred.device, dtype=pred.dtype)
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    diff = (target - source).abs().mean(dim=2, keepdim=True)

    flat = diff.flatten(start_dim=2)
    threshold = torch.quantile(flat.float(), float(mask_quantile), dim=2, keepdim=True).to(diff.dtype)
    threshold = threshold.reshape(diff.shape[0], diff.shape[1], 1, 1, 1)
    if mask_threshold > 0:
        threshold = torch.maximum(threshold, torch.as_tensor(mask_threshold, device=diff.device, dtype=diff.dtype))

    if soft_mask:
        temperature = max(float(soft_temperature), 1e-6)
        edit_weight = torch.sigmoid((diff.float() - threshold.float()) / temperature).to(diff.dtype)
        edit_weight = _temporal_smooth_mask_score(edit_weight, temporal_smoothing, temporal_kernel)
        edit_weight = _spatial_dilate_mask_score(edit_weight, dilation)
        edit_weight = _clamp_soft_edit_ratio(edit_weight, edit_ratio_min, edit_ratio_max).to(diff.dtype)
    else:
        edit_mask = diff > threshold
        edit_score = _temporal_smooth_mask_score(edit_mask.float(), temporal_smoothing, temporal_kernel)
        edit_score = _spatial_dilate_mask_score(edit_score, dilation)
        edit_mask = edit_score > 0.5
        edit_mask = _clamp_hard_edit_ratio(edit_mask, diff, edit_ratio_min, edit_ratio_max)
        edit_weight = edit_mask.to(diff.dtype)

    bg_weight = (1.0 - edit_weight.float()).to(pred.dtype)
    bg_weight = bg_weight.expand_as(pred)
    denom = bg_weight.float().sum().clamp_min(1.0)
    loss = ((pred.float() - source.float()).pow(2) * bg_weight.float()).sum() / denom
    bg_ratio = bg_weight.float().mean().detach()
    edit_ratio = edit_weight.float().mean().detach()

    if return_details:
        return loss, bg_ratio, edit_ratio, {
            "background_preservation_edit_weight": edit_weight.detach(),
            "background_preservation_bg_weight": (1.0 - edit_weight.float()).detach(),
            "background_preservation_diff": diff.detach(),
        }
    return loss, bg_ratio, edit_ratio
