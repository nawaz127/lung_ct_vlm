"""Attribution heatmap and overlay utilities."""

import torch


def normalize_heatmap(heatmap):
    value = torch.as_tensor(heatmap, dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("heatmap must be finite")
    span = value.max() - value.min()
    return (value - value.min()) / span.clamp_min(1e-12)


def overlay_heatmap(image, heatmap, alpha=0.45):
    image = torch.as_tensor(image, dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W]")
    heatmap = normalize_heatmap(heatmap)
    if heatmap.shape != image.shape[-2:]:
        raise ValueError("heatmap and image dimensions must match")
    color = torch.stack((heatmap, torch.zeros_like(heatmap), 1 - heatmap))
    return ((1 - alpha) * image.clamp(0, 1) + alpha * color).clamp(0, 1)
