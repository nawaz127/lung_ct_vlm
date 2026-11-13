"""Architecture-aware Grad-CAM for Sector 09."""

import torch
import torch.nn.functional as F

TARGET_LAYERS = {
    "mobilenet_v3_large": ("backbone.features", -1),
    "efficientnet_b0": ("backbone.features", -1),
    "resnet18": ("backbone.layer4", -1),
}


def target_layer(model):
    if model.model_name not in TARGET_LAYERS:
        raise ValueError(f"unsupported model architecture: {model.model_name}")
    parent_name, index = TARGET_LAYERS[model.model_name]
    parent = model
    for part in parent_name.split("."):
        parent = getattr(parent, part)
    return parent[index], f"{parent_name}[{index}]"


def grad_cam(model, images, target_class):
    if images.ndim != 4 or not 0 <= int(target_class) < model.num_classes:
        raise ValueError("invalid input shape or target class")
    layer, _ = target_layer(model)
    captured = {}

    def capture(_, __, output):
        captured["feature"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(images)
        logits[:, int(target_class)].sum().backward()
        feature = captured["feature"]
        weights = feature.grad.mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * feature).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        heatmap = heatmap / heatmap.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        if not torch.isfinite(heatmap).all():
            raise FloatingPointError("Grad-CAM produced non-finite values")
        return heatmap.detach(), logits.detach()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
