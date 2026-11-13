"""Historical Integrated Gradients implementation used by prior XAI artifacts."""

import torch


def integrated_gradients_legacy(
    model, images, target_class, steps=32, baseline=None, return_delta=True,
):
    if images.ndim != 4 or steps < 1 or not 0 <= int(target_class) < model.num_classes:
        raise ValueError("invalid input or integration configuration")
    base = torch.zeros_like(images) if baseline is None else baseline
    if base.shape != images.shape:
        raise ValueError("baseline must match input shape")
    difference = images - base
    total = torch.zeros_like(images)
    model.eval()
    for alpha in torch.linspace(0, 1, steps, device=images.device):
        point = (base + alpha * difference).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = model(point)[:, int(target_class)].sum()
        gradients = torch.autograd.grad(score, point)[0]
        total += gradients.detach()
    attribution = difference * total / steps
    with torch.no_grad():
        output_delta = model(images)[:, int(target_class)] - model(base)[:, int(target_class)]
    convergence_delta = output_delta - attribution.flatten(1).sum(1)
    if not torch.isfinite(attribution).all() or not torch.isfinite(convergence_delta).all():
        raise FloatingPointError("Integrated Gradients produced non-finite values")
    return (attribution.detach(), convergence_delta.detach()) if return_delta else attribution.detach()
