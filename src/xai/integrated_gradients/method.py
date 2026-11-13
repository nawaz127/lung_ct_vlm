"""Integrated Gradients with explicit numerical quadrature methods."""

import torch


def _quadrature_grid(steps, method, device):
    if method == "left":
        alphas = torch.arange(steps, device=device, dtype=torch.float32) / steps
        weights = torch.full((steps,), 1.0 / steps, device=device, dtype=torch.float32)
    elif method == "trapezoidal":
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
        weights = torch.full((steps + 1,), 1.0 / steps, device=device, dtype=torch.float32)
        weights[[0, -1]] *= 0.5
    elif method == "midpoint":
        alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
        weights = torch.full((steps,), 1.0 / steps, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported integration method: {method}")
    return alphas, weights


def integrated_gradients(
    model, images, target_class, steps=32, baseline=None, return_delta=True,
    integration_method="left", gradient_batch_size=1,
):
    if images.ndim != 4 or steps < 1 or not 0 <= int(target_class) < model.num_classes:
        raise ValueError("invalid input or integration configuration")
    if not isinstance(integration_method, str):
        raise ValueError("integration_method must be a string")
    if gradient_batch_size < 1:
        raise ValueError("gradient_batch_size must be positive")
    if images.dtype != torch.float32:
        images = images.float()
    base = torch.zeros_like(images) if baseline is None else baseline.float()
    if base.shape != images.shape:
        raise ValueError("baseline must match input shape")
    difference = images - base
    total = torch.zeros_like(images)
    model.eval()
    alphas, weights = _quadrature_grid(steps, integration_method, images.device)
    for start in range(0, len(alphas), gradient_batch_size):
        alpha_batch = alphas[start:start + gradient_batch_size]
        weight_batch = weights[start:start + gradient_batch_size]
        if images.shape[0] == 1:
            point = (
                base + alpha_batch[:, None, None, None] * difference
            ).detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            score = model(point)[:, int(target_class)].sum()
            gradients = torch.autograd.grad(score, point)[0]
            total += (gradients.detach() * weight_batch[:, None, None, None]).sum(dim=0, keepdim=True)
        else:
            batch_size, channels, height, width = images.shape
            k = len(alpha_batch)
            point = (
                base[None] + alpha_batch[:, None, None, None, None] * difference[None]
            ).reshape(k * batch_size, channels, height, width).detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            score = model(point)[:, int(target_class)].sum()
            gradients = torch.autograd.grad(score, point)[0].reshape(k, batch_size, channels, height, width)
            total += (gradients.detach() * weight_batch[:, None, None, None, None]).sum(dim=0)
    attribution = difference * total
    with torch.no_grad():
        output_delta = model(images)[:, int(target_class)] - model(base)[:, int(target_class)]
    convergence_delta = output_delta - attribution.flatten(1).sum(1)
    if not torch.isfinite(attribution).all() or not torch.isfinite(convergence_delta).all():
        raise FloatingPointError("Integrated Gradients produced non-finite values")
    return (attribution.detach(), convergence_delta.detach()) if return_delta else attribution.detach()
