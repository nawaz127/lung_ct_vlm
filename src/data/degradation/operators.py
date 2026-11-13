"""Pixel-domain image quality operators for Sector 06."""

import torch
import torch.nn.functional as F


def _validate(image):
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected an RGB image with shape [3, H, W]")
    if not torch.is_floating_point(image):
        raise TypeError("image must be a floating-point tensor")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite and in [0, 1]")


def gaussian_noise(image, std, generator=None):
    _validate(image)
    if std <= 0:
        raise ValueError("noise standard deviation must be positive")
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
    return (image + noise * float(std)).clamp(0, 1)


def _kernel_size(value):
    size = int(value)
    if size < 3 or size % 2 == 0:
        raise ValueError("blur kernel_size must be an odd integer >= 3")
    return size


def gaussian_blur(image, sigma, kernel_size):
    _validate(image)
    if sigma <= 0:
        raise ValueError("blur sigma must be positive")
    size = _kernel_size(kernel_size)
    coordinates = torch.arange(size, device=image.device, dtype=image.dtype) - size // 2
    kernel_1d = torch.exp(-(coordinates ** 2) / (2 * float(sigma) ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(3, 1, size, size)
    padded = F.pad(image.unsqueeze(0), (size // 2,) * 4, mode="reflect")
    return F.conv2d(padded, kernel, groups=3).squeeze(0).clamp(0, 1)


def reduce_contrast(image, factor):
    _validate(image)
    if not 0 < factor <= 1:
        raise ValueError("contrast factor must be in (0, 1]")
    mean = image.mean(dim=(1, 2), keepdim=True)
    return ((image - mean) * float(factor) + mean).clamp(0, 1)
