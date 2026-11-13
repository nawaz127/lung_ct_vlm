"""Numerically safe explanation stability metrics."""

import torch


def _vector(attribution):
    value = torch.as_tensor(attribution, dtype=torch.float64)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 3:
        value = value[0] if value.shape[0] == 1 else value.abs().mean(dim=0)
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError("attribution must be a finite 2-D map")
    return value.flatten()


def _ranks(value):
    order = torch.argsort(value)
    rank = torch.empty_like(order, dtype=torch.float64)
    rank[order] = torch.arange(len(value), dtype=torch.float64, device=value.device)
    return rank


def cosine_similarity(first, second):
    a, b = _vector(first), _vector(second)
    if a.shape != b.shape:
        raise ValueError("attribution maps must have equal dimensions")
    denominator = a.norm() * b.norm()
    if denominator == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.dot(a, b) / denominator)


def ssim(first, second, data_range=1.0):
    a, b = _vector(first), _vector(second)
    if a.shape != b.shape:
        raise ValueError("attribution maps must have equal dimensions")
    def normalize(value):
        span = value.max() - value.min()
        return (value - value.min()) / span if span > 0 else torch.zeros_like(value)
    a, b = normalize(a), normalize(b)
    if torch.equal(a, b):
        return 1.0
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(unbiased=False), b.var(unbiased=False)
    covariance = ((a - ma) * (b - mb)).mean()
    value = ((2 * ma * mb + c1) * (2 * covariance + c2) /
             ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))
    return float(value) if torch.isfinite(value) else 0.0


def spearman_rank_correlation(first, second):
    a, b = _vector(first), _vector(second)
    a, b = _ranks(a), _ranks(b)
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    if denominator == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.dot(a, b) / denominator)
