"""Reusable classification heads for convolutional feature maps."""

import torch
from torch import nn


def build_head(in_features: int, num_classes: int, head_type: str = "gap_linear",
               dropout: float = 0.0) -> nn.Module:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if head_type not in {"gap_linear", "gap_dropout_linear"}:
        raise ValueError("head_type must be gap_linear or gap_dropout_linear")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    layers = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
    if head_type == "gap_dropout_linear":
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)