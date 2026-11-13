"""Torchvision backbone construction for Sector 04."""

from collections.abc import Callable

from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
)


BACKBONES: dict[str, tuple[Callable, object, int]] = {
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT, 960),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
}


def build_backbone(model_name: str, pretrained: bool = False):
    if model_name not in BACKBONES:
        raise ValueError(f"Unsupported model: {model_name}")
    constructor, weights, feature_dim = BACKBONES[model_name]
    return constructor(weights=weights if pretrained else None), feature_dim