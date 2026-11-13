"""Shared Sector 04 model factory and model complexity reporting."""

from dataclasses import asdict, dataclass
from typing import Any

from torch import nn

from models.backbones.torchvision_models import build_backbone
from models.heads.classification import build_head


ALIASES = {"mobilenetv3_large": "mobilenet_v3_large", "efficientnet-b0": "efficientnet_b0"}


class CNNClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, model_name: str, feature_dim: int,
                 num_classes: int, head_type: str, dropout: float, pretrained: bool):
        super().__init__()
        self.backbone = backbone
        self.classifier = build_head(feature_dim, num_classes, head_type, dropout)
        self.model_name = model_name
        self.initialization = "imagenet" if pretrained else "scratch"
        self.head_type = head_type
        self.num_classes = num_classes

    def _features(self, images):
        if self.model_name == "resnet18":
            modules = list(self.backbone.children())[:-2]
            for module in modules:
                images = module(images)
            return images
        return self.backbone.features(images)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError("expected input shape [B, 3, 224, 224]")
        return self.classifier(self._features(images))

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False


@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    initialization: str
    head_type: str
    num_classes: int
    input_shape: tuple[int, int, int]
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    parameter_memory_mb: float
    architecture: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_model(model_name: str, num_classes: int = 3, pretrained: bool = False,
                 head_type: str = "gap_linear", dropout: float = 0.0,
                 freeze_backbone: bool = False) -> CNNClassifier:
    canonical = ALIASES.get(model_name.casefold(), model_name.casefold())
    backbone, feature_dim = build_backbone(canonical, pretrained)
    model = CNNClassifier(backbone, canonical, feature_dim, num_classes, head_type, dropout, pretrained)
    if freeze_backbone:
        model.freeze_backbone()
    return model


def model_info(model: CNNClassifier) -> ModelInfo:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    memory = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024 ** 2)
    return ModelInfo(model.model_name, model.initialization, model.head_type, model.num_classes,
                     (3, 224, 224), total, trainable, total - trainable, round(memory, 4),
                     model.__class__.__name__)