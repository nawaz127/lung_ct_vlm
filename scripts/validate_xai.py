"""Run small synthetic Sector 09 validation without research data."""

import argparse

import torch

from models import create_model
from xai import grad_cam, integrated_gradients, paired_explanation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    image = torch.randn(1, 3, 224, 224)
    row = {"source_relative_path": "synthetic/sample.JPG", "class_index": "0"}
    for architecture in ("mobilenet_v3_large", "efficientnet_b0", "resnet18"):
        model = create_model(architecture, pretrained=False, head_type="gap_linear")
        heatmap, logits = grad_cam(model, image.clone(), 1)
        attribution, delta = integrated_gradients(model, image.clone(), 1, args.steps)
        result = paired_explanation(model, image, image + 0.01, row, "synthetic_s1",
                                    "integrated_gradients", ig_steps=args.steps)
        print(architecture, list(heatmap.shape), list(attribution.shape),
              list(logits.shape), list(delta.shape), result["target_class"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
