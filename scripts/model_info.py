"""Inspect a Sector 04 CNN architecture and run a dummy forward pass."""

import argparse
import json

import torch

from models import create_model, model_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet18",
                        choices=["mobilenet_v3_large", "efficientnet_b0", "resnet18"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--head-type", default="gap_dropout_linear",
                        choices=["gap_linear", "gap_dropout_linear"])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model = create_model(args.model, pretrained=args.pretrained,
                         head_type=args.head_type, dropout=args.dropout)
    device = torch.device(args.device)
    model.to(device).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224, device=device))
    result = model_info(model).as_dict()
    result["device"] = str(device)
    result["output_shape"] = list(output.shape)
    result["finite_output"] = bool(torch.isfinite(output).all())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
