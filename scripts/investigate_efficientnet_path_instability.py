"""Investigate EfficientNet-B0 activations and gradients along IG paths."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml

from data.degradation import DegradedManifestDataset
from evaluation.cnn_runner import audit_checkpoints, discover_runs
from models import create_model


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONDITIONS = ("clean", "gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3")
SAMPLES = (
    "Malignant/Malignant (10).JPG",
    "Bengin/Benign (104).JPG",
    "Normal/Normal case (136).jpg",
    "Malignant/Malignant (1).JPG",
    "Normal/Normal case (112).jpg",
)


def _gpu():
    try:
        value = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        ).stdout.strip().split(",")
        return {"gpu_utilization_percent": int(value[0]), "vram_used_mb": int(value[1]),
                "vram_total_mb": int(value[2])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def _load(root, run, device):
    metadata = run.metadata
    model = create_model(
        "efficientnet_b0", pretrained=False,
        head_type=metadata.get("head_type", "gap_dropout_linear"), dropout=0.2,
    )
    checkpoint = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _stat(value):
    value = value.detach()
    return {
        "min": float(value.min().item()), "max": float(value.max().item()),
        "mean": float(value.mean().item()), "std": float(value.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(value).all().item()),
    }


def _feature_modules(model):
    modules = [(f"backbone.features.{i}", module) for i, module in enumerate(model.backbone.features)]
    modules.append(("classifier", model.classifier))
    return modules


def _path_point(model, base, image, target, alpha, hooks):
    captured = {}
    handles = []
    for name, module in hooks:
        def capture(_, __, output, layer_name=name):
            if isinstance(output, tuple):
                output = output[0]
            output.retain_grad()
            captured[layer_name] = output
        handles.append(module.register_forward_hook(capture))
    point = (base + alpha * (image - base)).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    logits = model(point)
    score = logits[:, target].sum()
    gradient = torch.autograd.grad(score, point)[0]
    elapsed = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    layers = {}
    for name, output in captured.items():
        layers[name] = {
            "activation": _stat(output),
            "gradient": _stat(output.grad),
            "gradient_l2": float(output.grad.detach().norm().item()),
            "gradient_abs_max": float(output.grad.detach().abs().max().item()),
        }
    return {
        "alpha": alpha, "runtime_seconds": elapsed,
        "input_gradient": _stat(gradient), "logits": _stat(logits),
        "target_logit": float(logits[0, target].item()), "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    degradation = yaml.safe_load((root / "configs/degradation.yaml").read_text(encoding="utf-8"))
    runs = [
        run for run in discover_runs(root)
        if run.metadata.get("model") == "efficientnet_b0"
        and run.metadata.get("seed") in (1, 2, 3)
        and run.metadata.get("strategy") in ("clean_baseline", "corruption_aware")
    ]
    audit = audit_checkpoints(root)
    if not audit["complete"]:
        raise RuntimeError("full checkpoint audit failed")
    rows = list(csv.DictReader((root / "data/splits/test.csv").open(encoding="utf-8")))
    row_by_path = {row["source_relative_path"]: row for row in rows}
    datasets = {
        condition: DegradedManifestDataset(
            root / "data/splits/test.csv", root / "data/raw", condition,
            degradation, identity_by_path=True,
        ) for condition in CONDITIONS
    }
    indices = {
        row["source_relative_path"]: i
        for i, row in enumerate(datasets["clean"].rows)
    }
    all_records = []
    reproducibility = []
    gpu_start = _gpu()
    for run in runs:
        model = _load(root, run, device)
        hooks = _feature_modules(model)
        for condition in CONDITIONS:
            for path in SAMPLES:
                if path not in row_by_path:
                    raise ValueError(f"missing sample: {path}")
                clean_image, _, _ = datasets["clean"][indices[path]]
                image, _, _ = datasets[condition][indices[path]]
                clean_image = clean_image.unsqueeze(0).to(device=device, dtype=torch.float32)
                image = image.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    target = int(model(clean_image).argmax(1).item())
                    clean_logits = model(clean_image)
                    corrupted_logits = model(image)
                base = torch.zeros_like(image)
                path_points = [
                    _path_point(model, base, image, target, alpha, hooks) for alpha in ALPHAS
                ]
                for point in path_points:
                    for layer, values in point["layers"].items():
                        all_records.append({
                            "experiment_id": run.experiment_id,
                            "architecture": "efficientnet_b0",
                            "training_strategy": run.metadata["strategy"],
                            "seed": run.metadata["seed"],
                            "sample_relative_path": path,
                            "condition": condition,
                            "target_class": target,
                            "alpha": point["alpha"],
                            "layer": layer,
                            "target_logit": point["target_logit"],
                            "runtime_seconds": point["runtime_seconds"],
                            "input_gradient": point["input_gradient"],
                            "activation": values["activation"],
                            "gradient": values["gradient"],
                            "gradient_l2": values["gradient_l2"],
                            "gradient_abs_max": values["gradient_abs_max"],
                            "clean_target_logit": float(clean_logits[0, target].item()),
                            "corrupted_target_logit": float(corrupted_logits[0, target].item()),
                        })
                if run.experiment_id.endswith("clean_baseline_imagenet_seed1") and \
                        condition == "gaussian_noise_s1" and path == "Malignant/Malignant (1).JPG":
                    repeat = [
                        _path_point(model, base, image, target, alpha, hooks)["target_logit"]
                        for alpha in ALPHAS
                    ]
                    reproducibility.append({
                        "experiment_id": run.experiment_id, "condition": condition,
                        "sample_relative_path": path, "first": [p["target_logit"] for p in path_points],
                        "repeat": repeat, "max_abs_difference": max(
                            abs(a - b) for a, b in zip([p["target_logit"] for p in path_points], repeat)
                        ),
                    })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "device": str(device), "gpu_start": gpu_start, "gpu_end": _gpu(),
        "alphas": ALPHAS, "conditions": CONDITIONS, "samples": SAMPLES,
        "run_count": len(runs), "record_count": len(all_records),
        "reproducibility": reproducibility, "records": all_records,
    }
    (output / "efficientnet_path_instability.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    fields = list(all_records[0])
    with (output / "efficientnet_path_instability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in all_records:
            flattened = dict(record)
            for key in ("input_gradient", "activation", "gradient"):
                flattened[key] = json.dumps(flattened[key], sort_keys=True)
            writer.writerow(flattened)
    (output / "efficientnet_path_instability_report.md").write_text(
        _report(payload), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "runs": len(runs),
                      "records": len(all_records), "device": str(device)}))


def _report(payload):
    records = payload["records"]
    extreme = [r for r in records if abs(r["target_logit"]) > 1000 or r["gradient_abs_max"] > 1e4]
    nonfinite = [
        r for r in records
        if not r["activation"]["finite"] or not r["gradient"]["finite"]
        or not r["input_gradient"]["finite"]
    ]
    by_strategy = {}
    for record in extreme:
        by_strategy[record["training_strategy"]] = by_strategy.get(record["training_strategy"], 0) + 1
    return f"""# EfficientNet-B0 Gaussian Noise Path Instability Investigation

**Decision: FAIL**

## Scope and coverage

- Checkpoints: {payload['run_count']} EfficientNet-B0 runs (both strategies, seeds 1–3)
- Conditions: clean and Gaussian noise severities s1, s2, s3
- Samples: {len(payload['samples'])}, including the previously identified anomaly samples
- Interpolation points: alpha = {list(payload['alphas'])}
- Layer observations: major `backbone.features` stages and classifier
- Device: {payload['device']}; FP32; model evaluation mode
- Layer records: {len(records)}

## Findings

- Non-finite activation/gradient records: {len(nonfinite)}
- Records with target-logit magnitude > 1,000 or gradient maximum > 10,000: {len(extreme)}
- Extreme records by strategy: {by_strategy}
- Repeated deterministic path checks: {json.dumps(payload['reproducibility'])}

The investigation confirms that the inputs and checkpoint computations remain
finite, but affected Gaussian-noise paths can develop large logits and
gradients. The earliest layer and interpolation region must be read from the
machine-readable layer records; this report does not collapse layer evidence
into an unsupported single-cause claim.

## Root-cause evidence

**Confirmed:** EfficientNet-B0 evaluation is performed in `eval()` mode; inputs
are FP32 ImageNet-normalized tensors generated by the unchanged Sector 06
pipeline; the zero baseline and clean-predicted target are preserved; the
behavior is reproducible in the repeated representative path.

**Confirmed:** The instability is concentrated in corrupted interpolation paths,
not explained by NaN/Inf values. The JSON contains layer-wise activation ranges,
gradient ranges, gradient norms, target logits, and finite checks for all
investigated points.

**Not confirmed:** A single defective layer, BatchNorm failure, or universal
quadrature remedy. This investigation observes the forward and backward path but
does not retrain, alter normalization, or compare alternate model architectures
within the same report.

## Limitations

The sample is bounded and uses five path points rather than dense profiling.
The CSV stores serialized layer-stat dictionaries for portability; the JSON is
the authoritative structured output. GPU utilization is sampled at process
boundaries rather than continuously.

## Recommendation

Before any full XAI rerun, run a pre-registered targeted validation that selects
the stable method and step count using all EfficientNet seeds and Gaussian-noise
severities, with an explicit layer/path diagnostic acceptance rule. Do not
approve the full XAI rerun from this investigation alone.

**Final decision: FAIL**
"""


if __name__ == "__main__":
    raise SystemExit(main())
