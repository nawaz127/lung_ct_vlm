"""Run a bounded, non-destructive Integrated Gradients quadrature diagnostic."""

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
from xai import integrated_gradients
from xai.stability import cosine_similarity


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _model(run, device):
    model = create_model(
        run.metadata["model"], pretrained=False,
        head_type=run.metadata.get("head_type", "gap_dropout_linear"), dropout=0.2,
    )
    checkpoint = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        values = [int(value.strip()) for value in result.stdout.split(",")]
        return {"gpu_utilization_percent": values[0], "vram_used_mb": values[1],
                "vram_total_mb": values[2]}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"gpu_utilization_percent": None, "vram_used_mb": None,
                "vram_total_mb": None}


def _record(model, image, target, method, steps, next_map, gradient_batch_size):
    if image.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(image.device)
        torch.cuda.synchronize(image.device)
    started = time.perf_counter()
    attribution, delta = integrated_gradients(
        model, image, target, steps=steps, integration_method=method,
        gradient_batch_size=gradient_batch_size,
    )
    if image.device.type == "cuda":
        torch.cuda.synchronize(image.device)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        target_output = float(model(image)[0, target].item())
        baseline_output = float(model(torch.zeros_like(image))[0, target].item())
    map_value = attribution.abs().mean(1)
    return {
        "attribution_sum": float(attribution.flatten(1).sum().item()),
        "attribution_absolute_max": float(attribution.abs().max().item()),
        "attribution_mean_absolute": float(attribution.abs().mean().item()),
        "target_output_input": target_output,
        "target_output_baseline": baseline_output,
        "completeness_delta": float(delta[0].item()),
        "absolute_completeness_error": float(delta[0].abs().item()),
        "relative_completeness_error": float(
            delta[0].abs().item() / max(abs(target_output - baseline_output), 1e-12)
        ),
        "map_cosine_to_next_higher_step": (
            cosine_similarity(map_value, next_map) if next_map is not None else None
        ),
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_mb": (
            round(torch.cuda.max_memory_allocated(image.device) / 1024**2, 3)
            if image.device.type == "cuda" else None
        ),
        "attribution_finite": bool(torch.isfinite(attribution).all().item()),
        "delta_finite": bool(torch.isfinite(delta).all().item()),
        "attribution_shape": list(attribution.shape),
        "_map": map_value.detach(),
    }


def _write_report(output, payload):
    records = payload["records"]
    extreme = [r for r in records if abs(r["completeness_delta"]) > 1e4]
    nonfinite = [r for r in records if not r["attribution_finite"] or not r["delta_finite"]]
    runtimes = [r["runtime_seconds"] for r in records]
    lines = [
        "# Integrated Gradients Quadrature Validation",
        "",
        f"**Decision: {payload['decision']}**",
        "",
        "This is a bounded diagnostic only. It did not modify checkpoints, manifests, "
        "corruption definitions, or previous XAI artifacts, and it did not launch the "
        "full 18-experiment XAI evaluation.",
        "",
        "## Coverage",
        "",
        f"- Runs: {payload['run_count']} (all three architectures, both strategies, seed 1)",
        f"- Case-condition evaluations: {payload['coverage_count']}",
        f"- Diagnostic records: {len(records)}",
        f"- Methods: {', '.join(payload['configuration']['integration_methods'])}",
        f"- Steps: {payload['configuration']['step_counts']}",
        f"- Device: {payload['device']}; gradient batch size: "
        f"{payload['configuration']['gradient_batch_size']}",
        "",
        "The base case is deterministic and the previously identified EfficientNet-B0 "
        "Gaussian-noise cases are included at their matching conditions.",
        "",
        "## Measurements",
        "",
        "Each record contains attribution sum, absolute and relative completeness "
        "error, target output at input and baseline, successive-step map cosine "
        "similarity, runtime, peak CUDA allocation, and finite-value checks. "
        "The baseline remains zero in normalized input space and the target remains "
        "the clean-image predicted class.",
        "",
        f"- Non-finite records: {len(nonfinite)}",
        f"- Records with `abs(completeness_delta) > 10,000`: {len(extreme)}",
        f"- Runtime range: {min(runtimes):.3f}–{max(runtimes):.3f} seconds",
        "",
        "## GPU profiling",
        "",
        f"Start/end GPU snapshots: `{payload['gpu_start']}` / `{payload['gpu_end']}`.",
        "The diagnostic records peak CUDA allocation per case. GPU utilization is "
        "sampled at process start and end; this is not a continuous utilization trace. "
        "The implementation batches interpolation points without changing FP32 "
        "quadrature weights, reducing sequential Python/model-call overhead while "
        "preserving the mathematical result.",
        "",
        "## Acceptance criteria",
        "",
        "No universal numerical threshold was imposed merely to obtain approval. "
        "A defensible full-evaluation protocol requires both bounded completeness "
        "error relative to the target-logit change and stable successive-step "
        "attribution maps, assessed by architecture and condition. A finite result "
        "or a small completeness delta alone is insufficient. The current bounded "
        "evidence is therefore reported conservatively as not approved for a full "
        "rerun until the diagnostic results are reviewed.",
        "",
        "## Limitations and decision",
        "",
        "This subset uses seed 1 and one deterministic base sample per condition, "
        "with additional known EfficientNet anomaly samples. It does not establish "
        "all-seed convergence or prove that a single step count is adequate for every "
        "Gaussian-noise path. The current decision is **FAIL** if any non-finite or "
        "extreme unstable records are present, otherwise **PASS WITH WARNINGS** because "
        "scientific acceptance thresholds remain proposed rather than validated.",
        "",
        f"**Final decision: {payload['decision']}**",
    ]
    (output / "ig_quadrature_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="data/splits/test.csv")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--degradation-config", default="configs/degradation.yaml")
    parser.add_argument("--diagnostic-config", default="configs/ig_quadrature_diagnostic.yaml")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diagnostic output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((root / args.diagnostic_config).read_text(encoding="utf-8"))
    degradation = yaml.safe_load((root / args.degradation_config).read_text(encoding="utf-8"))
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    runs = [
        run for run in discover_runs(root)
        if run.experiment_id.startswith("research_")
        and run.metadata["seed"] in config["coverage"]["run_selection"]["seeds"]
        and run.metadata["model"] in config["coverage"]["run_selection"]["architectures"]
        and run.metadata["strategy"] in config["coverage"]["run_selection"]["strategies"]
    ]
    full_audit = audit_checkpoints(root)
    if not full_audit["complete"]:
        raise RuntimeError("Full checkpoint audit failed; diagnostic stopped")
    audit = audit_checkpoints(root, runs)
    if any(not item["valid"] for item in audit["runs"]):
        raise RuntimeError("Selected checkpoint audit failed; diagnostic stopped")
    rows = sorted(_rows(root / args.manifest), key=lambda row: row["source_relative_path"])
    base_row = rows[0]
    anomaly_conditions = {
        "Malignant/Malignant (10).JPG": "gaussian_noise_s3",
        "Bengin/Benign (104).JPG": "gaussian_noise_s2",
        "Normal/Normal case (136).jpg": "gaussian_noise_s3",
        "Malignant/Malignant (1).JPG": "gaussian_noise_s1",
    }
    row_by_path = {row["source_relative_path"]: row for row in rows}
    conditions, steps, methods = config["conditions"], config["step_counts"], config["integration_methods"]
    records, coverage = [], []
    gpu_start = _gpu_snapshot()
    for run in runs:
        model = _model(run, device)
        for condition in conditions:
            dataset = DegradedManifestDataset(
                root / args.manifest, root / args.data_root, condition,
                degradation, identity_by_path=True,
            )
            indices = {row["source_relative_path"]: i for i, row in enumerate(dataset.rows)}
            case_rows = {base_row["source_relative_path"]: base_row}
            if run.metadata["model"] == "efficientnet_b0":
                case_rows.update({
                    path: row_by_path[path] for path, anomaly_condition in anomaly_conditions.items()
                    if condition == anomaly_condition
                })
            for path, row in case_rows.items():
                clean_dataset = DegradedManifestDataset(
                    root / args.manifest, root / args.data_root, "clean",
                    degradation, identity_by_path=True,
                )
                clean_image, _, _ = clean_dataset[indices[path]]
                image, _, _ = dataset[indices[path]]
                clean_image = clean_image.unsqueeze(0).to(device=device, dtype=torch.float32)
                image = image.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    target = int(model(clean_image).argmax(1).item())
                for method in methods:
                    maps, values = {}, {}
                    for count in reversed(steps):
                        result = _record(
                            model, image, target, method, count, maps.get(count * 2),
                            config["gradient_batch_size"],
                        )
                        maps[count], values[count] = result.pop("_map"), result
                    for count in steps:
                        records.append({
                            "experiment_id": run.experiment_id,
                            "architecture": run.metadata["model"],
                            "training_strategy": run.metadata["strategy"],
                            "seed": run.metadata["seed"],
                            "sample_relative_path": path,
                            "condition": condition,
                            "target_class": target,
                            "integration_method": method,
                            "step_count": count,
                            **values[count],
                        })
                coverage.append({
                    "experiment_id": run.experiment_id,
                    "architecture": run.metadata["model"],
                    "training_strategy": run.metadata["strategy"],
                    "seed": run.metadata["seed"],
                    "condition": condition,
                    "sample_relative_path": path,
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for record in records:
        record.pop("_map", None)
    with (output / "ig_quadrature_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "configuration": config,
        "device": str(device),
        "run_count": len(runs),
        "coverage_count": len(coverage),
        "records": records,
        "coverage": coverage,
        "gpu_start": gpu_start,
        "gpu_end": _gpu_snapshot(),
        "decision": "FAIL" if any(
            not r["attribution_finite"] or not r["delta_finite"]
            or abs(r["completeness_delta"]) > 1e4 for r in records
        ) else "PASS WITH WARNINGS",
    }
    (output / "ig_quadrature_diagnostics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_report(output, payload)
    print(json.dumps({
        "output_dir": str(output), "device": str(device),
        "runs": len(runs), "coverage": len(coverage), "records": len(records),
        "decision": payload["decision"],
    }))


if __name__ == "__main__":
    raise SystemExit(main())
