"""Audit and evaluate all research CNN checkpoints on the frozen test set."""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from evaluation.cnn_runner import (
    audit_checkpoints, discover_runs, evaluate_run, measure_efficiency,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/final_evaluation")
    parser.add_argument("--manifest", default="data/splits/test.csv")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--degradation-config", default="configs/degradation.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(root)
    audit = audit_checkpoints(root, runs)
    (output / "checkpoint_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Pre-evaluation checkpoint audit", "",
             f"- Discovered runs: **{audit['discovered_runs']}**",
             f"- Expected runs: **{audit['expected_runs']}**",
             f"- Valid checkpoints: **{audit['valid_checkpoints']}**",
             f"- Unique identities: **{audit['unique_experiment_identities']}**", ""]
    for item in audit["runs"]:
        status = "VALID" if item["valid"] else "INVALID"
        lines.append(f"- `{item['experiment_id']}`: **{status}**")
        for issue in item["issues"]:
            lines.append(f"  - {issue}")
        for warning in item.get("warnings", []):
            lines.append(f"  - Warning: {warning}")
    (output / "pre_evaluation_checkpoint_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not audit["complete"]:
        raise RuntimeError("Checkpoint audit failed; evaluation stopped. See checkpoint_audit.json")
    selected = runs[:args.max_experiments] if args.max_experiments else runs
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but unavailable")
    degradation = yaml.safe_load((root / args.degradation_config).read_text(encoding="utf-8"))
    prediction_path = output / "predictions.jsonl"
    if prediction_path.exists():
        prediction_path.unlink()
    metrics = []
    for run in selected:
        metrics.extend(evaluate_run(
            run, root, output, root / args.manifest, root / args.data_root,
            degradation, device, args.batch_size, args.max_samples,
        ))
    metrics_path = output / "metrics.csv"
    fields = list(metrics[0]) if metrics else []
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if selected:
        measure_efficiency(root, selected, device, output / "efficiency.json")
    print(json.dumps({
        "device": str(device), "experiments": len(selected),
        "prediction_records": sum(1 for _ in prediction_path.open(encoding="utf-8")),
        "metrics": len(metrics), "output": str(output),
    }))


if __name__ == "__main__":
    raise SystemExit(main())
