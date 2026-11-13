"""Validate available Sector 08 artifacts and compute BiomedCLIP metrics."""

import argparse
import json
from pathlib import Path

from evaluation import evaluate_predictions, robustness_summary, validate_prediction_records

CONDITIONS = (
    "clean", "gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3",
    "gaussian_blur_s1", "gaussian_blur_s2", "gaussian_blur_s3",
    "contrast_reduction_s1", "contrast_reduction_s2", "contrast_reduction_s3",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="reports/biomedclip/predictions.jsonl")
    parser.add_argument("--output-dir", default="reports/evaluation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root / args.predictions).read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["model"] = "biomedclip"
        row["source_relative_path"] = row["sample_id"]
    integrity = validate_prediction_records(rows, CONDITIONS)
    metrics = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        mapping = {"Normal": 0, "Benign": 1, "Malignant": 2}
        metrics[condition] = evaluate_predictions(
            [row["ground_truth"] for row in subset],
            [mapping[row["predicted_class"]] for row in subset],
            [[row["normal_score"], row["benign_score"], row["malignant_score"]] for row in subset],
        )
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "biomedclip_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "biomedclip_robustness.json").write_text(
        json.dumps(robustness_summary(metrics), indent=2), encoding="utf-8"
    )
    (output / "sample_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (output / "checkpoint_audit.json").write_text(json.dumps({
        "research_checkpoints_found": [],
        "smoke_checkpoints_excluded": [
            "experiments/sector05_mobilenetv3_smoke_20261111T132904Z/best.pt",
            "experiments/sector05_mobilenetv3_smoke_20261111T132904Z/last.pt",
        ],
        "missing_research_checkpoints": ["mobilenet_v3_large", "efficientnet_b0", "resnet18"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(integrity))


if __name__ == "__main__":
    raise SystemExit(main())
