"""Run the fixed-sample research Grad-CAM and Integrated Gradients protocol."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.degradation import CONDITION_IDS, DegradedManifestDataset
from evaluation.cnn_runner import audit_checkpoints, discover_runs
from models import create_model
from xai import PROTOCOL_VERSION, grad_cam, integrated_gradients
from xai.stability import cosine_similarity, spearman_rank_correlation, ssim


def _read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_rows(rows, per_class):
    selected = []
    for class_index in ("0", "1", "2"):
        candidates = [row for row in rows if row["class_index"] == class_index]
        selected.extend(sorted(candidates, key=lambda row: row["source_relative_path"])[:per_class])
    if len(selected) != per_class * 3:
        raise ValueError("test manifest does not contain enough samples per class")
    return selected


def _model(run, device):
    model = create_model(
        run.metadata["model"], pretrained=False,
        head_type=run.metadata.get("head_type", "gap_dropout_linear"), dropout=0.2,
    )
    checkpoint = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _attribution(model, image, target, method, steps, ig_method="trapezoidal", ig_batch_size=8):
    if method == "gradcam":
        heatmap, _ = grad_cam(model, image, target)
        return heatmap[0].detach().cpu().numpy(), None
    attribution, delta = integrated_gradients(
        model, image, target, steps,
        integration_method=ig_method, gradient_batch_size=ig_batch_size,
    )
    return attribution[0].abs().mean(0).detach().cpu().numpy(), float(delta[0].detach().cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/final_evaluation/xai")
    parser.add_argument("--manifest", default="data/splits/test.csv")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--degradation-config", default="configs/degradation.yaml")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-method", default="trapezoidal")
    parser.add_argument("--ig-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    del args.batch_size
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    arrays_dir, metadata_dir = output / "attributions", output / "metadata"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(root)
    audit = audit_checkpoints(root, runs)
    if not audit["complete"]:
        raise RuntimeError("Checkpoint audit failed; XAI stopped")
    runs = runs[:args.max_experiments] if args.max_experiments else runs
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but unavailable")
    rows = _read_rows(root / args.manifest)
    selected = _select_rows(rows, args.per_class)
    degradation = yaml.safe_load((root / args.degradation_config).read_text(encoding="utf-8"))
    stability_rows = []
    protocol = {
        "manifest": args.manifest, "data_root": args.data_root,
        "per_class": args.per_class, "sample_count": len(selected),
        "sample_identities": [row["source_relative_path"] for row in selected],
        "conditions": list(CONDITION_IDS), "methods": ["gradcam", "integrated_gradients"],
        "integrated_gradients_steps": args.ig_steps,
        "ig_protocol_version": PROTOCOL_VERSION,
        "ig_integration_method": args.ig_method,
        "ig_gradient_batch_size": args.ig_batch_size,
        "ig_dtype": "float32",
        "attribution_target": "clean_image_predicted_class",
        "true_class_mapping": {"Normal": 0, "Benign": 1, "Malignant": 2},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for run in runs:
        model = _model(run, device)
        for method in ("gradcam", "integrated_gradients"):
            for condition in CONDITION_IDS:
                dataset = DegradedManifestDataset(
                    root / args.manifest, root / args.data_root, condition,
                    degradation, identity_by_path=True,
                )
                by_identity = {row["source_relative_path"]: index for index, row in enumerate(dataset.rows)}
                clean_maps, corrupted_maps = [], []
                records = []
                for row in selected:
                    clean_dataset = DegradedManifestDataset(
                        root / args.manifest, root / args.data_root, "clean",
                        degradation, identity_by_path=True,
                    )
                    clean_image, true_label, _ = clean_dataset[by_identity[row["source_relative_path"]]]
                    condition_image, _, _ = dataset[by_identity[row["source_relative_path"]]]
                    clean_image = clean_image.unsqueeze(0).to(device)
                    condition_image = condition_image.unsqueeze(0).to(device)
                    with torch.no_grad():
                        clean_prediction = int(model(clean_image).argmax(1).item())
                        corrupted_prediction = int(model(condition_image).argmax(1).item())
                    clean_map, clean_delta = _attribution(
                        model, clean_image, clean_prediction, method, args.ig_steps,
                        ig_method=args.ig_method, ig_batch_size=args.ig_batch_size,
                    )
                    corrupted_map, corrupted_delta = _attribution(
                        model, condition_image, clean_prediction, method, args.ig_steps,
                        ig_method=args.ig_method, ig_batch_size=args.ig_batch_size,
                    )
                    clean_maps.append(clean_map)
                    corrupted_maps.append(corrupted_map)
                    family = "clean" if condition == "clean" else condition.rsplit("_s", 1)[0]
                    severity = 0 if condition == "clean" else int(condition.rsplit("_s", 1)[1])
                    record = {
                        "experiment_id": run.experiment_id,
                        "architecture": run.metadata["model"],
                        "training_strategy": run.metadata["strategy"],
                        "seed": run.metadata["seed"],
                        "method": method,
                        "ig_protocol_version": PROTOCOL_VERSION if method == "integrated_gradients" else None,
                        "ig_integration_method": args.ig_method if method == "integrated_gradients" else None,
                        "ig_steps": args.ig_steps if method == "integrated_gradients" else None,
                        "sample_relative_path": row["source_relative_path"],
                        "true_label": int(true_label),
                        "clean_prediction": clean_prediction,
                        "corrupted_prediction": corrupted_prediction,
                        "target_class": clean_prediction,
                        "condition": condition,
                        "corruption_family": family,
                        "severity": severity,
                        "clean_convergence_delta": clean_delta,
                        "corrupted_convergence_delta": corrupted_delta,
                    }
                    record.update({
                        "ssim": ssim(clean_map, corrupted_map),
                        "cosine": cosine_similarity(clean_map, corrupted_map),
                        "spearman": spearman_rank_correlation(clean_map, corrupted_map),
                    })
                    records.append(record)
                    del clean_image, condition_image
                stem = f"{run.experiment_id}__{method}__{condition}"
                np.savez_compressed(
                    arrays_dir / f"{stem}.npz",
                    clean=np.asarray(clean_maps, dtype=np.float32),
                    corrupted=np.asarray(corrupted_maps, dtype=np.float32),
                )
                (metadata_dir / f"{stem}.json").write_text(
                    json.dumps(records, indent=2), encoding="utf-8"
                )
                stability_rows.extend(records)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    (output / "stability.json").write_text(json.dumps(stability_rows, indent=2), encoding="utf-8")
    with (output / "stability.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(stability_rows[0]) if stability_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(stability_rows)
    print(json.dumps({"experiments": len(runs), "samples": len(selected),
                      "conditions": len(CONDITION_IDS), "methods": 2,
                      "stability_records": len(stability_rows), "device": str(device)}))


if __name__ == "__main__":
    raise SystemExit(main())
