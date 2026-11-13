"""Checkpoint audit and reproducible all-condition CNN evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
import yaml

from data.degradation import CONDITION_IDS, DegradedManifestDataset
from models import create_model
from evaluation.metrics import evaluate_predictions


CLASS_NAMES = ("Normal", "Benign", "Malignant")


@dataclass(frozen=True)
class ResearchRun:
    experiment_id: str
    directory: Path
    metadata: dict
    status: dict
    checkpoint: Path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_runs(root: Path) -> list[ResearchRun]:
    runs = []
    for metadata_path in sorted((root / "experiments").glob("research_*/experiment_metadata.json")):
        directory = metadata_path.parent
        metadata = _read_json(metadata_path)
        status_path = directory / "run_status.json"
        status = _read_json(status_path) if status_path.is_file() else {}
        runs.append(ResearchRun(
            directory.name, directory, metadata, status, directory / "best.pt"
        ))
    return runs


def audit_checkpoints(root: Path, runs: list[ResearchRun] | None = None) -> dict:
    runs = runs or discover_runs(root)
    expected = {
        f"{architecture}_{strategy}_imagenet_seed{seed}"
        for architecture in ("mobilenet_v3_large", "efficientnet_b0", "resnet18")
        for strategy in ("clean_baseline", "corruption_aware")
        for seed in (1, 2, 3)
    }
    findings = []
    identities = []
    for run in runs:
        metadata = run.metadata
        identity = metadata.get("configuration_id")
        identities.append(identity)
        errors = []
        if run.status.get("status") != "completed":
            errors.append(f"run_status.json status is {run.status.get('status')!r}")
        if metadata.get("initialization") != "imagenet":
            errors.append("initialization is not imagenet")
        if metadata.get("run_id") != run.experiment_id:
            errors.append("run_id does not match directory name")
        if identity not in expected:
            errors.append(f"unexpected configuration_id: {identity!r}")
        metadata_status_warning = None
        if metadata.get("status") != "completed":
            metadata_status_warning = (
                f"experiment_metadata.json status is {metadata.get('status')!r}; "
                "run_status.json is the authoritative completion record"
            )
        if set(metadata.get("manifest_paths", [])) & {"data\\splits\\test.csv", "data/splits/test.csv"}:
            errors.append("test manifest appears in training metadata")
        if not run.checkpoint.is_file():
            errors.append("best.pt is missing")
        else:
            try:
                checkpoint = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
                if checkpoint.get("model_name") != metadata.get("model"):
                    errors.append("checkpoint model_name does not match metadata")
                if checkpoint.get("num_classes") != 3:
                    errors.append("checkpoint num_classes is not 3")
                checkpoint_metadata = checkpoint.get("metadata", {})
                for field in ("model", "strategy", "seed", "configuration_id", "initialization"):
                    if checkpoint_metadata.get(field) != metadata.get(field):
                        errors.append(f"checkpoint metadata field mismatch: {field}")
                if not checkpoint.get("history"):
                    errors.append("checkpoint history is empty")
                elif checkpoint.get("best_val_macro_f1") is None:
                    errors.append("best validation Macro-F1 is missing")
            except Exception as error:
                errors.append(f"checkpoint load failed: {type(error).__name__}: {error}")
        findings.append({
            "experiment_id": run.experiment_id,
            "architecture": metadata.get("model"),
            "strategy": metadata.get("strategy"),
            "seed": metadata.get("seed"),
            "checkpoint": str(run.checkpoint),
            "valid": not errors,
            "issues": errors,
            "warnings": [metadata_status_warning] if metadata_status_warning else [],
        })
    duplicate_ids = sorted(identity for identity in set(identities) if identities.count(identity) > 1)
    missing = sorted(expected - set(identities))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovered_runs": len(runs),
        "expected_runs": len(expected),
        "unique_experiment_identities": len(set(identities)),
        "duplicate_experiment_identities": duplicate_ids,
        "missing_experiment_identities": missing,
        "valid_checkpoints": sum(item["valid"] for item in findings),
        "invalid_checkpoints": [item for item in findings if not item["valid"]],
        "runs": findings,
        "complete": len(runs) == 18 and not duplicate_ids and not missing
        and all(item["valid"] for item in findings),
    }


def _condition_metadata(condition):
    if condition == "clean":
        return "clean", 0
    family, severity = condition.rsplit("_s", 1)
    return family, int(severity)


def _load_model(run: ResearchRun, device):
    model = create_model(
        run.metadata["model"], pretrained=False,
        head_type=run.metadata.get("head_type", "gap_dropout_linear"),
        dropout=0.2,
    )
    checkpoint = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model


def _write_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def evaluate_run(run: ResearchRun, root: Path, output_dir: Path, test_manifest: Path,
                 data_root: Path, degradation: dict, device, batch_size=32,
                 max_samples=None):
    checkpoint_hash = hashlib.sha256(run.checkpoint.read_bytes()).hexdigest()
    model = _load_model(run, device)
    rows = list(csv.DictReader(test_manifest.open(encoding="utf-8")))
    if max_samples is not None:
        rows = rows[:max_samples]
    predictions_path = output_dir / "predictions.jsonl"
    metrics_rows = []
    for condition in CONDITION_IDS:
        dataset = DegradedManifestDataset(
            test_manifest, data_root, condition, degradation, identity_by_path=True
        )
        if max_samples is not None:
            dataset.rows = dataset.rows[:max_samples]
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        targets, predictions, scores = [], [], []
        records = []
        with torch.inference_mode():
            for images, labels, identities in loader:
                probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
                predicted = probabilities.argmax(1)
                for identity, label, prediction, probability in zip(
                    identities, labels.tolist(), predicted.tolist(), probabilities.tolist()
                ):
                    family, severity = _condition_metadata(condition)
                    records.append({
                        "experiment_id": run.experiment_id,
                        "architecture": run.metadata["model"],
                        "training_strategy": run.metadata["strategy"],
                        "random_seed": run.metadata["seed"],
                        "initialization": run.metadata["initialization"],
                        "configuration_id": run.metadata["configuration_id"],
                        "sample_relative_path": identity,
                        "true_label": int(label),
                        "predicted_label": int(prediction),
                        "predicted_class_probabilities": probability,
                        "condition_id": condition,
                        "corruption_family": family,
                        "corruption_severity": severity,
                        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
                        "checkpoint_path": str(run.checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                    })
                    targets.append(label)
                    predictions.append(prediction)
                    scores.append(probability)
        _write_jsonl(predictions_path, records)
        metrics = evaluate_predictions(targets, predictions, scores)
        metrics_rows.append({
            "experiment_id": run.experiment_id,
            "architecture": run.metadata["model"],
            "training_strategy": run.metadata["strategy"],
            "seed": run.metadata["seed"],
            "condition": condition,
            **metrics,
        })
    return metrics_rows


def measure_efficiency(root: Path, runs: list[ResearchRun], device, output_path: Path,
                       repetitions=30, warmup=5):
    rows = []
    for architecture in sorted({run.metadata["model"] for run in runs}):
        run = next(run for run in runs if run.metadata["model"] == architecture)
        model = _load_model(run, device)
        parameters = list(model.parameters())
        total = sum(parameter.numel() for parameter in parameters)
        input_tensor = torch.zeros(1, 3, 224, 224, device=device)
        for _ in range(warmup):
            with torch.inference_mode():
                model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
        rows.append({
            "architecture": architecture,
            "total_parameters": total,
            "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
            "checkpoint_size_bytes": run.checkpoint.stat().st_size,
            "latency_ms_mean": sum(timings) / len(timings),
            "latency_ms_std": (sum((t - sum(timings) / len(timings)) ** 2 for t in timings) / len(timings)) ** 0.5,
            "throughput_images_per_second": 1000 / (sum(timings) / len(timings)),
            "peak_evaluation_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "device": str(device),
            "precision": "fp32_inference",
        })
    Path(output_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
