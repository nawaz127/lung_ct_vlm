"""Paired clean/corrupted explanation protocol."""

import hashlib
import json
from pathlib import Path

import torch

from .gradcam import grad_cam
from .integrated_gradients import PROTOCOL_VERSION, integrated_gradients
from .stability import cosine_similarity, spearman_rank_correlation, ssim


def stable_identity(row):
    return row.get("source_relative_path") or row.get("sample_id")


def paired_explanation(model, clean, corrupted, row, condition, method="gradcam",
                       seed=20261111, checkpoint_id=None, ig_steps=32,
                       ig_integration_method="left", ig_gradient_batch_size=1):
    if clean.shape != corrupted.shape or clean.ndim != 4 or clean.shape[0] != 1:
        raise ValueError("paired inputs must be matching single-image batches")
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean)
        corrupted_logits = model(corrupted)
    clean_prediction = int(clean_logits.argmax(1).item())
    corrupted_prediction = int(corrupted_logits.argmax(1).item())
    if method == "gradcam":
        clean_map, _ = grad_cam(model, clean, clean_prediction)
        corrupted_map, _ = grad_cam(model, corrupted, clean_prediction)
    elif method == "integrated_gradients":
        clean_attr, clean_delta = integrated_gradients(
            model, clean, clean_prediction, ig_steps,
            integration_method=ig_integration_method,
            gradient_batch_size=ig_gradient_batch_size,
        )
        corrupted_attr, corrupted_delta = integrated_gradients(
            model, corrupted, clean_prediction, ig_steps,
            integration_method=ig_integration_method,
            gradient_batch_size=ig_gradient_batch_size,
        )
        clean_map, corrupted_map = clean_attr.abs().mean(1), corrupted_attr.abs().mean(1)
    else:
        raise ValueError(f"unsupported attribution method: {method}")
    return {
        "sample_identity": stable_identity(row),
        "canonical_label": int(row["class_index"]),
        "model_architecture": model.model_name,
        "checkpoint_id": checkpoint_id,
        "training_strategy": getattr(model, "initialization", "unknown"),
        "seed": seed,
        "method": method,
        **({"ig_protocol_version": PROTOCOL_VERSION,
            "ig_integration_method": ig_integration_method,
            "ig_steps": ig_steps,
            "ig_gradient_batch_size": ig_gradient_batch_size,
            "ig_dtype": "float32"} if method == "integrated_gradients" else {}),
        "target_class": clean_prediction,
        "clean_prediction": clean_prediction,
        "corrupted_prediction": corrupted_prediction,
        "prediction_changed": clean_prediction != corrupted_prediction,
        "condition": condition,
        "attribution_shape": list(clean_map.shape),
        "stability": {
            "ssim": ssim(clean_map, corrupted_map),
            "cosine": cosine_similarity(clean_map, corrupted_map),
            "spearman": spearman_rank_correlation(clean_map, corrupted_map),
        },
        "clean_attribution": clean_map.detach().cpu(),
        "corrupted_attribution": corrupted_map.detach().cpu(),
        **({"clean_convergence_delta": clean_delta.detach().cpu(),
            "corrupted_convergence_delta": corrupted_delta.detach().cpu()}
           if method == "integrated_gradients" else {}),
    }


def save_explanation(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    manifest = {key: value for key, value in result.items()
                if key not in {"clean_attribution", "corrupted_attribution",
                               "clean_convergence_delta", "corrupted_convergence_delta"}}
    manifest["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
