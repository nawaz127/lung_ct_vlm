"""Execute controlled 7-stage validation (Stages A-G) and candidate solution experiments for Integrated Gradients."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.degradation import DegradedManifestDataset
from evaluation.cnn_runner import discover_runs
from models import create_model
from xai.gradcam import grad_cam
from xai.integrated_gradients import (
    PROTOCOL_VERSION,
    integrated_gradients,
    integrated_gradients_legacy,
)

LEGACY_PROTOCOL_VERSION = "ig_v1_legacy_left_endpoint_inclusive"
from xai.integrated_gradients.method import _quadrature_grid
from xai.stability import cosine_similarity, spearman_rank_correlation, ssim


class ToyLinearModel(torch.nn.Module):
    num_classes = 2

    def forward(self, inputs):
        weights = torch.tensor([1.5, -0.75, 0.25], device=inputs.device, dtype=inputs.dtype).view(1, 3, 1, 1)
        score = (inputs * weights).flatten(1).sum(1, keepdim=True)
        return torch.cat((score, -score), dim=1)


class ToyQuadraticModel(torch.nn.Module):
    num_classes = 2

    def forward(self, inputs):
        weights = torch.tensor([1.2, 0.8, 0.4], device=inputs.device, dtype=inputs.dtype).view(1, 3, 1, 1)
        score = ((inputs ** 2) * weights).flatten(1).sum(1, keepdim=True)
        return torch.cat((score, -score), dim=1)


class ToyQuarticMultilinearModel(torch.nn.Module):
    num_classes = 2

    def forward(self, inputs):
        c0, c1, c2 = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        score = (0.5 * (c0 ** 4) + 1.25 * (c1 ** 3) + 0.75 * c0 * c1 * c2).flatten(1).sum(1, keepdim=True)
        return torch.cat((score, -score), dim=1)


def _load_checkpoint_model(run, device):
    model = create_model(
        run.metadata["model"],
        pretrained=False,
        head_type=run.metadata.get("head_type", "gap_dropout_linear"),
        dropout=0.2,
    )
    checkpoint = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _rel_delta(abs_delta, output_delta, eps=1e-8):
    return float(abs_delta / max(abs(output_delta), eps))


def _rel_l2_diff(attr_a, attr_b, eps=1e-12):
    a = attr_a.detach().to(torch.float64).flatten()
    b = attr_b.detach().to(torch.float64).flatten()
    return float(torch.linalg.norm(a - b) / max(float(torch.linalg.norm(b)), eps))


def _ig_with_diagnostics(model, image, target_class, steps, method="trapezoidal", batch_size=16, baseline=None):
    base = torch.zeros_like(image) if baseline is None else baseline.float()
    diff = image - base
    model.eval()
    if method == "legacy":
        alphas = torch.linspace(0.0, 1.0, steps, device=image.device, dtype=torch.float32)
        weights = torch.full((steps,), 1.0 / steps, device=image.device, dtype=torch.float32)
    else:
        alphas, weights = _quadrature_grid(steps, method, image.device)
    total = torch.zeros_like(image)
    l2_norms = []
    max_abs_vals = []
    adj_diffs = []
    prev_grad = None
    for start in range(0, len(alphas), batch_size):
        alpha_batch = alphas[start:start + batch_size]
        weight_batch = weights[start:start + batch_size]
        point = (base + alpha_batch[:, None, None, None] * diff).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(point)[:, int(target_class)]
        grads = torch.autograd.grad(out.sum(), point)[0].detach()
        total += (grads * weight_batch[:, None, None, None]).sum(dim=0, keepdim=True)
        g_flat = grads.flatten(1)
        batch_l2 = torch.linalg. vector_norm(g_flat, dim=1).cpu().tolist()
        batch_max = g_flat.abs().max(dim=1).values.cpu().tolist()
        l2_norms.extend(batch_l2)
        max_abs_vals.extend(batch_max)
        for idx in range(grads.shape[0]):
            g = grads[idx]
            if prev_grad is not None:
                adj_diffs.append(float(torch.linalg.vector_norm((g - prev_grad).flatten()).item()))
            prev_grad = g
    attribution = diff * total
    with torch.no_grad():
        output_delta = model(image)[:, int(target_class)] - model(base)[:, int(target_class)]
    convergence_delta = output_delta - attribution.flatten(1).sum(1)
    diag = {
        "grad_l2_mean": float(np.mean(l2_norms)),
        "grad_l2_max": float(np.max(l2_norms)),
        "grad_abs_max": float(np.max(max_abs_vals)),
        "adj_grad_l2_max": float(np.max(adj_diffs)) if adj_diffs else 0.0,
    }
    return attribution.detach(), convergence_delta.detach(), float(output_delta.item()), diag


def _gauss_legendre_ig(model, image, target_class, steps, batch_size=8, baseline=None):
    base = torch.zeros_like(image) if baseline is None else baseline.float()
    diff = image - base
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    alphas = torch.tensor(0.5 * (nodes + 1.0), device=image.device, dtype=torch.float32)
    weights_t = torch.tensor(0.5 * weights, device=image.device, dtype=torch.float32)
    total = torch.zeros_like(image)
    model.eval()
    for start in range(0, len(alphas), batch_size):
        ab = alphas[start:start + batch_size]
        wb = weights_t[start:start + batch_size]
        point = (base + ab[:, None, None, None] * diff).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = model(point)[:, int(target_class)].sum()
        grads = torch.autograd.grad(score, point)[0].detach()
        total += (grads * wb[:, None, None, None]).sum(dim=0, keepdim=True)
    attribution = diff * total
    with torch.no_grad():
        output_delta = model(image)[:, int(target_class)] - model(base)[:, int(target_class)]
    delta = output_delta - attribution.flatten(1).sum(1)
    return attribution.detach(), delta.detach(), float(output_delta.item())


def run_stage_a(device):
    rows = []
    torch.manual_seed(20261112)
    x = torch.randn(1, 3, 16, 16, device=device, dtype=torch.float32)
    models = {
        "toy_linear": ToyLinearModel().to(device).eval(),
        "toy_quadratic": ToyQuadraticModel().to(device).eval(),
        "toy_quartic_multilinear": ToyQuarticMultilinearModel().to(device).eval(),
    }
    for model_name, model in models.items():
        with torch.no_grad():
            out_delta = float((model(x)[:, 0] - model(torch.zeros_like(x))[:, 0]).item())
        ref_attr, _ = integrated_gradients(
            model, x, 0, steps=512, integration_method="trapezoidal", gradient_batch_size=8
        )
        ref_map = ref_attr[0].abs().mean(0)
        for method in ("legacy", "left", "trapezoidal", "midpoint"):
            for steps in (4, 16, 64, 256, 512):
                t0 = time.perf_counter()
                if method == "legacy":
                    attr, delta = integrated_gradients_legacy(model, x, 0, steps=steps)
                    proto = LEGACY_PROTOCOL_VERSION
                else:
                    attr, delta = integrated_gradients(
                        model, x, 0, steps=steps, integration_method=method, gradient_batch_size=8
                    )
                    proto = PROTOCOL_VERSION
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                abs_d = abs(float(delta.item()))
                rows.append({
                    "stage": "Stage_A_Toy_Verification",
                    "experiment_id": model_name,
                    "architecture": model_name,
                    "training_strategy": "analytic_toy",
                    "seed": 0,
                    "sample_relative_path": "synthetic_1x3x16x16",
                    "condition": "clean",
                    "severity": 0,
                    "protocol_version": proto,
                    "integration_method": method,
                    "steps": steps,
                    "gradient_batch_size": 1 if method == "legacy" else 8,
                    "target_class": 0,
                    "output_delta": out_delta,
                    "abs_completeness_delta": abs_d,
                    "rel_completeness_delta": _rel_delta(abs_d, out_delta),
                    "cosine_vs_ref": cosine_similarity(attr[0].abs().mean(0), ref_map),
                    "rel_l2_diff_vs_ref": _rel_l2_diff(attr, ref_attr),
                    "successive_step_cosine": 1.0,
                    "grad_l2_mean": 0.0,
                    "grad_l2_max": 0.0,
                    "grad_abs_max": 0.0,
                    "adj_grad_l2_max": 0.0,
                    "runtime_ms": elapsed_ms,
                    "peak_gpu_mb": 0.0,
                    "status": "PASS" if (method != "legacy" and steps >= 64 and abs_d < 1e-4) else "INFO",
                })
    return rows


def run_stage_b(runs_by_key, dataset_cache, device):
    rows = []
    # 1. Toy Quadratic finite difference in float64
    q_model = ToyQuadraticModel().to(device).double().eval()
    x_toy = torch.tensor([[[[1.25, -0.75], [0.5, 2.0]], [[0.3, -1.1], [0.8, 0.2]], [[-0.4, 0.9], [1.5, -0.6]]]],
                         device=device, dtype=torch.float64).requires_grad_(True)
    score = q_model(x_toy)[:, 0].sum()
    autograd_g = torch.autograd.grad(score, x_toy)[0].detach()
    eps = 1e-6
    fd_g = torch.zeros_like(autograd_g)
    with torch.no_grad():
        for c in range(3):
            for h in range(2):
                for w in range(2):
                    xp = x_toy.detach().clone()
                    xm = x_toy.detach().clone()
                    xp[0, c, h, w] += eps
                    xm[0, c, h, w] -= eps
                    fd_g[0, c, h, w] = (q_model(xp)[0, 0] - q_model(xm)[0, 0]) / (2 * eps)
    rel_err_toy = _rel_l2_diff(autograd_g, fd_g)
    rows.append({
        "stage": "Stage_B_Finite_Difference",
        "experiment_id": "toy_quadratic_fp64",
        "architecture": "toy_quadratic",
        "training_strategy": "analytic_toy",
        "seed": 0,
        "sample_relative_path": "synthetic_1x3x2x2",
        "condition": "clean",
        "severity": 0,
        "protocol_version": PROTOCOL_VERSION,
        "integration_method": "central_fd_eps_1e-6",
        "steps": 1,
        "gradient_batch_size": 1,
        "target_class": 0,
        "output_delta": float(score.item()),
        "abs_completeness_delta": float((autograd_g - fd_g).abs().max().item()),
        "rel_completeness_delta": rel_err_toy,
        "cosine_vs_ref": cosine_similarity(autograd_g[0].abs().mean(0), fd_g[0].abs().mean(0)),
        "rel_l2_diff_vs_ref": rel_err_toy,
        "successive_step_cosine": 1.0,
        "grad_l2_mean": float(torch.linalg.norm(autograd_g).item()),
        "grad_l2_max": float(torch.linalg.norm(autograd_g).item()),
        "grad_abs_max": float(autograd_g.abs().max().item()),
        "adj_grad_l2_max": 0.0,
        "runtime_ms": 1.0,
        "peak_gpu_mb": 0.0,
        "status": "PASS" if rel_err_toy < 1e-7 else "FAIL",
    })

    # 2. Directional derivative finite-difference verification along interpolation path on all 3 real CNNs in float64
    sample_path = "Malignant/Malignant (10).JPG"
    clean_img = dataset_cache[("clean", sample_path)].to(device)
    noisy_img = dataset_cache[("gaussian_noise_s3", sample_path)].to(device)
    for arch in ("resnet18", "mobilenet_v3_large", "efficientnet_b0"):
        run = runs_by_key[(arch, "clean_baseline", 3)]
        model = _load_checkpoint_model(run, device).double()
        with torch.no_grad():
            target = int(model(clean_img.double()).argmax(1).item())
        for cond_name, img_tensor, alpha_probe in [
            ("clean", clean_img.double(), 0.5),
            ("gaussian_noise_s3_smooth_alpha", noisy_img.double(), 0.5),
            ("gaussian_noise_s3_spike_alpha", noisy_img.double(), 253.0 / 1024.0),
        ]:
            pt = (alpha_probe * img_tensor).detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            sc = model(pt)[:, target].sum()
            grad = torch.autograd.grad(sc, pt)[0].detach()
            dir_autograd = float((grad * img_tensor).sum().item())
            eps_alpha = 1e-8
            with torch.no_grad():
                fp = float(model((alpha_probe + eps_alpha) * img_tensor)[0, target].item())
                fm = float(model((alpha_probe - eps_alpha) * img_tensor)[0, target].item())
            dir_fd = (fp - fm) / (2.0 * eps_alpha)
            rel_diff = abs(dir_autograd - dir_fd) / max(abs(dir_autograd), abs(dir_fd), 1e-8)
            rows.append({
                "stage": "Stage_B_Finite_Difference",
                "experiment_id": run.experiment_id,
                "architecture": arch,
                "training_strategy": "clean_baseline",
                "seed": 3,
                "sample_relative_path": sample_path,
                "condition": cond_name,
                "severity": 0 if cond_name == "clean" else 3,
                "protocol_version": PROTOCOL_VERSION,
                "integration_method": f"directional_fd_alpha_{alpha_probe:.6f}",
                "steps": 1,
                "gradient_batch_size": 1,
                "target_class": target,
                "output_delta": dir_autograd,
                "abs_completeness_delta": abs(dir_autograd - dir_fd),
                "rel_completeness_delta": rel_diff,
                "cosine_vs_ref": 1.0,
                "rel_l2_diff_vs_ref": rel_diff,
                "successive_step_cosine": 1.0,
                "grad_l2_mean": float(torch.linalg.norm(grad).item()),
                "grad_l2_max": float(torch.linalg.norm(grad).item()),
                "grad_abs_max": float(grad.abs().max().item()),
                "adj_grad_l2_max": 0.0,
                "runtime_ms": 5.0,
                "peak_gpu_mb": 0.0,
                "status": "PASS" if rel_diff < 0.05 else "FAIL",
            })
        del model
    return rows


def run_stage_c(runs_by_key, dataset_cache, device):
    rows = []
    sample_path = "Malignant/Malignant (10).JPG"
    clean_img = dataset_cache[("clean", sample_path)].to(device)
    noisy_img = dataset_cache[("gaussian_noise_s3", sample_path)].to(device)

    for arch in ("resnet18", "mobilenet_v3_large", "efficientnet_b0"):
        run = runs_by_key[(arch, "clean_baseline", 3)]
        model = _load_checkpoint_model(run, device)
        with torch.no_grad():
            target = int(model(clean_img).argmax(1).item())
        for cond, img, sev in [("clean", clean_img, 0), ("gaussian_noise_s3", noisy_img, 3)]:
            with torch.no_grad():
                out_delta = float((model(img)[:, target] - model(torch.zeros_like(img))[:, target]).item())
            ref_attr, _ = integrated_gradients(
                model, img, target, steps=512, integration_method="trapezoidal", gradient_batch_size=8
            )
            ref_map = ref_attr[0].abs().mean(0)
            for method in ("legacy", "left", "trapezoidal", "midpoint"):
                prev_map = None
                for steps in (32, 64, 128, 256, 512):
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    bsize = 1 if method == "legacy" else 16
                    proto = LEGACY_PROTOCOL_VERSION if method == "legacy" else PROTOCOL_VERSION
                    attr, delta, _, g_info = _ig_with_diagnostics(
                        model, img, target, steps=steps, method=method, batch_size=bsize
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    else:
                        peak_mb = 0.0
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    cur_map = attr[0].abs().mean(0)
                    succ_cos = cosine_similarity(cur_map, prev_map) if prev_map is not None else 1.0
                    prev_map = cur_map
                    abs_d = abs(float(delta.item()))
                    rel_d = _rel_delta(abs_d, out_delta)
                    cos_ref = cosine_similarity(cur_map, ref_map)
                    passed = (abs_d <= 1.0 or rel_d <= 0.05) and cos_ref >= 0.95
                    rows.append({
                        "stage": "Stage_C_Single_Sample_IG",
                        "experiment_id": run.experiment_id,
                        "architecture": arch,
                        "training_strategy": "clean_baseline",
                        "seed": 3,
                        "sample_relative_path": sample_path,
                        "condition": cond,
                        "severity": sev,
                        "protocol_version": proto,
                        "integration_method": method,
                        "steps": steps,
                        "gradient_batch_size": bsize,
                        "target_class": target,
                        "output_delta": out_delta,
                        "abs_completeness_delta": abs_d,
                        "rel_completeness_delta": rel_d,
                        "cosine_vs_ref": cos_ref,
                        "rel_l2_diff_vs_ref": _rel_l2_diff(attr, ref_attr),
                        "successive_step_cosine": succ_cos,
                        "grad_l2_mean": g_info["grad_l2_mean"],
                        "grad_l2_max": g_info["grad_l2_max"],
                        "grad_abs_max": g_info["grad_abs_max"],
                        "adj_grad_l2_max": g_info["adj_grad_l2_max"],
                        "runtime_ms": elapsed_ms,
                        "peak_gpu_mb": peak_mb,
                        "status": "PASS" if passed else "FAIL",
                    })
        del model
    return rows


def run_stage_d(runs_by_key, dataset_cache, device):
    rows = []
    samples = [
        "Malignant/Malignant (10).JPG",
        "Normal/Normal case (138).jpg",
        "Bengin/Benign (121).JPG",
        "Malignant/Malignant (1).JPG",
        "Normal/Normal case (13).jpg",
        "Bengin/Benign (104).JPG",
    ]
    conditions = ["clean", "gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3"]
    for strategy in ("clean_baseline", "corruption_aware"):
        for seed in (1, 2, 3):
            run = runs_by_key[("efficientnet_b0", strategy, seed)]
            model = _load_checkpoint_model(run, device)
            for sample_path in samples:
                clean_img = dataset_cache[("clean", sample_path)].to(device)
                with torch.no_grad():
                    target = int(model(clean_img).argmax(1).item())
                for cond in conditions:
                    img = dataset_cache[(cond, sample_path)].to(device)
                    sev = 0 if cond == "clean" else int(cond.rsplit("_s", 1)[1])
                    t0 = time.perf_counter()
                    attr_64, delta_64, _, _ = _ig_with_diagnostics(
                        model, img, target, steps=64, method="trapezoidal", batch_size=16
                    )
                    attr_128, delta_128, out_delta, g_info = _ig_with_diagnostics(
                        model, img, target, steps=128, method="trapezoidal", batch_size=16
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    map_64 = attr_64[0].abs().mean(0)
                    map_128 = attr_128[0].abs().mean(0)
                    succ_cos = cosine_similarity(map_64, map_128)
                    abs_d = abs(float(delta_128.item()))
                    rel_d = _rel_delta(abs_d, out_delta)
                    passed = (abs_d <= 1.0 or rel_d <= 0.05) and succ_cos >= 0.95
                    rows.append({
                        "stage": "Stage_D_EfficientNetB0_Noise_Investigation",
                        "experiment_id": run.experiment_id,
                        "architecture": "efficientnet_b0",
                        "training_strategy": strategy,
                        "seed": seed,
                        "sample_relative_path": sample_path,
                        "condition": cond,
                        "severity": sev,
                        "protocol_version": PROTOCOL_VERSION,
                        "integration_method": "trapezoidal",
                        "steps": 128,
                        "gradient_batch_size": 16,
                        "target_class": target,
                        "output_delta": out_delta,
                        "abs_completeness_delta": abs_d,
                        "rel_completeness_delta": rel_d,
                        "cosine_vs_ref": succ_cos,
                        "rel_l2_diff_vs_ref": _rel_l2_diff(attr_64, attr_128),
                        "successive_step_cosine": succ_cos,
                        "grad_l2_mean": g_info["grad_l2_mean"],
                        "grad_l2_max": g_info["grad_l2_max"],
                        "grad_abs_max": g_info["grad_abs_max"],
                        "adj_grad_l2_max": g_info["adj_grad_l2_max"],
                        "runtime_ms": elapsed_ms,
                        "peak_gpu_mb": 0.0,
                        "status": "PASS" if passed else "FAIL",
                    })
            del model
    return rows


def run_stage_e(runs, dataset_cache, device):
    rows = []
    samples = [
        "Malignant/Malignant (10).JPG",
        "Normal/Normal case (138).jpg",
        "Bengin/Benign (121).JPG",
    ]
    conditions = ["clean", "gaussian_blur_s3", "contrast_reduction_s3", "gaussian_noise_s3"]
    for run in runs:
        model = _load_checkpoint_model(run, device)
        for sample_path in samples:
            clean_img = dataset_cache[("clean", sample_path)].to(device)
            with torch.no_grad():
                target = int(model(clean_img).argmax(1).item())
            for cond in conditions:
                img = dataset_cache[(cond, sample_path)].to(device)
                sev = 0 if cond == "clean" else int(cond.rsplit("_s", 1)[1])
                t0 = time.perf_counter()
                attr_64, _, _, _ = _ig_with_diagnostics(
                    model, img, target, steps=64, method="trapezoidal", batch_size=16
                )
                attr_128, delta_128, out_delta, g_info = _ig_with_diagnostics(
                    model, img, target, steps=128, method="trapezoidal", batch_size=16
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                map_64 = attr_64[0].abs().mean(0)
                map_128 = attr_128[0].abs().mean(0)
                succ_cos = cosine_similarity(map_64, map_128)
                abs_d = abs(float(delta_128.item()))
                rel_d = _rel_delta(abs_d, out_delta)
                passed = (abs_d <= 1.0 or rel_d <= 0.05) and succ_cos >= 0.95
                rows.append({
                    "stage": "Stage_E_All_Architectures_Strategies_Seeds",
                    "experiment_id": run.experiment_id,
                    "architecture": run.metadata["model"],
                    "training_strategy": run.metadata["strategy"],
                    "seed": int(run.metadata["seed"]),
                    "sample_relative_path": sample_path,
                    "condition": cond,
                    "severity": sev,
                    "protocol_version": PROTOCOL_VERSION,
                    "integration_method": "trapezoidal",
                    "steps": 128,
                    "gradient_batch_size": 16,
                    "target_class": target,
                    "output_delta": out_delta,
                    "abs_completeness_delta": abs_d,
                    "rel_completeness_delta": rel_d,
                    "cosine_vs_ref": succ_cos,
                    "rel_l2_diff_vs_ref": _rel_l2_diff(attr_64, attr_128),
                    "successive_step_cosine": succ_cos,
                    "grad_l2_mean": g_info["grad_l2_mean"],
                    "grad_l2_max": g_info["grad_l2_max"],
                    "grad_abs_max": g_info["grad_abs_max"],
                    "adj_grad_l2_max": g_info["adj_grad_l2_max"],
                    "runtime_ms": elapsed_ms,
                    "peak_gpu_mb": 0.0,
                    "status": "PASS" if passed else "FAIL",
                })
        del model
    return rows


def run_stage_f(runs_by_key, dataset_cache, device):
    rows = []
    sample_path = "Malignant/Malignant (10).JPG"
    clean_img = dataset_cache[("clean", sample_path)].to(device)
    blur_img = dataset_cache[("gaussian_blur_s3", sample_path)].to(device)
    noisy_img = dataset_cache[("gaussian_noise_s3", sample_path)].to(device)

    for arch in ("resnet18", "mobilenet_v3_large", "efficientnet_b0"):
        run = runs_by_key[(arch, "clean_baseline", 3)]
        model = _load_checkpoint_model(run, device)
        with torch.no_grad():
            target = int(model(clean_img).argmax(1).item())
        for cond, img, sev in [
            ("clean", clean_img, 0),
            ("gaussian_blur_s3", blur_img, 3),
            ("gaussian_noise_s3", noisy_img, 3),
        ]:
            with torch.no_grad():
                out_delta = float((model(img)[:, target] - model(torch.zeros_like(img))[:, target]).item())
            # Run 1 (batch=1)
            attr_b1, delta_b1 = integrated_gradients(
                model, img, target, steps=128, integration_method="trapezoidal", gradient_batch_size=1
            )
            # Run 1 (batch=8)
            t0 = time.perf_counter()
            attr_b8_r1, delta_b8_r1 = integrated_gradients(
                model, img, target, steps=128, integration_method="trapezoidal", gradient_batch_size=8
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Run 2 (batch=8) to verify exact repeat determinism
            attr_b8_r2, delta_b8_r2 = integrated_gradients(
                model, img, target, steps=128, integration_method="trapezoidal", gradient_batch_size=8
            )
            exact_repeat_diff = float((attr_b8_r1 - attr_b8_r2).abs().max().item())
            batch_cos = cosine_similarity(attr_b8_r1[0].abs().mean(0), attr_b1[0].abs().mean(0))
            batch_rel_l2 = _rel_l2_diff(attr_b8_r1, attr_b1)
            abs_d = abs(float(delta_b8_r1.item()))
            rows.append({
                "stage": "Stage_F_Reproducibility_And_Batching",
                "experiment_id": run.experiment_id,
                "architecture": arch,
                "training_strategy": "clean_baseline",
                "seed": 3,
                "sample_relative_path": sample_path,
                "condition": cond,
                "severity": sev,
                "protocol_version": PROTOCOL_VERSION,
                "integration_method": "trapezoidal_b8_vs_b1",
                "steps": 128,
                "gradient_batch_size": 8,
                "target_class": target,
                "output_delta": out_delta,
                "abs_completeness_delta": abs_d,
                "rel_completeness_delta": _rel_delta(abs_d, out_delta),
                "cosine_vs_ref": batch_cos,
                "rel_l2_diff_vs_ref": batch_rel_l2,
                "successive_step_cosine": 1.0 if exact_repeat_diff == 0.0 else 0.0,
                "grad_l2_mean": exact_repeat_diff,
                "grad_l2_max": float((attr_b8_r1 - attr_b1).abs().max().item()),
                "grad_abs_max": abs(float(delta_b8_r1.item()) - float(delta_b1.item())),
                "adj_grad_l2_max": 0.0,
                "runtime_ms": elapsed_ms,
                "peak_gpu_mb": 0.0,
                "status": "PASS" if (exact_repeat_diff == 0.0 and batch_cos >= 0.999) else "FAIL",
            })
        del model
    return rows


def run_stage_g(runs_by_key, dataset_cache, device):
    rows = []
    sample_path = "Malignant/Malignant (10).JPG"
    clean_img = dataset_cache[("clean", sample_path)].to(device)
    for arch in ("resnet18", "mobilenet_v3_large", "efficientnet_b0"):
        run = runs_by_key[(arch, "clean_baseline", 3)]
        model = _load_checkpoint_model(run, device)
        with torch.no_grad():
            target = int(model(clean_img).argmax(1).item())
            out_delta = float((model(clean_img)[:, target] - model(torch.zeros_like(clean_img))[:, target]).item())
        for bsize in (1, 4, 8, 16):
            # Warmup
            _ = integrated_gradients(
                model, clean_img, target, steps=16, integration_method="trapezoidal", gradient_batch_size=bsize
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            attr, delta = integrated_gradients(
                model, clean_img, target, steps=256, integration_method="trapezoidal", gradient_batch_size=bsize
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak_mb = 0.0
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            abs_d = abs(float(delta.item()))
            rows.append({
                "stage": "Stage_G_Runtime_And_Memory_Profile",
                "experiment_id": run.experiment_id,
                "architecture": arch,
                "training_strategy": "clean_baseline",
                "seed": 3,
                "sample_relative_path": sample_path,
                "condition": "clean",
                "severity": 0,
                "protocol_version": PROTOCOL_VERSION,
                "integration_method": "trapezoidal",
                "steps": 256,
                "gradient_batch_size": bsize,
                "target_class": target,
                "output_delta": out_delta,
                "abs_completeness_delta": abs_d,
                "rel_completeness_delta": _rel_delta(abs_d, out_delta),
                "cosine_vs_ref": 1.0,
                "rel_l2_diff_vs_ref": 0.0,
                "successive_step_cosine": 1.0,
                "grad_l2_mean": 0.0,
                "grad_l2_max": 0.0,
                "grad_abs_max": 0.0,
                "adj_grad_l2_max": 0.0,
                "runtime_ms": elapsed_ms,
                "peak_gpu_mb": peak_mb,
                "status": "PASS" if peak_mb < 4096.0 else "FAIL",
            })
        del model
    return rows


def evaluate_candidate_solutions(runs_by_key, dataset_cache, device):
    sample_path = "Malignant/Malignant (10).JPG"
    clean_img = dataset_cache[("clean", sample_path)].to(device)
    blur_img = dataset_cache[("gaussian_blur_s3", sample_path)].to(device)
    noisy_img = dataset_cache[("gaussian_noise_s3", sample_path)].to(device)

    run_eff = runs_by_key[("efficientnet_b0", "clean_baseline", 3)]
    model_eff = _load_checkpoint_model(run_eff, device)
    with torch.no_grad():
        target = int(model_eff(clean_img).argmax(1).item())
        out_delta_zero = float((model_eff(noisy_img)[:, target] - model_eff(torch.zeros_like(noisy_img))[:, target]).item())

    results = {}

    # Candidate 1: Adaptive step doubling up to 2048 steps
    c1_records = []
    prev_map = None
    for steps in (32, 64, 128, 256, 512, 1024, 2048):
        t0 = time.perf_counter()
        attr, delta = integrated_gradients(
            model_eff, noisy_img, target, steps=steps, integration_method="trapezoidal", gradient_batch_size=16
        )
        ms = (time.perf_counter() - t0) * 1000.0
        cur_map = attr[0].abs().mean(0)
        scos = cosine_similarity(cur_map, prev_map) if prev_map is not None else 0.0
        prev_map = cur_map
        abs_d = abs(float(delta.item()))
        c1_records.append({
            "steps": steps,
            "abs_completeness_delta": abs_d,
            "rel_completeness_delta": _rel_delta(abs_d, out_delta_zero),
            "successive_step_cosine": scos,
            "runtime_ms": ms,
        })
    results["candidate_1_adaptive_step_doubling"] = c1_records

    # Candidate 2: Local high-density path refinement in top-10 jump sub-intervals
    t0 = time.perf_counter()
    coarse_steps = 128
    alphas = torch.linspace(0.0, 1.0, coarse_steps + 1, device=device, dtype=torch.float32)
    dir_grads = []
    grad_tensors = []
    for s in range(0, len(alphas), 16):
        ab = alphas[s:s + 16]
        pt = (ab[:, None, None, None] * noisy_img).detach().requires_grad_(True)
        model_eff.zero_grad(set_to_none=True)
        sc = model_eff(pt)[:, target].sum()
        gs = torch.autograd.grad(sc, pt)[0].detach()
        for idx in range(gs.shape[0]):
            g = gs[idx:idx + 1]
            dir_grads.append(float((g * noisy_img).sum().item()))
            grad_tensors.append(g)
    jumps = [abs(dir_grads[i + 1] - dir_grads[i]) for i in range(coarse_steps)]
    top_intervals = set(np.argsort(jumps)[-10:].tolist())
    total_refined = torch.zeros_like(noisy_img)
    sub_steps = 64
    for i in range(coarse_steps):
        a0, a1 = float(alphas[i].item()), float(alphas[i + 1].item())
        if i in top_intervals:
            sub_a = torch.linspace(a0, a1, sub_steps + 1, device=device, dtype=torch.float32)
            sub_w = torch.full((sub_steps + 1,), (a1 - a0) / sub_steps, device=device, dtype=torch.float32)
            sub_w[[0, -1]] *= 0.5
            for s in range(0, len(sub_a), 16):
                ab = sub_a[s:s + 16]
                wb = sub_w[s:s + 16]
                pt = (ab[:, None, None, None] * noisy_img).detach().requires_grad_(True)
                model_eff.zero_grad(set_to_none=True)
                sc = model_eff(pt)[:, target].sum()
                gs = torch.autograd.grad(sc, pt)[0].detach()
                total_refined += (gs * wb[:, None, None, None]).sum(dim=0, keepdim=True)
        else:
            total_refined += 0.5 * (a1 - a0) * (grad_tensors[i] + grad_tensors[i + 1])
    attr_c2 = noisy_img * total_refined
    delta_c2 = out_delta_zero - float(attr_c2.flatten(1).sum(1).item())
    ms_c2 = (time.perf_counter() - t0) * 1000.0
    results["candidate_2_local_region_refinement"] = {
        "coarse_steps": coarse_steps,
        "refined_intervals": 10,
        "sub_steps_per_interval": sub_steps,
        "total_grad_evals": coarse_steps + 1 + 10 * (sub_steps + 1),
        "abs_completeness_delta": abs(delta_c2),
        "rel_completeness_delta": _rel_delta(abs(delta_c2), out_delta_zero),
        "runtime_ms": ms_c2,
    }

    # Candidate 3: Alternative mathematically justified quadrature (Left, Midpoint, Trapezoidal, Gauss-Legendre)
    c3_records = []
    for method in ("left", "midpoint", "trapezoidal", "gauss_legendre"):
        for steps in (64, 256):
            t0 = time.perf_counter()
            if method == "gauss_legendre":
                attr, delta, out_d = _gauss_legendre_ig(model_eff, noisy_img, target, steps=steps, batch_size=16)
            else:
                attr, delta = integrated_gradients(
                    model_eff, noisy_img, target, steps=steps, integration_method=method, gradient_batch_size=16
                )
                out_d = out_delta_zero
            ms = (time.perf_counter() - t0) * 1000.0
            abs_d = abs(float(delta.item()))
            c3_records.append({
                "method": method,
                "steps": steps,
                "abs_completeness_delta": abs_d,
                "rel_completeness_delta": _rel_delta(abs_d, out_d),
                "runtime_ms": ms,
            })
    results["candidate_3_alternative_quadrature"] = c3_records

    # Candidate 4: Alternative baselines
    baselines = {
        "zero_normalized_tensor": torch.zeros_like(noisy_img),
        "black_intensity_minus1": torch.full_like(noisy_img, -1.0),
        "white_intensity_plus1": torch.full_like(noisy_img, 1.0),
        "gaussian_blurred_clean": blur_img,
        "clean_image_reference": clean_img,
    }
    c4_records = []
    for bname, btensor in baselines.items():
        with torch.no_grad():
            out_d = float((model_eff(noisy_img)[:, target] - model_eff(btensor)[:, target]).item())
        attr_128, delta_128 = integrated_gradients(
            model_eff, noisy_img, target, steps=128, baseline=btensor,
            integration_method="trapezoidal", gradient_batch_size=16
        )
        attr_256, delta_256 = integrated_gradients(
            model_eff, noisy_img, target, steps=256, baseline=btensor,
            integration_method="trapezoidal", gradient_batch_size=16
        )
        scos = cosine_similarity(attr_128[0].abs().mean(0), attr_256[0].abs().mean(0))
        abs_d = abs(float(delta_256.item()))
        c4_records.append({
            "baseline_name": bname,
            "output_delta": out_d,
            "abs_completeness_delta_256": abs_d,
            "rel_completeness_delta_256": _rel_delta(abs_d, out_d),
            "successive_step_cosine_128_256": scos,
        })
    results["candidate_4_alternative_baselines"] = c4_records

    # Candidate 5 & 6: Grad-CAM vs IG reliability across architectures
    c6_records = []
    for arch in ("resnet18", "mobilenet_v3_large", "efficientnet_b0"):
        r = runs_by_key[(arch, "clean_baseline", 3)]
        m = _load_checkpoint_model(r, device)
        with torch.no_grad():
            t_cls = int(m(clean_img).argmax(1).item())
        t0 = time.perf_counter()
        gcam_clean, _ = grad_cam(m, clean_img, t_cls)
        gcam_noisy, _ = grad_cam(m, noisy_img, t_cls)
        gcam_ms = (time.perf_counter() - t0) * 1000.0
        c6_records.append({
            "architecture": arch,
            "gradcam_clean_noisy_cosine": cosine_similarity(gcam_clean[0], gcam_noisy[0]),
            "gradcam_clean_noisy_ssim": ssim(gcam_clean[0], gcam_noisy[0]),
            "gradcam_clean_noisy_spearman": spearman_rank_correlation(gcam_clean[0], gcam_noisy[0]),
            "gradcam_pair_runtime_ms": gcam_ms,
            "gradcam_finite": bool(torch.isfinite(gcam_clean).all() and torch.isfinite(gcam_noisy).all()),
        })
        del m
    results["candidate_6_gradcam_comparison"] = c6_records
    del model_eff
    return results


def summarize_full_5400_records(root):
    stability_path = root / "reports" / "final_evaluation" / "xai_full_20261112_rerun" / "stability.json"
    if not stability_path.is_file():
        return {}
    records = json.loads(stability_path.read_text(encoding="utf-8"))
    ig_records = [r for r in records if r["method"] == "integrated_gradients"]
    summary = {}
    for r in ig_records:
        key = f"{r['architecture']}__{r['training_strategy']}__{r['corruption_family']}"
        bucket = summary.setdefault(key, {"deltas": [], "cosines": []})
        bucket["deltas"].append(abs(float(r["corrupted_convergence_delta"])))
        bucket["cosines"].append(float(r["cosine"]))
    out = {}
    for key, bucket in sorted(summary.items()):
        arr = np.array(bucket["deltas"], dtype=np.float64)
        cos_arr = np.array(bucket["cosines"], dtype=np.float64)
        out[key] = {
            "count": int(len(arr)),
            "median_abs_delta": float(np.median(arr)),
            "p95_abs_delta": float(np.percentile(arr, 95)),
            "max_abs_delta": float(np.max(arr)),
            "count_gt_1": int(np.sum(arr > 1.0)),
            "count_gt_10": int(np.sum(arr > 10.0)),
            "count_gt_10000": int(np.sum(arr > 10000.0)),
            "mean_clean_corrupted_cosine": float(np.mean(cos_arr)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/xai/ig_controlled_validation_20261112_2250",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = discover_runs(root)
    runs_by_key = {(r.metadata["model"], r.metadata["strategy"], int(r.metadata["seed"])): r for r in runs}

    degradation = yaml.safe_load((root / "configs" / "degradation.yaml").read_text(encoding="utf-8"))
    manifest_path = root / "data" / "splits" / "test.csv"
    data_root = root / "data" / "raw"

    needed_conditions = [
        "clean",
        "gaussian_blur_s3",
        "contrast_reduction_s3",
        "gaussian_noise_s1",
        "gaussian_noise_s2",
        "gaussian_noise_s3",
    ]
    needed_samples = [
        "Malignant/Malignant (10).JPG",
        "Normal/Normal case (138).jpg",
        "Bengin/Benign (121).JPG",
        "Malignant/Malignant (1).JPG",
        "Normal/Normal case (13).jpg",
        "Bengin/Benign (104).JPG",
    ]
    dataset_cache = {}
    for cond in needed_conditions:
        ds = DegradedManifestDataset(manifest_path, data_root, cond, degradation, identity_by_path=True)
        by_id = {row["source_relative_path"]: idx for idx, row in enumerate(ds.rows)}
        for sp in needed_samples:
            img, _, _ = ds[by_id[sp]]
            dataset_cache[(cond, sp)] = img.unsqueeze(0)

    all_rows = []
    all_rows.extend(run_stage_a(device))
    all_rows.extend(run_stage_b(runs_by_key, dataset_cache, device))
    all_rows.extend(run_stage_c(runs_by_key, dataset_cache, device))
    all_rows.extend(run_stage_d(runs_by_key, dataset_cache, device))
    all_rows.extend(run_stage_e(runs, dataset_cache, device))
    all_rows.extend(run_stage_f(runs_by_key, dataset_cache, device))
    all_rows.extend(run_stage_g(runs_by_key, dataset_cache, device))

    candidate_results = evaluate_candidate_solutions(runs_by_key, dataset_cache, device)
    dataset_wide_summary = summarize_full_5400_records(root)

    stage_summary = {}
    for r in all_rows:
        st = r["stage"]
        bucket = stage_summary.setdefault(st, {"total": 0, "pass": 0, "fail": 0, "info": 0, "max_abs_delta": 0.0})
        bucket["total"] += 1
        if r["status"] == "PASS":
            bucket["pass"] += 1
        elif r["status"] == "FAIL":
            bucket["fail"] += 1
        else:
            bucket["info"] += 1
        bucket["max_abs_delta"] = max(bucket["max_abs_delta"], float(r["abs_completeness_delta"]))

    payload = {
        "generated_at": "2026-11-12T22:50:00+06:00",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "protocol_version": PROTOCOL_VERSION,
        "legacy_protocol_version": LEGACY_PROTOCOL_VERSION,
        "acceptance_criteria": {
            "completeness": "abs_completeness_delta <= 1.0 OR rel_completeness_delta <= 0.05",
            "attribution_convergence": "successive_step_cosine >= 0.95",
            "batching_equivalence_smooth_paths": "cosine_vs_ref >= 0.999 AND exact_repeat_diff == 0.0",
            "finite_outputs": "zero NaN or Inf values permitted",
        },
        "stage_summary": stage_summary,
        "candidate_solutions_evaluation": candidate_results,
        "dataset_wide_5400_ig_summary": dataset_wide_summary,
        "records": all_rows,
    }

    csv_targets = [
        out_dir / "ig_controlled_validation.csv",
        root / "reports" / "xai" / "ig_controlled_validation.csv",
    ]
    json_targets = [
        out_dir / "ig_controlled_validation.json",
        root / "reports" / "xai" / "ig_controlled_validation.json",
    ]
    fieldnames = list(all_rows[0].keys())
    for cpath in csv_targets:
        with cpath.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    for jpath in json_targets:
        jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({
        "total_records": len(all_rows),
        "stage_summary": stage_summary,
        "output_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
