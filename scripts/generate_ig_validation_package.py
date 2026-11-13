"""Package bounded IG evidence into versioned validation deliverables."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, content):
    Path(path).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrature-dir", required=True)
    parser.add_argument("--path-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    quadrature = _read_json(root / args.quadrature_dir / "ig_quadrature_diagnostics.json")
    path = _read_json(root / args.path_dir / "efficientnet_path_instability.json")
    output = root / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for record in quadrature["records"]:
        records.append({
            **record,
            "validation_source": "ig_quadrature_validation",
            "protocol_version": "ig_v2_quadrature_fp32",
        })
    with (output / "ig_validation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    extreme = [r for r in records if abs(r["completeness_delta"]) > 1e4]
    nonfinite = [
        r for r in records
        if not r["attribution_finite"] or not r["delta_finite"]
    ]
    method_summary = {}
    for method in sorted({r["integration_method"] for r in records}):
        subset = [r for r in records if r["integration_method"] == method]
        method_summary[method] = {
            "records": len(subset),
            "extreme_abs_delta_over_1e4": sum(abs(r["completeness_delta"]) > 1e4 for r in subset),
            "max_abs_delta": max(abs(r["completeness_delta"]) for r in subset),
            "min_successive_map_cosine": min(
                r["map_cosine_to_next_higher_step"] for r in subset
                if r["map_cosine_to_next_higher_step"] is not None
            ),
        }
    summary = {
        "protocol_version": "ig_v2_quadrature_fp32",
        "historical_artifacts_preserved": True,
        "full_xai_rerun_executed": False,
        "quadrature_source": args.quadrature_dir,
        "path_instability_source": args.path_dir,
        "device": quadrature["device"],
        "coverage": {
            "runs": quadrature["run_count"],
            "case_conditions": quadrature["coverage_count"],
            "records": len(records),
            "architectures": ["mobilenet_v3_large", "efficientnet_b0", "resnet18"],
            "strategies": ["clean_baseline", "corruption_aware"],
            "seeds": [1],
            "conditions": quadrature["configuration"]["conditions"],
        },
        "acceptance_policy": {
            "completeness": "Report absolute and relative errors; no universal absolute threshold is accepted across differing logit scales.",
            "map_convergence": "Require successive-step cosine and relative map difference to be stable; finite values alone are insufficient.",
            "reproducibility": "Repeated deterministic executions must agree within explicitly recorded FP32 GPU tolerance.",
            "model_behavior": "Extreme logits or gradients are evidence requiring review, not values to clamp or discard.",
            "status": "predefined_before_interpretation; not approved for full evaluation",
        },
        "nonfinite_records": len(nonfinite),
        "extreme_records_abs_delta_over_1e4": len(extreme),
        "method_summary": method_summary,
        "gpu_profile": {
            "start": quadrature["gpu_start"],
            "end": quadrature["gpu_end"],
            "batching_profile": "batch 8 was 8.15x faster than batch 1 in the recorded 128-step benchmark",
        },
        "path_summary": {
            "records": path["record_count"],
            "runs": path["run_count"],
            "reproducibility": path["reproducibility"],
        },
        "decision": "FAIL",
    }
    (output / "ig_validation_results.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / "ig_validation_policy.yaml").write_text(
        """protocol_version: ig_v2_quadrature_fp32
baseline: zero_normalized_input
target: clean_image_predicted_class
dtype: float32
approval:
  require_completeness_and_map_convergence: true
  reject_nonfinite_values: true
  reject_unresolved_extreme_path_behavior: true
  universal_absolute_delta_threshold: false
""",
        encoding="utf-8",
    )
    _write(output / "ig_root_cause_report.md", _root_cause(path, quadrature))
    _write(output / "ig_implementation_audit.md", _implementation_audit())
    _write(output / "ig_validation_report.md", _validation_report(summary))
    _write(output / "full_xai_rerun_readiness.md", _readiness(summary))
    print(json.dumps({"output_dir": str(output), "records": len(records), "decision": "FAIL"}))


def _root_cause(path, quadrature):
    return f"""# Integrated Gradients Root-Cause Report

## Decision: FAIL

### Confirmed causes and evidence

1. The historical implementation used `torch.linspace(0, 1, steps)` while
   describing the rule as left-endpoint rectangular integration. It included
   the right endpoint and did not represent the standard left rule.
2. The corrected explicit quadrature implementation removes that ambiguity and
   records method and step count, but the EfficientNet-B0 anomaly persists.
3. The layer investigation covered {path['run_count']} EfficientNet checkpoints,
   {path['record_count']} layer records, clean and Gaussian-noise severities,
   and found finite but extreme logits, activations, and gradients.
4. The bounded quadrature validation produced {len(quadrature['records'])}
   records with {sum(abs(r['completeness_delta']) > 1e4 for r in quadrature['records'])}
   records over the prior anomaly screen.

### Rejected hypotheses

- NaN/Inf input corruption: not observed.
- Non-finite checkpoint parameters: not observed.
- FP16/autocast overflow: not used by the IG path.
- A universal quadrature cure: left, trapezoidal, and midpoint methods all
  retained extreme EfficientNet Gaussian-noise behavior.
- A single proven BatchNorm defect: evaluation mode was used and no single
  defective layer was established.

### Remaining uncertainty

The evidence supports an interaction between model sensitivity under Gaussian
noise and highly nonlinear zero-baseline paths. It does not establish whether
the research should change baseline, retrain models, or redefine the test
protocol. Those are methodological decisions requiring separate approval.
"""


def _implementation_audit():
    return """# Integrated Gradients Implementation Audit

## Historical implementation

The historical function used zero baseline, model evaluation mode, autograd
with respect to each interpolated input, and the same target logit for output
delta and attribution sum. Its integration grid included alpha=1 while using a
steps-sized left-rule average, which is an endpoint/weighting defect relative
to the standard left Riemann rule.

## Corrected version

`src/xai/integrated_gradients/method.py` now exposes explicit `left`,
`trapezoidal`, and `midpoint` methods. Each method has weights summing to one,
uses FP32 inputs and baseline, preserves attribution shape, and computes
completeness against the same target logit. Interpolation points may be
processed in bounded gradient batches without changing the configured
quadrature weights.

`src/xai/integrated_gradients/v2.py` provides the versioned
`ig_v2_quadrature_fp32` entry point. `legacy.py` preserves the previous
algorithm for historical comparison. Existing callers remain compatible.

## Mathematical and numerical tests

Tests cover endpoint grids, weight normalization, analytic linear and quadratic
toy models, finite-difference gradient agreement, FP32 conversion, shape
preservation, deterministic repeated execution, and legacy availability.
"""


def _validation_report(summary):
    return f"""# Integrated Gradients Validation Report

## Final decision: FAIL

The corrected implementation is mathematically explicit and passes the
software tests, but it is not approved for the full XAI evaluation because
bounded real-checkpoint evidence still contains unresolved EfficientNet-B0
Gaussian-noise instability.

## Coverage

- Runs: {summary['coverage']['runs']}
- Case-condition evaluations: {summary['coverage']['case_conditions']}
- Numerical records: {summary['coverage']['records']}
- Architectures: MobileNetV3-Large, EfficientNet-B0, ResNet18
- Strategies: clean baseline and corruption-aware
- Seed coverage: seed 1 in quadrature validation; all seeds in the separate
  EfficientNet path investigation
- Conditions: clean, Gaussian noise s1/s2/s3, Gaussian blur s1, contrast
  reduction s1
- Methods: left, trapezoidal, midpoint
- Steps: 32, 64, 128, 256, 512

## Acceptance policy

The policy was defined before interpreting final outcomes. Approval requires
both completeness behavior relative to target-logit scale and successive-step
attribution-map convergence, plus reproducibility. No universal absolute delta
threshold is applied across models with different output scales. Extreme
finite values are retained as failures, not clamped or removed.

## Results

- Non-finite records: {summary['nonfinite_records']}
- Records with absolute delta over 10,000: {summary['extreme_records_abs_delta_over_1e4']}
- Method summary: `{json.dumps(summary['method_summary'], sort_keys=True)}`
- GPU: `{summary['device']}`
- Recorded batch-8 speedup: 8.15x; peak allocation approximately 784 MB
  of 8,188 MB.

The prior path investigation found deterministic, finite but extreme logits and
gradients, with earlier backward amplification and later activation growth.
Quadrature changes alone did not remove the behavior.

## Tests

Focused Sector 09 tests and the full regression suite passed after the
implementation changes. The existing Windows pytest cache permission warning
remains non-fatal.

## Limitations

The bounded quadrature subset uses seed 1 and does not establish all-seed,
all-condition convergence for every architecture. It is sufficient to reject
unconditional full-run approval, not to prove that no future protocol can be
validated.
"""


def _readiness(summary):
    return """# Full XAI Rerun Readiness

## Status: NOT APPROVED

The full 18-experiment XAI rerun must not be launched from this validation
cycle. The versioned implementation is software-ready for further controlled
experiments, but scientific approval is blocked by unresolved EfficientNet-B0
Gaussian-noise path instability.

Required next experiment:

1. Pre-register the acceptance rule and diagnostic subset.
2. Compare the versioned quadrature methods and step counts on all EfficientNet
   seeds and Gaussian-noise severities.
3. Retain all failed records and layer diagnostics.
4. Decide separately whether a baseline or model/training methodological change
   is scientifically permitted.
5. Only after the targeted protocol passes should the full XAI evaluation be
   considered.

Historical XAI outputs, checkpoints, manifests, corruption definitions, and
BiomedCLIP artifacts remain preserved.
"""


if __name__ == "__main__":
    main()
