"""Generate Sector 10 completeness and provenance reports without experiments."""

import argparse
import json
from pathlib import Path

from analysis.final_analysis import ingest_results, validate_biomedclip
from analysis.reporting.final_report import generate_report


CONDITIONS = (
    "clean", "gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3",
    "gaussian_blur_s1", "gaussian_blur_s2", "gaussian_blur_s3",
    "contrast_reduction_s1", "contrast_reduction_s2", "contrast_reduction_s3",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/final_analysis")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prediction_path = root / "reports/biomedclip/predictions.jsonl"
    records = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    matrix_audit = ingest_results([])
    biomed_audit = validate_biomedclip(records, CONDITIONS)
    output = generate_report(root / args.output_dir, matrix_audit, biomed_audit)
    print(json.dumps({"output": str(output), "missing_cnn_runs": len(matrix_audit["missing"])}))


if __name__ == "__main__":
    raise SystemExit(main())
