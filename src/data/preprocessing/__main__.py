"""Run Sector 03 validation or splitting."""

import argparse
import csv
import json
from pathlib import Path
import yaml

from data.preprocessing import load_audit_metadata, validate_split_ratios
from data.splitting import split_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--audit-json", default="reports/dataset_audit/run_20261111T053825Z/dataset_audit.json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    dataset_config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))["dataset"]
    preprocessing = yaml.safe_load((root / args.preprocessing_config).read_text(encoding="utf-8"))
    split = preprocessing["split"]
    records, groups = load_audit_metadata(root / args.audit_json, dataset_config["expected_classes"],
                                          dataset_config.get("class_aliases", {}))
    validate_split_ratios(split["train_ratio"], split["validation_ratio"], split["test_ratio"])
    cross_class = {int(row["duplicate_group"]) for row in __import__("json").loads(
        (root / args.audit_json).read_text(encoding="utf-8"))["duplicate_images"]
                   if row["cross_class_duplicate"]}
    if args.validate_only:
        print(f"metadata_records={len(records)}")
        print(f"cross_class_duplicate_groups={len(cross_class)}")
        print(f"cross_class_duplicate_policy={split['cross_class_duplicate_policy']}")
        return 0
    manifests = split_records(records, split["train_ratio"], split["validation_ratio"], split["test_ratio"],
                               split["random_seed"], groups, cross_class, split["cross_class_duplicate_policy"])
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        _write_csv(output / f"{split_name}.csv", [row for row in manifests if row["split"] == split_name])
    _write_csv(output / "combined.csv", manifests)
    summary = {
        "total_input_records": len(records),
        "included_records": len(manifests),
        "excluded_records": len(records) - len(manifests),
        "cross_class_duplicate_groups": len(cross_class),
        "cross_class_duplicate_policy": split["cross_class_duplicate_policy"],
        "ratios": {"train": split["train_ratio"], "validation": split["validation_ratio"], "test": split["test_ratio"]},
        "random_seed": split["random_seed"],
        "patient_grouping_available": split["patient_grouping_available"],
        "class_mapping": preprocessing.get("class_mapping", {}),
        "source_audit": str((root / args.audit_json).resolve()),
    }
    (output / "split_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "preprocessing_metadata.json").write_text(json.dumps({
        "image_size": preprocessing["image"]["size"],
        "color_mode": preprocessing["image"]["color_mode"],
        "normalization": preprocessing["normalization"],
        "augmentation": preprocessing["augmentation"],
        "source_images_modified": False,
    }, indent=2), encoding="utf-8")
    (output / "split_report.md").write_text(_report(summary, manifests), encoding="utf-8")
    print(f"split_records={len(manifests)}")
    print(f"excluded_records={len(records) - len(manifests)}")
    print(f"output_dir={output}")
    return 0


def _write_csv(path, rows):
    fields = ["sample_id", "source_relative_path", "canonical_class", "class_index", "split", "sha256", "duplicate_group_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _report(summary, manifests):
    counts = {}
    for row in manifests:
        key = (row["split"], row["canonical_class"])
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# Sector 03 Split Report", "",
        f"- Included records: **{summary['included_records']}**",
        f"- Excluded records: **{summary['excluded_records']}**",
        f"- Duplicate policy: **{summary['cross_class_duplicate_policy']}**",
        f"- Random seed: **{summary['random_seed']}**", "",
        "| Split | Class | Count |", "|---|---|---:|",
    ]
    lines.extend(f"| {split} | {class_name} | {count} |"
                 for (split, class_name), count in sorted(counts.items()))
    lines.extend(["", "Splits are image-level because patient/examination identifiers are unavailable.",
                  "Source images were not modified; SHA-256 identifies byte-identical files only."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
