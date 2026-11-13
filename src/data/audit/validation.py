"""Post-audit investigation and source-integrity validation."""

from collections import defaultdict
from hashlib import sha256
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def load_audit(path: Path) -> dict:
    """Load a previously generated audit without rerunning or changing it."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def investigate_audit(audit: dict) -> dict:
    """Derive duplicate, class, dimension, and reproducibility findings."""
    records = audit["image_metadata"]
    total = len(records)
    duplicate_groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["validation_status"] == "valid" and record["sha256"]:
            duplicate_groups[record["sha256"]].append(record)
    duplicate_groups = {
        digest: sorted(rows, key=lambda row: row["relative_path"])
        for digest, rows in duplicate_groups.items()
        if len(rows) > 1
    }
    cross_class = [
        {
            "group_id": index,
            "sha256": digest,
            "file_count": len(rows),
            "classes": sorted({row["class_name"] for row in rows}),
            "potential_label_conflict": len({row["class_name"] for row in rows}) > 1,
            "files": [
                {"relative_path": row["relative_path"], "class_name": row["class_name"]}
                for row in rows
            ],
        }
        for index, (digest, rows) in enumerate(sorted(duplicate_groups.items()), start=1)
        if len({row["class_name"] for row in rows}) > 1
    ]
    counts = {
        row["class_name"]: sum(1 for record in records if record["class_name"] == row["class_name"])
        for row in audit["class_distribution"]
    }
    majority_class = max(counts, key=counts.get) if counts else ""
    minority_class = min(counts, key=counts.get) if counts else ""
    majority_count = counts.get(majority_class, 0)
    minority_count = counts.get(minority_class, 0)
    dimensions = defaultdict(int)
    for record in records:
        if record["validation_status"] == "valid":
            dimensions[(int(record["width"]), int(record["height"]))] += 1
    dimension_rows = [
        {
            "width": width,
            "height": height,
            "count": count,
            "percentage": round(count * 100 / total, 4) if total else 0.0,
        }
        for (width, height), count in sorted(dimensions.items())
    ]
    source_checks = []
    for record in records:
        path = Path(record["absolute_path"])
        current_hash = sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        source_checks.append({
            "relative_path": record["relative_path"],
            "exists": path.is_file(),
            "recorded_sha256": record["sha256"],
            "current_sha256": current_hash,
            "unchanged": bool(current_hash) and current_hash == record["sha256"],
        })
    metadata_hashes = sum(bool(record["sha256"]) for record in records)
    invalid_records = [record for record in records if record["validation_status"] != "valid"]
    return {
        "audit_source": audit.get("dataset_root", ""),
        "audit_status": audit.get("status", ""),
        "audit_timestamp": audit.get("audit_timestamp", ""),
        "total_metadata_records": total,
        "metadata_hash_count": metadata_hashes,
        "valid_image_count": audit.get("valid_image_count", 0),
        "invalid_image_count": audit.get("invalid_image_count", 0),
        "invalid_metadata_count": len(invalid_records),
        "duplicate_group_count": len(duplicate_groups),
        "cross_class_duplicate_group_count": len(cross_class),
        "cross_class_duplicate_groups": cross_class,
        "class_distribution": [
            {
                "class_name": name,
                "count": count,
                "percentage": round(count * 100 / total, 4) if total else 0.0,
                "historical_count": audit.get("historical_class_counts", {}).get(name),
            }
            for name, count in counts.items()
        ],
        "majority_class": majority_class,
        "minority_class": minority_class,
        "majority_to_minority_ratio": round(majority_count / minority_count, 6) if minority_count else None,
        "majority_minus_minority": majority_count - minority_count,
        "dimension_distribution": dimension_rows,
        "distinct_dimension_count": len(dimension_rows),
        "minimum_width": min((row["width"] for row in dimension_rows), default=None),
        "minimum_height": min((row["height"] for row in dimension_rows), default=None),
        "maximum_width": max((row["width"] for row in dimension_rows), default=None),
        "maximum_height": max((row["height"] for row in dimension_rows), default=None),
        "most_common_dimensions": [
            row for row in dimension_rows
            if row["count"] == max((item["count"] for item in dimension_rows), default=0)
        ],
        "source_checks": source_checks,
        "source_files_unchanged": all(item["unchanged"] for item in source_checks),
        "source_file_check_count": len(source_checks),
        "source_hash_match_count": sum(item["unchanged"] for item in source_checks),
        "duplicate_recheck_matches_audit": _duplicate_rows_match(audit, duplicate_groups),
        "invalid_reporting_complete": len(invalid_records) == audit.get("invalid_image_count", 0),
    }


def write_validation_reports(findings: dict, output_dir: Path, run_name: str | None = None) -> Path:
    """Write three detailed investigation reports and machine-readable tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / (run_name or datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%SZ"))
    suffix = 1
    while run_dir.exists():
        run_dir = output_dir / f"{run_name or 'validation'}_{suffix}"
        suffix += 1
    run_dir.mkdir()
    _write_csv(
        run_dir / "cross_class_duplicates.csv",
        [
            {"group_id": group["group_id"], "sha256": group["sha256"], "file_count": group["file_count"],
             "class_name": file["class_name"], "relative_path": file["relative_path"],
             "potential_label_conflict": group["potential_label_conflict"]}
            for group in findings["cross_class_duplicate_groups"] for file in group["files"]
        ],
        ["group_id", "sha256", "file_count", "class_name", "relative_path", "potential_label_conflict"],
    )
    _write_csv(run_dir / "class_distribution.csv", findings["class_distribution"],
               ["class_name", "count", "percentage", "historical_count"])
    _write_csv(run_dir / "dimension_distribution.csv", findings["dimension_distribution"],
               ["width", "height", "count", "percentage"])
    (run_dir / "sector02_validation.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    (run_dir / "duplicate_investigation_report.md").write_text(_duplicate_report(findings), encoding="utf-8")
    (run_dir / "class_distribution_report.md").write_text(_class_report(findings), encoding="utf-8")
    (run_dir / "dimension_distribution_report.md").write_text(_dimension_report(findings), encoding="utf-8")
    (run_dir / "sector02_validation_report.md").write_text(_validation_report(findings), encoding="utf-8")
    return run_dir


def _duplicate_rows_match(audit: dict, groups: dict[str, list[dict]]) -> bool:
    recorded = {
        (row["sha256"], row["relative_path"], row["class_name"])
        for row in audit.get("duplicate_images", [])
    }
    derived = {
        (digest, row["relative_path"], row["class_name"])
        for digest, rows in groups.items() for row in rows
    }
    return recorded == derived


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _duplicate_report(findings: dict) -> str:
    lines = ["# Cross-Class Duplicate Investigation", "",
             f"Source audit: `{findings['audit_source']}`", "",
             f"Cross-class groups: **{findings['cross_class_duplicate_group_count']}**", ""]
    for group in findings["cross_class_duplicate_groups"]:
        lines += [
            f"## Group {group['group_id']}",
            f"- SHA-256: `{group['sha256']}`",
            f"- Files: {group['file_count']}",
            f"- Classes: {', '.join(group['classes'])}",
            "- Potential label conflict: **YES**",
        ]
        lines += [f"  - `{file['relative_path']}` — {file['class_name']}" for file in group["files"]]
        lines.append("")
    lines.append("SHA-256 identifies byte-identical files only; it does not establish visual similarity or patient independence.")
    return "\n".join(lines) + "\n"


def _class_report(findings: dict) -> str:
    lines = ["# Class Distribution Investigation", "",
             f"- Total images: **{sum(row['count'] for row in findings['class_distribution'])}**",
             f"- Majority class: **{findings['majority_class']}**",
             f"- Minority class: **{findings['minority_class']}**",
             f"- Majority-to-minority ratio: **{findings['majority_to_minority_ratio']}:1**",
             f"- Count difference: **{findings['majority_minus_minority']}**", "",
             "| Class | Current count | Current % | Historical reference |",
             "|---|---:|---:|---:|"]
    lines += [
        f"| {row['class_name']} | {row['count']} | {row['percentage']:.4f}% | {row['historical_count']} |"
        for row in findings["class_distribution"]
    ]
    lines += ["", "Historical values are references only and were not substituted for current measurements."]
    return "\n".join(lines) + "\n"


def _dimension_report(findings: dict) -> str:
    common_dimensions = ", ".join(
        f"{row['width']} x {row['height']} ({row['count']})"
        for row in findings["most_common_dimensions"]
    )
    lines = ["# Image Dimension Distribution", "",
             f"- Distinct dimension combinations: **{findings['distinct_dimension_count']}**",
             f"- Minimum width/height: **{findings['minimum_width']} x {findings['minimum_height']}**",
             f"- Maximum width/height: **{findings['maximum_width']} x {findings['maximum_height']}**",
             f"- Most common dimensions: **{common_dimensions}**",
             "", "| Width | Height | Images | Percentage |", "|---:|---:|---:|---:|"]
    lines += [f"| {row['width']} | {row['height']} | {row['count']} | {row['percentage']:.4f}% |"
              for row in findings["dimension_distribution"]]
    return "\n".join(lines) + "\n"


def _validation_report(findings: dict) -> str:
    checks = [
        ("Metadata records cover all discovered images",
         findings["total_metadata_records"] == findings["valid_image_count"] + findings["invalid_image_count"]),
        ("All metadata records have SHA-256 hashes", findings["metadata_hash_count"] == findings["total_metadata_records"]),
        ("Invalid-image reporting is complete", findings["invalid_reporting_complete"]),
        ("Source image hashes remain unchanged", findings["source_files_unchanged"]),
        ("Duplicate groups reproduce from metadata", findings["duplicate_recheck_matches_audit"]),
    ]
    lines = ["# Sector 02 Post-Audit Validation", "",
             f"- Preserved audit status: **{findings['audit_status']}**",
             f"- Source files checked: **{findings['source_file_check_count']}**",
             f"- Source hash matches: **{findings['source_hash_match_count']}**", "",
             "| Check | Result |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks]
    lines += ["", "No images were modified, renamed, moved, deleted, relabeled, resized, augmented, or split.",
              "Patient-level independence was not assessed because grouping identifiers are unavailable."]
    return "\n".join(lines) + "\n"
