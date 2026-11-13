"""Inventory, duplicate analysis, and report generation for Sector 02."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

from .discovery import discover_dataset
from .integrity import inspect_image


def _percentage(count: int, total: int) -> float:
    return round(count * 100 / total, 4) if total else 0.0


def audit_dataset(
    root: Path,
    expected_classes: list[str],
    supported_extensions: set[str],
    recursive: bool = True,
    historical_counts: dict[str, int] | None = None,
    class_aliases: dict[str, str] | None = None,
) -> dict:
    discovery = discover_dataset(root, expected_classes, supported_extensions, recursive, class_aliases)
    records = [inspect_image(item.path, root, item.class_name) for item in discovery.candidates]
    valid = [record for record in records if record["validation_status"] == "valid"]
    invalid = [record for record in records if record["validation_status"] != "valid"]
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for record in valid:
        by_hash[record["sha256"]].append(record)
    duplicate_groups = [group for group in by_hash.values() if len(group) > 1]
    duplicate_rows = []
    cross_class_groups = 0
    for index, group in enumerate(sorted(duplicate_groups, key=lambda group: group[0]["sha256"]), start=1):
        classes = sorted({row["class_name"] for row in group})
        cross_class = len(classes) > 1
        cross_class_groups += int(cross_class)
        for row in group:
            duplicate_rows.append({
                "duplicate_group": index,
                "sha256": row["sha256"],
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "cross_class_duplicate": cross_class,
            })
    counts = Counter(record["class_name"] for record in records)
    class_distribution = [
        {"class_name": name, "count": counts[name], "percentage": _percentage(counts[name], len(records))}
        for name in expected_classes
    ]
    dimensions = Counter((record["width"], record["height"]) for record in valid)
    modes = Counter(record["mode"] for record in valid)
    extensions = Counter(record["extension"] for record in records)
    errors = []
    warnings = []
    if not discovery.dataset_root_exists:
        errors.append("dataset_root_missing")
    if discovery.missing_classes:
        errors.append("missing_expected_classes:" + ",".join(discovery.missing_classes))
    if discovery.unexpected_classes:
        warnings.append("unexpected_class_directories:" + ",".join(discovery.unexpected_classes))
    if discovery.empty_classes:
        warnings.append("empty_class_directories:" + ",".join(discovery.empty_classes))
    if discovery.unsupported_files:
        warnings.append(f"unsupported_files:{len(discovery.unsupported_files)}")
    if invalid:
        warnings.append(f"unreadable_images:{len(invalid)}")
    if cross_class_groups:
        warnings.append(f"cross_class_duplicate_groups:{cross_class_groups}")
    if len(dimensions) > 1:
        warnings.append("dimension_variation")
    if len(modes) > 1:
        warnings.append("mode_variation")
    if len(counts) > 1 and max(counts.values()) > 2 * max(1, min(counts.values())):
        warnings.append("class_imbalance")
    status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
    return {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(Path(root).resolve()),
        "total_discovered_files": len(records),
        "valid_image_count": len(valid),
        "invalid_image_count": len(invalid),
        "class_distribution": class_distribution,
        "historical_class_counts": historical_counts or {},
        "extension_distribution": dict(sorted(extensions.items())),
        "mode_distribution": {str(key): value for key, value in sorted(modes.items())},
        "dimension_distribution": {f"{width}x{height}": value for (width, height), value in sorted(dimensions.items())},
        "missing_classes": discovery.missing_classes,
        "unexpected_classes": discovery.unexpected_classes,
        "empty_classes": discovery.empty_classes,
        "unsupported_files": discovery.unsupported_files,
        "image_metadata": records,
        "invalid_images": invalid,
        "duplicate_images": duplicate_rows,
        "duplicate_group_count": len(duplicate_groups),
        "cross_class_duplicate_group_count": cross_class_groups,
        "warnings": warnings,
        "errors": errors,
        "status": status,
        "completion_status": "completed",
        "sha256_scope": "byte-identical files only; not visual similarity, adjacent slices, or patient independence",
    }


def write_reports(result: dict, output_dir: Path, run_name: str | None = None) -> Path:
    """Write a unique, machine-readable report bundle without overwriting prior runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / (run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
    suffix = 1
    while run_dir.exists():
        run_dir = output_dir / f"{run_name or 'run'}_{suffix}"
        suffix += 1
    run_dir.mkdir()
    metadata_fields = list(result["image_metadata"][0]) if result["image_metadata"] else [
        "relative_path", "absolute_path", "class_name", "extension", "format", "width", "height",
        "mode", "channels", "file_size_bytes", "file_size_kb", "sha256", "validation_status", "validation_error"
    ]
    invalid_fields = metadata_fields
    duplicate_fields = ["duplicate_group", "sha256", "relative_path", "class_name", "cross_class_duplicate"]
    inventory = [
        {"metric": "total_discovered_files", "value": result["total_discovered_files"]},
        {"metric": "valid_image_count", "value": result["valid_image_count"]},
        {"metric": "invalid_image_count", "value": result["invalid_image_count"]},
        {"metric": "duplicate_group_count", "value": result["duplicate_group_count"]},
        {"metric": "cross_class_duplicate_group_count", "value": result["cross_class_duplicate_group_count"]},
    ]
    _write_csv(run_dir / "dataset_inventory.csv", inventory, ["metric", "value"])
    _write_csv(run_dir / "image_metadata.csv", result["image_metadata"], metadata_fields)
    _write_csv(run_dir / "invalid_images.csv", result["invalid_images"], invalid_fields)
    _write_csv(run_dir / "duplicate_images.csv", result["duplicate_images"], duplicate_fields)
    _write_csv(run_dir / "class_distribution.csv", result["class_distribution"], ["class_name", "count", "percentage"])
    (run_dir / "dataset_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = [
        "# Dataset Audit Report", "", f"- Status: **{result['status']}**",
        f"- Dataset root: `{result['dataset_root']}`",
        f"- Discovered files: {result['total_discovered_files']}",
        f"- Valid images: {result['valid_image_count']}",
        f"- Invalid images: {result['invalid_image_count']}", "",
        "## Findings", "",
        *[f"- {warning}" for warning in result["warnings"]],
        *[f"- ERROR: {error}" for error in result["errors"]],
        "", "SHA-256 detects byte-identical files only; it does not establish visual similarity or patient-level independence.",
    ]
    (run_dir / "dataset_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return run_dir


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)