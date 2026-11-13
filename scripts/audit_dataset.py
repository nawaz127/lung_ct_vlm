"""Command-line entry point for the read-only Sector 02 dataset audit."""

import argparse
import os
from pathlib import Path
import re
import sys

import yaml

from data.audit import audit_dataset, write_reports


def _resolve(value: str, project_root: Path) -> Path:
    match = re.fullmatch(r"\$\{([^:}]+):([^}]+)\}", value)
    if match:
        value = os.environ.get(match.group(1), match.group(2))
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", default="reports/dataset_audit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root or Path(__file__).resolve().parents[1]).resolve()
    try:
        config = yaml.safe_load((project_root / args.config).read_text(encoding="utf-8")) or {}
        dataset = config["dataset"]
        root = _resolve(dataset["raw_dir"], project_root)
        expected = list(dataset["expected_classes"])
        extensions = {str(extension).casefold() for extension in dataset.get("image_extensions", [".png"])}
        if args.dry_run:
            print(f"dataset_root={root}")
            print(f"exists={root.is_dir()}")
            print(f"expected_classes={','.join(expected)}")
            return 0 if root.is_dir() else 2
        result = audit_dataset(
            root,
            expected,
            extensions,
            bool(dataset.get("recursive", True)),
            dict(dataset.get("historical_counts", {})),
            dict(dataset.get("class_aliases", {})),
        )
        output = write_reports(result, project_root / args.output_dir)
        print(f"status={result['status']}")
        print(f"reports={output}")
        return 1 if result["status"] == "FAIL" else 0
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"Audit configuration or execution error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
