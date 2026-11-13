"""Investigate a preserved Sector 02 audit and verify source integrity."""

import argparse
from pathlib import Path

from data.audit.validation import investigate_audit, load_audit, write_validation_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output-dir", default="reports/dataset_audit")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    audit = load_audit(project_root / args.audit_json)
    findings = investigate_audit(audit)
    output = write_validation_reports(findings, project_root / args.output_dir)
    print(f"validation_reports={output}")
    print(f"source_files_unchanged={findings['source_files_unchanged']}")
    print(f"duplicate_recheck_matches_audit={findings['duplicate_recheck_matches_audit']}")
    return 0 if findings["source_files_unchanged"] and findings["duplicate_recheck_matches_audit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
