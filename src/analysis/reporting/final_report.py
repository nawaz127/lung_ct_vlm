"""Sector 10 machine-readable and Markdown report generation."""

import json
from pathlib import Path

from analysis.final_analysis import (
    EXPECTED_ARCHITECTURES, EXPECTED_SEEDS, EXPECTED_STRATEGIES,
    expected_matrix, summarize_seed_results, write_csv,
)


def generate_report(output_dir, matrix_audit, biomed_audit, records=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize_seed_results(records or []) if records else []
    write_csv(summaries, output / "seed_summary.csv")
    (output / "matrix_validation.json").write_text(json.dumps(matrix_audit, indent=2), encoding="utf-8")
    (output / "biomedclip_validation.json").write_text(json.dumps(biomed_audit, indent=2), encoding="utf-8")
    report = [
        "# Sector 10 Final Research Analysis",
        "",
        "**Result: TESTED WITH WARNINGS**",
        "",
        "## Experiment matrix",
        "",
        f"- Expected CNN runs: **{len(expected_matrix())}**",
        f"- Architectures: {', '.join(EXPECTED_ARCHITECTURES)}",
        f"- Strategies: {', '.join(EXPECTED_STRATEGIES)}",
        f"- Seeds: {', '.join(map(str, EXPECTED_SEEDS))}",
        f"- Missing runs: **{len(matrix_audit.get('missing', []))}**",
        f"- Integrity issues: **{len(matrix_audit.get('issues', []))}**",
        "",
        "No CNN research metrics are reported because completed research runs "
        "and checkpoints are unavailable. The Sector 05 smoke checkpoint is "
        "not treated as a final research result.",
        "",
        "## BiomedCLIP supplementary baseline",
        "",
        f"- Records: {biomed_audit.get('records', 0)}",
        f"- Unique records: {biomed_audit.get('unique_records', 0)}",
        f"- Samples: {biomed_audit.get('samples', 0)}",
        f"- Complete: {biomed_audit.get('complete', False)}",
        "",
        "## Limitations",
        "",
        "- Splits are image-level; patient/examination identifiers are unavailable.",
        "- No expert lesion masks are available.",
        "- Explanation stability is not clinical validity.",
        "- BiomedCLIP is a frozen supplementary zero-shot baseline.",
        "- Three seeds are planned per configuration; missing runs remain blockers.",
    ]
    (output / "sector10_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output
