import json

import pytest

from analysis.final_analysis import ingest_results, paired_strategy_differences, summarize_seed_results, validate_biomedclip
from analysis.reporting.final_report import generate_report
from analysis.reporting.plots import plot_condition_metric


def _records():
    return [
        {"architecture": "resnet18", "training_strategy": "clean_baseline", "seed": seed, "macro_f1": value}
        for seed, value in ((1, 0.5), (2, 0.6), (3, 0.7))
    ] + [
        {"architecture": "resnet18", "training_strategy": "corruption_aware", "seed": seed, "macro_f1": value}
        for seed, value in ((1, 0.6), (2, 0.7), (3, 0.8))
    ]


def test_matrix_reports_missing_and_duplicate_runs():
    audit = ingest_results(_records() + [_records()[0]])
    assert audit["missing"]
    assert any(issue["type"] == "duplicate_experiment" for issue in audit["issues"])


def test_seed_summary_and_paired_differences():
    summary = summarize_seed_results(_records())
    assert summary[0]["n_seeds"] == 3
    differences = paired_strategy_differences(_records())
    assert [row["difference"] for row in differences] == pytest.approx([0.1, 0.1, 0.1])


def test_biomedclip_validation_and_report(tmp_path):
    rows = [{"sample_id": "a", "condition": "clean", "prompt_version": "sector07_v1"}]
    validation = validate_biomedclip(rows, ("clean",), expected_samples=1)
    assert validation["complete"]
    output = generate_report(tmp_path, ingest_results([]), validation)
    assert (output / "sector10_summary.md").is_file()
    assert json.loads((output / "biomedclip_validation.json").read_text())["complete"]


def test_plot_generation_rejects_empty_and_writes_file(tmp_path):
    with pytest.raises(ValueError):
        plot_condition_metric([], tmp_path / "empty.png")
    path = plot_condition_metric(
        [{"condition": "clean", "macro_f1": 0.5}],
        tmp_path / "metric.png",
    )
    assert path.is_file()
