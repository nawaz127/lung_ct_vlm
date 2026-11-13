import pytest

from evaluation import evaluate_predictions, robustness_summary, validate_prediction_records


def test_metrics_and_auc_unavailable_for_missing_class():
    result = evaluate_predictions([0, 1], [0, 1], [[1, 0, 0], [0, 1, 0]])
    assert result["accuracy"] == 1
    assert result["roc_auc_ovr"] is None


def test_robustness_sign_and_grouping():
    metrics = {
        "clean": {"macro_f1": 0.8, "accuracy": 0.9, "per_class_recall": [1, 0.8, 0.8]},
        "gaussian_noise_s1": {"macro_f1": 0.6, "accuracy": 0.7, "per_class_recall": [0.8, 0.6, 0.5]},
    }
    result = robustness_summary(metrics)
    assert result["conditions"][0]["macro_f1_degradation"] == pytest.approx(0.2)
    assert result["conditions"][0]["per_class_recall_change"] == pytest.approx([-0.2, -0.2, -0.3])


def test_integrity_detects_duplicates():
    rows = [
        {"sample_id": "a", "condition": "clean", "ground_truth": 0},
        {"sample_id": "a", "condition": "clean", "ground_truth": 0},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_records(rows, ("clean",))
