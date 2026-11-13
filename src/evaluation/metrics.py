"""Classification metrics for fixed three-class evaluation."""

import numpy as np
from sklearn.metrics import roc_auc_score

CLASS_NAMES = ("Normal", "Benign", "Malignant")


def _json_metric(value):
    return None if not np.isfinite(value) else float(value)


def _mean_defined(values):
    return _json_metric(np.nanmean(values)) if np.isfinite(values).any() else None


def evaluate_predictions(targets, predictions, scores=None):
    targets = np.asarray(targets, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("targets and predictions must be non-empty and equal length")
    confusion = np.zeros((3, 3), dtype=int)
    for target, prediction in zip(targets, predictions):
        if target not in range(3) or prediction not in range(3):
            raise ValueError("class indices must be in [0, 2]")
        confusion[target, prediction] += 1
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.full(3, np.nan), where=predicted != 0)
    recall = np.divide(tp, support, out=np.full(3, np.nan), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                    out=np.full(3, np.nan), where=(precision + recall) != 0)
    result = {
        "accuracy": float(tp.sum() / len(targets)),
        "balanced_accuracy": _mean_defined(recall),
        "macro_precision": _mean_defined(precision),
        "macro_recall": _mean_defined(recall),
        "macro_f1": _mean_defined(f1),
        "per_class_precision": [_json_metric(value) for value in precision],
        "per_class_recall": [_json_metric(value) for value in recall],
        "per_class_f1": [_json_metric(value) for value in f1],
        "confusion_matrix": confusion.tolist(),
        "class_names": list(CLASS_NAMES),
    }
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(targets), 3) or not np.isfinite(scores).all():
            raise ValueError("scores must be finite with shape [N, 3]")
        if len(np.unique(targets)) == 3:
            shifted = scores - scores.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            result["roc_auc_ovr"] = float(roc_auc_score(targets, probabilities, multi_class="ovr"))
        else:
            result["roc_auc_ovr"] = None
            result["roc_auc_unavailable_reason"] = "fewer than three target classes"
    return result
