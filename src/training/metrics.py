"""Classification metrics without a dependency on the evaluation sector."""

import numpy as np


def classification_metrics(targets, predictions, num_classes=3):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            confusion[target, prediction] += 1
    true_positive = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    return {
        "accuracy": float(true_positive.sum() / max(1, confusion.sum())),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class_recall": recall.tolist(),
    }
