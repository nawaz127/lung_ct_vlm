"""Clean-relative degradation summaries."""

from collections import defaultdict


def robustness_summary(metrics_by_condition):
    if "clean" not in metrics_by_condition:
        raise ValueError("clean metrics are required")
    clean = metrics_by_condition["clean"]
    rows = []
    for condition, metrics in metrics_by_condition.items():
        if condition == "clean":
            continue
        rows.append({
            "condition": condition,
            "clean_macro_f1": clean["macro_f1"],
            "corrupted_macro_f1": metrics["macro_f1"],
            "macro_f1_degradation": clean["macro_f1"] - metrics["macro_f1"],
            "relative_macro_f1_change": (
                (metrics["macro_f1"] - clean["macro_f1"]) / clean["macro_f1"]
                if clean["macro_f1"] else None
            ),
            "accuracy_change": metrics["accuracy"] - clean["accuracy"],
            "per_class_recall_change": [
                b - a for a, b in zip(clean["per_class_recall"], metrics["per_class_recall"])
            ],
        })
    by_type = defaultdict(list)
    by_severity = defaultdict(list)
    for row in rows:
        parts = row["condition"].rsplit("_s", 1)
        by_type[parts[0]].append(row["corrupted_macro_f1"])
        by_severity[int(parts[1])].append(row["corrupted_macro_f1"])
    return {
        "conditions": rows,
        "mean_corrupted_macro_f1": sum(r["corrupted_macro_f1"] for r in rows) / len(rows),
        "worst_case_corrupted_macro_f1": min(r["corrupted_macro_f1"] for r in rows),
        "mean_macro_f1_by_corruption": {k: sum(v) / len(v) for k, v in by_type.items()},
        "mean_macro_f1_by_severity": {str(k): sum(v) / len(v) for k, v in by_severity.items()},
    }
