"""Prediction identity and label integrity checks."""


def canonical_identity(row):
    return row.get("source_relative_path") or row.get("sample_id")


def validate_prediction_records(records, conditions, models=None):
    expected = set(conditions)
    keys, labels, duplicates = set(), {}, []
    for row in records:
        condition = row.get("condition")
        model = row.get("model", "biomedclip")
        identity = canonical_identity(row)
        key = (model, identity, condition)
        if key in keys:
            duplicates.append(key)
        keys.add(key)
        label = int(row["ground_truth"])
        if identity in labels and labels[identity] != label:
            raise ValueError(f"label mismatch for {identity}")
        labels[identity] = label
        if condition not in expected:
            raise ValueError(f"invalid condition: {condition}")
    model_names = models or sorted({row.get("model", "biomedclip") for row in records})
    if duplicates:
        raise ValueError(f"duplicate prediction records: {duplicates[:3]}")
    expected_count = len(labels) * len(expected)
    return {
        "expected_unique_identities": len(labels),
        "actual_unique_identities": len(labels),
        "records": len(records),
        "records_per_condition": {condition: sum(row.get("condition") == condition for row in records)
                                  for condition in conditions},
        "duplicate_records": 0,
        "missing_records_by_model": {
            model: expected_count - sum(1 for row in records if row.get("model", "biomedclip") == model)
            for model in model_names
        },
        "labels_consistent": True,
    }
