"""Reproducible, duplicate-aware stratified image-level splitting."""

from collections import defaultdict
import random

from data.preprocessing.pipeline import PreprocessingError, validate_split_ratios


def split_records(records, train_ratio, validation_ratio, test_ratio, seed,
                  duplicate_groups=None, cross_class_groups=None,
                  cross_class_policy="error"):
    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    if cross_class_policy not in {"error", "exclude_conflicted_groups", "manual_review"}:
        raise PreprocessingError("Invalid cross-class duplicate policy")
    cross_class_groups = cross_class_groups or set()
    if cross_class_groups and cross_class_policy in {"error", "manual_review"}:
        raise PreprocessingError("Unresolved cross-class duplicate groups require review")
    excluded = set()
    if cross_class_policy == "exclude_conflicted_groups":
        for group_id in cross_class_groups:
            excluded.update((duplicate_groups or {}).get(group_id, []))
    usable = [record for record in records if record.relative_path not in excluded]
    grouped = defaultdict(list)
    for record in usable:
        grouped[record.duplicate_group_id or f"single:{record.relative_path}"].append(record)
    by_class = defaultdict(list)
    for group in grouped.values():
        classes = {record.class_name for record in group}
        if len(classes) != 1:
            raise PreprocessingError("Cross-class duplicate group entered split dataset")
        by_class[next(iter(classes))].append(sorted(group, key=lambda item: item.relative_path.casefold()))
    assignments = {}
    rng = random.Random(seed)
    for class_name in sorted(by_class):
        groups = by_class[class_name]
        rng.shuffle(groups)
        total = sum(len(group) for group in groups)
        targets = (round(total * train_ratio), round(total * validation_ratio))
        counts = [0, 0]
        for group in groups:
            if counts[0] < targets[0]:
                split = "train"; counts[0] += len(group)
            elif counts[1] < targets[1]:
                split = "validation"; counts[1] += len(group)
            else:
                split = "test"
            for record in group:
                assignments[record.relative_path] = split
    return [
        {
            "sample_id": record.sha256[:16],
            "source_relative_path": record.relative_path,
            "canonical_class": record.class_name,
            "class_index": {"Normal": 0, "Benign": 1, "Malignant": 2}[record.class_name],
            "split": assignments[record.relative_path],
            "sha256": record.sha256,
            "duplicate_group_id": record.duplicate_group_id,
        }
        for record in sorted(usable, key=lambda item: item.relative_path.casefold())
    ]