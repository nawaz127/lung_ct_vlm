"""Sector 10 experiment ingestion, integrity validation, and summaries."""

from collections import defaultdict
from dataclasses import dataclass
import csv
import math
from pathlib import Path

EXPECTED_ARCHITECTURES = ("mobilenet_v3_large", "efficientnet_b0", "resnet18")
EXPECTED_STRATEGIES = ("clean_baseline", "corruption_aware")
EXPECTED_SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class ExperimentKey:
    architecture: str
    strategy: str
    seed: int


def expected_matrix():
    return [ExperimentKey(a, s, seed)
            for a in EXPECTED_ARCHITECTURES
            for s in EXPECTED_STRATEGIES
            for seed in EXPECTED_SEEDS]


def ingest_results(records):
    seen = set()
    issues = []
    valid = []
    for record in records:
        key = (record.get("architecture"), record.get("training_strategy"), record.get("seed"))
        if key in seen:
            issues.append({"type": "duplicate_experiment", "key": key})
        seen.add(key)
        if key[0] not in EXPECTED_ARCHITECTURES:
            issues.append({"type": "invalid_architecture", "key": key})
        if key[1] not in EXPECTED_STRATEGIES:
            issues.append({"type": "invalid_strategy", "key": key})
        try:
            seed = int(key[2])
            metric = float(record["macro_f1"])
            if not math.isfinite(metric):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append({"type": "invalid_metric_or_seed", "key": key})
            continue
        valid.append({**record, "seed": seed, "macro_f1": metric})
    expected = {(item.architecture, item.strategy, item.seed) for item in expected_matrix()}
    missing = sorted(expected - seen)
    issues.extend({"type": "missing_experiment", "key": key} for key in missing)
    return {"valid_records": valid, "issues": issues, "missing": missing}


def summarize_seed_results(records, metric="macro_f1"):
    groups = defaultdict(list)
    for record in records:
        value = float(record[metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric}")
        groups[(record["architecture"], record["training_strategy"])].append(value)
    output = []
    for (architecture, strategy), values in sorted(groups.items()):
        ordered = sorted(values)
        q1 = ordered[(len(ordered) - 1) // 4]
        q3 = ordered[(3 * (len(ordered) - 1)) // 4]
        mean = sum(values) / len(values)
        output.append({
            "architecture": architecture, "training_strategy": strategy,
            "n_seeds": len(values), "mean": mean,
            "std": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
            "median": ordered[len(ordered) // 2], "iqr": q3 - q1,
        })
    return output


def paired_strategy_differences(records, metric="macro_f1"):
    by_key = {(r["architecture"], r["seed"], r["training_strategy"]): r[metric] for r in records}
    output = []
    for architecture in EXPECTED_ARCHITECTURES:
        for seed in EXPECTED_SEEDS:
            baseline = by_key.get((architecture, seed, "clean_baseline"))
            aware = by_key.get((architecture, seed, "corruption_aware"))
            if baseline is not None and aware is not None:
                output.append({
                    "architecture": architecture, "seed": seed,
                    "difference": float(aware) - float(baseline),
                })
    return output


def validate_biomedclip(records, expected_conditions, expected_samples=290,
                        expected_prompt_version="sector07_v1"):
    keys = [(row["sample_id"], row["condition"]) for row in records]
    return {
        "records": len(records),
        "expected_records": expected_samples * len(expected_conditions),
        "unique_records": len(set(keys)),
        "duplicate_records": len(keys) - len(set(keys)),
        "conditions": sorted({row["condition"] for row in records}),
        "samples": len({row["sample_id"] for row in records}),
        "prompt_versions": sorted({row.get("prompt_version") for row in records}),
        "prompt_version_consistent": {row.get("prompt_version") for row in records} == {expected_prompt_version},
        "complete": (
            len(records) == expected_samples * len(expected_conditions)
            and len(keys) == len(set(keys))
            and {row.get("prompt_version") for row in records} == {expected_prompt_version}
        ),
    }


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
