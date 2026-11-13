"""Generate measured Sector 08/10 tables, analysis, and publication figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from sklearn.metrics import auc, roc_curve

from analysis.final_analysis import validate_biomedclip


CONDITIONS = (
    "clean", "gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3",
    "gaussian_blur_s1", "gaussian_blur_s2", "gaussian_blur_s3",
    "contrast_reduction_s1", "contrast_reduction_s2", "contrast_reduction_s3",
)
FAMILY_NAMES = {
    "gaussian_noise": "Gaussian noise",
    "gaussian_blur": "Gaussian blur",
    "contrast_reduction": "Contrast reduction",
}
ARCH_NAMES = {
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
}
STRATEGY_NAMES = {"clean_baseline": "Clean baseline", "corruption_aware": "Corruption-aware"}
COLORS = {"clean_baseline": "#377eb8", "corruption_aware": "#e41a1c"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_figure(fig, figures, stem):
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _summary(group, column="macro_f1"):
    values = group[column].dropna().to_numpy(dtype=float)
    if not len(values):
        return pd.Series({"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan})
    return pd.Series({
        "n": len(values), "mean": values.mean(),
        "std": values.std(ddof=1) if len(values) > 1 else 0.0,
        "median": np.median(values), "iqr": np.percentile(values, 75) - np.percentile(values, 25),
    })


def _write_table(frame, tables, name):
    frame.to_csv(tables / f"{name}.csv", index=False)
    return frame


def _metric_frame(metrics):
    frame = pd.DataFrame(metrics)
    frame["architecture_label"] = frame["architecture"].map(ARCH_NAMES)
    frame["strategy_label"] = frame["training_strategy"].map(STRATEGY_NAMES)
    frame["family"] = frame["condition"].map(
        lambda value: "clean" if value == "clean" else value.rsplit("_s", 1)[0]
    )
    frame["severity"] = frame["condition"].map(
        lambda value: 0 if value == "clean" else int(value.rsplit("_s", 1)[1])
    )
    return frame


def _generate_tables(frame, predictions, efficiency, tables):
    clean = frame[frame.condition == "clean"][[
        "experiment_id", "architecture", "training_strategy", "seed",
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
    ]]
    _write_table(clean, tables, "classification_clean_summary")
    summary = frame.groupby(["architecture", "training_strategy"], as_index=False).apply(
        _summary, include_groups=False
    ).reset_index()
    _write_table(summary, tables, "seed_summary_macro_f1")
    robustness = frame[frame.condition != "clean"].merge(
        frame[frame.condition == "clean"][["experiment_id", "macro_f1"]].rename(
            columns={"macro_f1": "clean_macro_f1"}
        ), on="experiment_id", how="left"
    )
    robustness["absolute_degradation"] = robustness["clean_macro_f1"] - robustness["macro_f1"]
    robustness["relative_degradation"] = robustness["absolute_degradation"] / robustness["clean_macro_f1"]
    _write_table(robustness, tables, "robustness_condition_summary")
    family = robustness.groupby(["architecture", "training_strategy", "family", "severity"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        degradation_mean=("absolute_degradation", "mean"),
        degradation_std=("absolute_degradation", "std"),
    )
    _write_table(family, tables, "robustness_family_severity_summary")
    paired = []
    for architecture in sorted(frame.architecture.unique()):
        for condition in CONDITIONS:
            if condition == "clean":
                continue
            left = frame[(frame.architecture == architecture) & (frame.training_strategy == "clean_baseline") & (frame.condition == condition)].set_index("seed").macro_f1
            right = frame[(frame.architecture == architecture) & (frame.training_strategy == "corruption_aware") & (frame.condition == condition)].set_index("seed").macro_f1
            common = sorted(set(left.index) & set(right.index))
            differences = (right.loc[common] - left.loc[common]).to_numpy(dtype=float)
            pvalue = float(ttest_rel(right.loc[common], left.loc[common]).pvalue) if len(common) == 3 else None
            paired.append({
                "architecture": architecture, "condition": condition, "n_pairs": len(common),
                "mean_difference_aware_minus_baseline": differences.mean() if len(differences) else None,
                "std_difference": differences.std(ddof=1) if len(differences) > 1 else None,
                "paired_t_pvalue_descriptive": pvalue,
            })
    _write_table(pd.DataFrame(paired), tables, "paired_strategy_comparison")
    _write_table(pd.DataFrame(efficiency), tables, "efficiency_summary")
    return robustness, family, paired


def _figures(frame, predictions, efficiency, robustness, family, figures):
    sns.set_theme(style="whitegrid", context="paper")
    # 2: dataset distribution
    manifest = pd.read_csv("data/splits/combined.csv")
    counts = manifest.groupby(["split", "canonical_class"]).size().reset_index(name="count")
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="split", y="count", hue="canonical_class", ax=axis)
    axis.set(title="Dataset distribution", xlabel="Split", ylabel="Images")
    _save_figure(fig, figures, "figure_02_dataset_distribution")
    # 3: training curves
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    for (architecture, strategy), axis_pair in zip(
        [(a, s) for a in ARCH_NAMES for s in STRATEGY_NAMES], axes.reshape(-1, 2)
    ):
        histories = []
        for path in sorted(Path("experiments").glob(f"research_{architecture}_{strategy}_*/history.json")):
            data = _read_json(path)
            for record in data:
                histories.append({"epoch": record["epoch"], "train_loss": record["train"]["loss"],
                                  "val_loss": record["validation"]["loss"], "val_macro_f1": record["validation"]["macro_f1"]})
        history = pd.DataFrame(histories)
        for column, axis in zip(("train_loss", "val_loss"), axis_pair):
            sns.lineplot(data=history, x="epoch", y=column, estimator="mean", errorbar="sd", ax=axis)
            axis.set_title(f"{ARCH_NAMES[architecture]} — {STRATEGY_NAMES[strategy]}")
            axis.set_ylabel(column.replace("_", " ").title())
        axis_pair[1].set_xlabel("Epoch")
    _save_figure(fig, figures, "figure_03_training_curves")
    # 4: clean performance
    fig, axis = plt.subplots(figsize=(10, 5))
    clean = frame[frame.condition == "clean"].copy()
    clean["model"] = clean.architecture.map(ARCH_NAMES) + " / " + clean.strategy_label
    sns.stripplot(data=clean, x="model", y="macro_f1", hue="strategy_label",
                  palette={"Clean baseline": COLORS["clean_baseline"],
                           "Corruption-aware": COLORS["corruption_aware"]},
                  ax=axis, size=7)
    axis.set(title="Clean test Macro-F1 by architecture and strategy", xlabel="", ylabel="Macro-F1")
    axis.tick_params(axis="x", rotation=25)
    _save_figure(fig, figures, "figure_04_clean_classification")
    # 5: all conditions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, architecture in zip(axes, ARCH_NAMES):
        subset = frame[frame.architecture == architecture]
        sns.lineplot(data=subset, x="condition", y="macro_f1", hue="strategy_label", style="seed",
                     palette={"Clean baseline": COLORS["clean_baseline"],
                              "Corruption-aware": COLORS["corruption_aware"]},
                     marker="o", ax=axis)
        axis.set_title(ARCH_NAMES[architecture]); axis.tick_params(axis="x", rotation=65)
        axis.set_xlabel(""); axis.set_ylabel("Macro-F1")
    _save_figure(fig, figures, "figure_05_robustness_performance")
    # 6 and 7
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, family_name in zip(axes, FAMILY_NAMES):
        subset = frame[frame.family == family_name]
        sns.lineplot(data=subset, x="severity", y="macro_f1", hue="strategy_label",
                     style="architecture_label",
                     palette={"Clean baseline": COLORS["clean_baseline"],
                              "Corruption-aware": COLORS["corruption_aware"]},
                     marker="o", ax=axis)
        axis.set_title(FAMILY_NAMES[family_name]); axis.set_xlabel("Severity"); axis.set_ylabel("Macro-F1")
    _save_figure(fig, figures, "figure_06_corruption_family")
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=robustness, x="condition", y="absolute_degradation", hue="training_strategy",
                palette={"clean_baseline": COLORS["clean_baseline"],
                         "corruption_aware": COLORS["corruption_aware"]},
                errorbar="sd", ax=axis)
    axis.set(title="Clean-relative Macro-F1 degradation", xlabel="Condition", ylabel="Degradation (Macro-F1 points)")
    axis.tick_params(axis="x", rotation=65)
    _save_figure(fig, figures, "figure_07_performance_degradation")
    # 8: clean confusion matrices, one per architecture/strategy
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (architecture, strategy) in zip(axes.flat, [(a, s) for a in ARCH_NAMES for s in STRATEGY_NAMES]):
        row = frame[(frame.architecture == architecture) & (frame.training_strategy == strategy) & (frame.condition == "clean")].iloc[0]
        matrix = np.asarray(row.confusion_matrix)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                    xticklabels=["Normal", "Benign", "Malignant"], yticklabels=["Normal", "Benign", "Malignant"])
        axis.set_title(f"{ARCH_NAMES[architecture]} / {STRATEGY_NAMES[strategy]}")
        axis.set_xlabel("Predicted"); axis.set_ylabel("True")
    _save_figure(fig, figures, "figure_08_confusion_matrices")
    # 9: per-class F1
    rows = []
    for _, row in frame[frame.condition == "clean"].iterrows():
        for class_name, value in zip(["Normal", "Benign", "Malignant"], row.per_class_f1):
            rows.append({**row.to_dict(), "class_name": class_name, "f1": value})
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.barplot(data=pd.DataFrame(rows), x="architecture_label", y="f1", hue="class_name", errorbar="sd", ax=axis)
    axis.set(title="Clean-condition per-class F1", xlabel="", ylabel="F1")
    _save_figure(fig, figures, "figure_09_per_class_performance")
    # 10: ROC curves from predictions
    pred = pd.DataFrame(predictions)
    pred["model"] = pred.architecture.map(ARCH_NAMES) + " / " + pred.training_strategy.map(STRATEGY_NAMES)
    fig, axis = plt.subplots(figsize=(8, 6))
    for model, subset in pred[pred.condition_id == "clean"].groupby("model"):
        y = subset.true_label.to_numpy()
        scores = np.asarray(subset.predicted_class_probabilities.tolist())
        fpr, tpr, _ = roc_curve((y == 2).astype(int), scores[:, 2])
        axis.plot(fpr, tpr, label=model)
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(title="Clean-condition malignant ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend(fontsize=7)
    _save_figure(fig, figures, "figure_10_roc_curves")
    # 11 efficiency
    eff = pd.DataFrame(efficiency)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column, label in zip(axes, ["total_parameters", "latency_ms_mean", "throughput_images_per_second"],
                                   ["Parameters", "Latency (ms)", "Throughput (images/s)"]):
        sns.barplot(data=eff, x="architecture", y=column, ax=axis, color="#4daf4a")
        axis.set_xlabel(""); axis.set_ylabel(label); axis.tick_params(axis="x", rotation=25)
    _save_figure(fig, figures, "figure_11_efficiency")
    # 12 BiomedCLIP
    fig, axis = plt.subplots(figsize=(10, 5))
    bpath = Path("reports/evaluation/biomedclip_metrics.json")
    if bpath.is_file():
        b = pd.DataFrame([{"condition": k, "macro_f1": v["macro_f1"], "model": "BiomedCLIP (frozen)"}
                          for k, v in _read_json(bpath).items()])
        c = frame.groupby("condition", as_index=False).macro_f1.mean()
        c["model"] = "CNN mean"
        sns.lineplot(data=pd.concat([b, c]), x="condition", y="macro_f1", hue="model", marker="o", ax=axis)
    axis.set(title="Supplementary frozen BiomedCLIP comparison", xlabel="Condition", ylabel="Macro-F1")
    axis.tick_params(axis="x", rotation=65)
    _save_figure(fig, figures, "figure_12_biomedclip_comparison")
    # 16 paired
    paired_rows = []
    for architecture in ARCH_NAMES:
        for seed in (1, 2, 3):
            for condition in CONDITIONS:
                left = frame[(frame.architecture == architecture) & (frame.training_strategy == "clean_baseline") & (frame.seed == seed) & (frame.condition == condition)]
                right = frame[(frame.architecture == architecture) & (frame.training_strategy == "corruption_aware") & (frame.seed == seed) & (frame.condition == condition)]
                if len(left) and len(right):
                    paired_rows.append({"architecture": ARCH_NAMES[architecture], "seed": seed, "condition": condition,
                                        "difference": right.iloc[0].macro_f1 - left.iloc[0].macro_f1})
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.pointplot(data=pd.DataFrame(paired_rows), x="condition", y="difference", hue="architecture",
                  dodge=0.3, errorbar="sd", ax=axis)
    axis.axhline(0, color="black", linewidth=1); axis.tick_params(axis="x", rotation=65)
    axis.set(title="Paired corruption-aware minus clean-baseline Macro-F1", xlabel="Condition", ylabel="Paired difference")
    _save_figure(fig, figures, "figure_16_paired_strategy")
    # 17 heatmap
    heat = frame[frame.family != "clean"].groupby(
        ["architecture_label", "family", "severity"], as_index=False
    ).macro_f1.mean()
    heat["condition"] = heat.family.map(FAMILY_NAMES) + " S" + heat.severity.astype(str)
    pivot = heat.pivot(index="architecture_label", columns="condition", values="macro_f1")
    fig, axis = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=axis)
    axis.set(title="Mean Macro-F1 across corruption family and severity", xlabel="", ylabel="")
    _save_figure(fig, figures, "figure_17_robustness_heatmap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", default="reports/final_evaluation/cnn_full_20261112_v2")
    parser.add_argument("--output-dir", default="reports/final_evaluation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evaluation = root / args.evaluation_dir
    output = root / args.output_dir
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    frame = _metric_frame(_read_json(evaluation / "metrics.json"))
    predictions = [json.loads(line) for line in (evaluation / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    efficiency = _read_json(evaluation / "efficiency.json")
    robustness, family, paired = _generate_tables(frame, predictions, efficiency, tables)
    _figures(frame, predictions, efficiency, robustness, family, figures)
    audit = _read_json(evaluation / "checkpoint_audit.json")
    biomed_path = root / "reports/biomedclip/predictions.jsonl"
    biomed_records = [json.loads(line) for line in biomed_path.read_text(encoding="utf-8").splitlines()]
    biomed_audit = validate_biomedclip(
        biomed_records, CONDITIONS, expected_prompt_version="sector07_v1"
    )
    (output / "biomedclip_validation.json").write_text(json.dumps(biomed_audit, indent=2), encoding="utf-8")
    summary = {
        "checkpoint_audit": audit,
        "prediction_records": len(predictions),
        "metric_rows": len(frame),
        "biomedclip_validation": biomed_audit,
        "figures": sorted(path.name for path in figures.glob("*")),
        "tables": sorted(path.name for path in tables.glob("*.csv")),
        "limitations": [
            "Image-level rather than patient-independent splitting.",
            "No expert lesion masks; XAI metrics represent stability, not clinical validity.",
            "Three seeds limit statistical power.",
            "BiomedCLIP is a frozen supplementary zero-shot baseline.",
        ],
    }
    (output / "final_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "statistical_analysis.md").write_text(
        "# Statistical analysis\n\n"
        "Seed summaries use mean, sample standard deviation, median, and IQR. "
        "Strategy comparisons are paired by architecture, seed, and condition. "
        "The paired t-test values in `paired_strategy_comparison.csv` are descriptive "
        "only; no significance claim is made because n=3 seeds provides limited power "
        "and no multiple-comparison correction is used.\n", encoding="utf-8"
    )
    print(json.dumps({"predictions": len(predictions), "metrics": len(frame),
                      "figures": len(list(figures.glob("*.png"))),
                      "tables": len(list(tables.glob("*.csv")))}))


if __name__ == "__main__":
    main()
