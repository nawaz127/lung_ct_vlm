"""Reusable publication-oriented plots for measured tabular results."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_condition_metric(rows, output_path, metric="macro_f1", title="Condition comparison"):
    if not rows:
        raise ValueError("cannot plot empty results")
    conditions = [row["condition"] for row in rows]
    values = [float(row[metric]) for row in rows]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(conditions, values, marker="o")
    axis.set_title(title)
    axis.set_xlabel("Condition")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output
