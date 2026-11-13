"""Create a ten-condition qualitative Sector 06 comparison grid."""

import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import yaml
import torch

from data.degradation import CONDITION_IDS, DegradedManifestDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/splits/test.csv")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--config", default="configs/degradation.yaml")
    parser.add_argument("--output", default="reports/degradation/sector06_comparison_grid.png")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    rows = list(csv.DictReader((root / args.manifest).open(encoding="utf-8")))
    if not rows:
        raise ValueError("test manifest is empty")
    sample_id = rows[0].get("sample_id", rows[0]["source_relative_path"])
    images = []
    for condition in CONDITION_IDS:
        dataset = DegradedManifestDataset(root / args.manifest, root / args.data_root,
                                          condition, config)
        index = next(i for i, row in enumerate(dataset.rows)
                     if row.get("sample_id", row["source_relative_path"]) == sample_id)
        image, _, _ = dataset[index]
        mean = torch.tensor(config["normalization"]["mean"]).view(3, 1, 1)
        std = torch.tensor(config["normalization"]["std"]).view(3, 1, 1)
        images.append((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
    figure, axes = plt.subplots(2, 5, figsize=(18, 8))
    for axis, condition, image in zip(axes.flat, CONDITION_IDS, images):
        axis.imshow(image)
        axis.set_title(condition)
        axis.axis("off")
    figure.tight_layout()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"sample_id={sample_id}")
    print(f"output={output}")


if __name__ == "__main__":
    raise SystemExit(main())
