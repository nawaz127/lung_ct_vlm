"""Run frozen BiomedCLIP inference across all Sector 06 test conditions."""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from data.degradation import CONDITION_IDS, build_condition
from data.preprocessing import preprocess_image
from models import FrozenBiomedCLIP, load_prompts, output_records


def _identity_map(rows):
    counts = {}
    for row in rows:
        key = row.get("sample_id", row["source_relative_path"])
        counts[key] = counts.get(key, 0) + 1
    return {
        id(row): (row.get("sample_id", row["source_relative_path"])
                  if counts[row.get("sample_id", row["source_relative_path"])] == 1
                  else row["source_relative_path"])
        for row in rows
    }


def _images(rows, root, condition, degradation, identities):
    operation = build_condition(condition, degradation)
    for row in rows:
        image = preprocess_image(root / row["source_relative_path"], normalize=False)
        generator = torch.Generator().manual_seed(
            __import__("data.degradation", fromlist=["stable_corruption_seed"])
            .stable_corruption_seed(
                degradation["seed"], identities[id(row)],
                condition, degradation["conditions"][condition].get("std", ""),
            )
        )
        image = operation(image, generator)
        array = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
        yield Image.fromarray(array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/splits/test.csv")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--model-config", default="configs/biomedclip.yaml")
    parser.add_argument("--degradation-config", default="configs/degradation.yaml")
    parser.add_argument("--output-dir", default="reports/biomedclip")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root / args.manifest).open(encoding="utf-8")))
    identities = _identity_map(rows)
    model_config = load_prompts(root / args.model_config)
    degradation = yaml.safe_load((root / args.degradation_config).read_text(encoding="utf-8"))
    model = FrozenBiomedCLIP(model_config)
    outputs = []
    for condition in CONDITION_IDS:
        batch_images, batch_rows = [], []
        for image, row in zip(_images(rows, root / args.data_root, condition, degradation, identities), rows):
            sample_id = identities[id(row)]
            row = dict(row)
            row["sample_id"] = sample_id
            batch_images.append(image)
            batch_rows.append(row)
            if len(batch_images) == args.batch_size:
                outputs.extend(model.predict_batch(batch_images, batch_rows, condition))
                batch_images, batch_rows = [], []
        if batch_images:
            outputs.extend(model.predict_batch(batch_images, batch_rows, condition))
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in output_records(outputs)) + "\n", encoding="utf-8"
    )
    (output / "model_metadata.json").write_text(json.dumps(model.metadata(), indent=2), encoding="utf-8")
    (output / "prompt_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (output / "inference_summary.json").write_text(json.dumps({
        "conditions": list(CONDITION_IDS), "samples_per_condition": len(rows),
        "prediction_records": len(outputs), "test_manifest": args.manifest,
        "model_identifier": model.model_identifier,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"conditions": len(CONDITION_IDS), "samples": len(rows), "records": len(outputs)}))


if __name__ == "__main__":
    raise SystemExit(main())
