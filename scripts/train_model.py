"""Run one configured Sector 05 CNN training experiment."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import yaml
import torch

from models import create_model
from training import TrainingEngine, build_loader, build_loss
from utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet18",
                        choices=["mobilenet_v3_large", "efficientnet_b0", "resnet18"])
    parser.add_argument("--initialization", choices=["scratch", "imagenet"], default="scratch")
    parser.add_argument("--strategy", choices=["clean_baseline", "corruption_aware"], default="clean_baseline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--training-corruption-config", default="configs/training_corruption.yaml")
    parser.add_argument("--head-type", default="gap_dropout_linear",
                        choices=["gap_linear", "gap_dropout_linear"])
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--max-epochs", type=int,
                        help="Temporary epoch limit; does not modify the YAML file")
    parser.add_argument("--run-id",
                        help="Unique output directory name under the configured experiment directory")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))["training"]
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    corruption_config = None
    if args.strategy == "corruption_aware":
        corruption_config = yaml.safe_load(
            (root / args.training_corruption_config).read_text(encoding="utf-8")
        )
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max-epochs must be at least 1")
        config["max_epochs"] = args.max_epochs
    train_manifest = root / args.manifest_dir / "train.csv"
    validation_manifest = root / args.manifest_dir / "validation.csv"
    data_root = root / "data/raw"
    train_rows = list(csv.DictReader(train_manifest.open(encoding="utf-8")))
    validation_rows = list(csv.DictReader(validation_manifest.open(encoding="utf-8")))
    all_rows = train_rows + validation_rows
    if not train_rows or not validation_rows:
        raise ValueError("Training and validation manifests must not be empty")
    if any(int(row["class_index"]) not in (0, 1, 2) for row in all_rows):
        raise ValueError("Training or validation manifest contains an invalid class index")
    train_paths = {row["source_relative_path"] for row in train_rows}
    validation_paths = {row["source_relative_path"] for row in validation_rows}
    if train_paths & validation_paths:
        raise ValueError("Training and validation manifests overlap")
    missing = [
        str(data_root / row["source_relative_path"])
        for row in all_rows
        if not (data_root / row["source_relative_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Manifest images are missing: {missing[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this smoke test but is unavailable")
    if args.dry_run:
        print({"model": args.model, "strategy": args.strategy, "seed": args.seed,
               "initialization": args.initialization, "config": config})
        return 0
    if not args.run_id:
        raise ValueError("--run-id is required for non-dry-run experiments")
    output = root / config.get("experiment_output_directory", "experiments") / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists; choose a new --run-id: {output}")
    seed_everything(args.seed, deterministic=True)
    train_loader = build_loader(train_manifest, data_root,
                                batch_size=config["batch_size"], num_workers=config["num_workers"],
                                pin_memory=config["pin_memory"], shuffle=True, seed=args.seed,
                                strategy=args.strategy, corruption_config=corruption_config)
    validation_loader = build_loader(validation_manifest, data_root,
                                     batch_size=config["batch_size"], num_workers=config["num_workers"],
                                     pin_memory=config["pin_memory"], seed=args.seed)
    model = create_model(args.model, pretrained=args.initialization == "imagenet",
                         head_type=args.head_type, dropout=0.2)
    if args.initialization == "imagenet" and model.initialization != "imagenet":
        raise RuntimeError("Requested ImageNet initialization was not loaded")
    manifest_hashes = {}
    for manifest in (train_manifest, validation_manifest):
        manifest_hashes[str(manifest.relative_to(root))] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata = {
        "model": args.model,
        "strategy": args.strategy,
        "seed": args.seed,
        "run_id": args.run_id,
        "configuration_id": f"{args.model}_{args.strategy}_{args.initialization}_seed{args.seed}",
        "initialization": model.initialization,
        "head_type": args.head_type,
        "class_mapping": {"Normal": 0, "Benign": 1, "Malignant": 2},
        "device": str(device),
        "amp_enabled": bool(config.get("mixed_precision", True) and device.type == "cuda"),
        "batch_size": config["batch_size"],
        "manifest_paths": [str(train_manifest.relative_to(root)), str(validation_manifest.relative_to(root))],
        "manifest_sha256": manifest_hashes,
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "status": "running",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    engine = TrainingEngine(model, config, device=device, output_dir=output,
                            checkpoint_metadata=metadata)
    (output / "run_status.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    start_epoch = 0
    history = []
    if args.resume:
        start_epoch, history = engine.load_checkpoint(root / args.resume)
    loss_fn = build_loss(model, config)
    try:
        engine.fit(train_loader, validation_loader, loss_fn, start_epoch=start_epoch, history=history)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failure_type"] = type(error).__name__
        metadata["failure_message"] = str(error)
        (output / "run_status.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    metadata["status"] = "completed"
    (output / "run_status.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
