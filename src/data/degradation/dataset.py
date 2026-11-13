"""Manifest-backed deterministic clean and corrupted dataset views."""

import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import preprocess_image
from .conditions import build_condition


def stable_corruption_seed(global_seed, sample_id, condition_id, severity):
    identity = f"{global_seed}|{sample_id}|{condition_id}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


class DegradedManifestDataset(Dataset):
    def __init__(self, manifest, data_root, condition_id, config, mean=None, std=None,
                 identity_by_path=False):
        self.manifest = Path(manifest)
        self.rows = list(csv.DictReader(self.manifest.open(encoding="utf-8")))
        if not self.rows:
            raise ValueError(f"Manifest is empty: {manifest}")
        self.data_root = Path(data_root)
        self.condition_id = condition_id
        self.config = config
        self.mean = tuple(mean or config["normalization"]["mean"])
        self.std = tuple(std or config["normalization"]["std"])
        self.operation = build_condition(condition_id, config)
        self.global_seed = int(config["seed"])
        self.identity_by_path = identity_by_path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative = row["source_relative_path"]
        path = self.data_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        image = preprocess_image(path, normalize=False)
        seed = stable_corruption_seed(
            self.global_seed, relative, self.condition_id,
            self.config["conditions"][self.condition_id].get("std", ""),
        )
        generator = torch.Generator(device=image.device)
        generator.manual_seed(seed)
        image = self.operation(image, generator)
        image = (image - torch.tensor(self.mean).view(3, 1, 1)) / torch.tensor(self.std).view(3, 1, 1)
        identity = relative if self.identity_by_path else row.get("sample_id", relative)
        return image, int(row["class_index"]), identity


def build_degraded_loader(manifest, data_root, condition_id, config, batch_size=32,
                          num_workers=0, pin_memory=True, shuffle=False):
    dataset = DegradedManifestDataset(manifest, data_root, condition_id, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)
