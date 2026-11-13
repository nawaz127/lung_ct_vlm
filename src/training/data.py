"""Manifest-backed dataset and DataLoader construction."""

import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import preprocess_image
from data.degradation.operators import gaussian_blur, gaussian_noise, reduce_contrast


class ManifestDataset(Dataset):
    def __init__(self, manifest, data_root=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.rows = list(csv.DictReader(Path(manifest).open(encoding="utf-8")))
        self.data_root = Path(data_root) if data_root else None
        self.mean = tuple(mean)
        self.std = tuple(std)
        if not self.rows:
            raise ValueError(f"Manifest is empty: {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["source_relative_path"])
        if not path.is_absolute() and self.data_root:
            path = self.data_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        image = preprocess_image(path, mean=self.mean, std=self.std)
        return image, int(row["class_index"])


class CorruptionAwareManifestDataset(ManifestDataset):
    """Keep one clean view and one deterministic training-corrupted view."""

    def __init__(self, manifest, data_root=None, corruption_config=None,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 seed=42):
        super().__init__(manifest, data_root, mean, std)
        if not corruption_config or not corruption_config.get("conditions"):
            raise ValueError("corruption-aware training requires non-empty conditions")
        clean_fraction = float(corruption_config.get("clean_fraction", 0.5))
        if not 0 < clean_fraction < 1:
            raise ValueError("clean_fraction must be between 0 and 1")
        self.corruption_config = corruption_config
        self.seed = int(seed)

    def __len__(self):
        return len(self.rows) * 2

    def __getitem__(self, index):
        row_index = index // 2
        row = self.rows[row_index]
        path = Path(row["source_relative_path"])
        if not path.is_absolute() and self.data_root:
            path = self.data_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        image = preprocess_image(path, normalize=False)
        if index % 2 == 0:
            return self._normalize(image), int(row["class_index"])
        identity = row.get("source_relative_path", str(row_index))
        digest = hashlib.sha256(f"{self.seed}|{identity}".encode("utf-8")).digest()
        condition_names = tuple(self.corruption_config["conditions"])
        condition = condition_names[int.from_bytes(digest[:4], "big") % len(condition_names)]
        parameters = self.corruption_config["conditions"][condition]
        values = parameters.get(
            {"gaussian_noise": "std", "gaussian_blur": "sigma",
             "contrast_reduction": "factor"}[parameters["type"]]
        )
        value_index = int.from_bytes(digest[4:8], "big") % len(values)
        generator = torch.Generator().manual_seed(
            int.from_bytes(digest[8:16], "big") % (2**63 - 1)
        )
        if parameters["type"] == "gaussian_noise":
            image = gaussian_noise(image, values[value_index], generator)
        elif parameters["type"] == "gaussian_blur":
            kernels = parameters.get("kernel_size", [3] * len(values))
            image = gaussian_blur(image, values[value_index], kernels[value_index])
        elif parameters["type"] == "contrast_reduction":
            image = reduce_contrast(image, values[value_index])
        else:
            raise ValueError(f"Unsupported training corruption: {parameters['type']}")
        return self._normalize(image), int(row["class_index"])

    def _normalize(self, image):
        return (image - torch.tensor(self.mean).view(3, 1, 1)) / torch.tensor(self.std).view(3, 1, 1)


def build_loader(manifest, data_root=None, batch_size=32, num_workers=0, pin_memory=True,
                 shuffle=False, seed=42, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), strategy="clean_baseline",
                 corruption_config=None):
    if strategy == "corruption_aware":
        dataset = CorruptionAwareManifestDataset(
            manifest, data_root, corruption_config, mean, std, seed
        )
    elif strategy == "clean_baseline":
        dataset = ManifestDataset(manifest, data_root, mean, std)
    else:
        raise ValueError(f"Unsupported training strategy: {strategy}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init_fn = None
    if num_workers:
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % (2**32)
            import random
            import numpy as np
            random.seed(worker_seed)
            np.random.seed(worker_seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      generator=generator, worker_init_fn=worker_init_fn,
                      persistent_workers=num_workers > 0)
