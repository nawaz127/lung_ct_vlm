"""Sector 03 metadata loading and deterministic image preprocessing."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch


class PreprocessingError(ValueError):
    """Raised for invalid metadata or preprocessing configuration."""


REQUIRED_METADATA_FIELDS = {
    "relative_path", "absolute_path", "class_name", "sha256", "validation_status",
    "width", "height", "mode",
}


@dataclass(frozen=True)
class MetadataRecord:
    relative_path: str
    absolute_path: Path
    class_name: str
    sha256: str
    duplicate_group_id: int | None = None


def load_audit_metadata(audit_json, expected_classes, class_aliases=None):
    data = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    rows = data.get("image_metadata")
    if not isinstance(rows, list):
        raise PreprocessingError("Sector 02 audit JSON has no image_metadata list")
    aliases = {key.casefold(): value.casefold() for key, value in (class_aliases or {}).items()}
    classes = {name.casefold(): name for name in expected_classes}
    seen = set()
    records = []
    for row in rows:
        missing = REQUIRED_METADATA_FIELDS - row.keys()
        if missing:
            raise PreprocessingError(f"Metadata missing fields: {', '.join(sorted(missing))}")
        if row["validation_status"] != "valid":
            continue
        path = Path(row["absolute_path"])
        if not path.is_file():
            raise PreprocessingError(f"Source file is missing: {path}")
        relative = row["relative_path"]
        if relative in seen:
            raise PreprocessingError(f"Duplicate metadata entry: {relative}")
        seen.add(relative)
        class_key = aliases.get(str(row["class_name"]).casefold(), str(row["class_name"]).casefold())
        if class_key not in classes:
            raise PreprocessingError(f"Invalid canonical class: {row['class_name']}")
        records.append(MetadataRecord(relative, path, classes[class_key], row["sha256"]))
    records.sort(key=lambda item: item.relative_path.casefold())
    duplicate_groups = {}
    for row in data.get("duplicate_images", []):
        duplicate_groups.setdefault(int(row["duplicate_group"]), []).append(row["relative_path"])
    by_path = {path: group_id for group_id, paths in duplicate_groups.items() for path in paths}
    return [
        MetadataRecord(r.relative_path, r.absolute_path, r.class_name, r.sha256, by_path.get(r.relative_path))
        for r in records
    ], duplicate_groups


def validate_split_ratios(train, validation, test, tolerance=1e-8):
    if min(train, validation, test) <= 0 or abs(train + validation + test - 1.0) > tolerance:
        raise PreprocessingError("Split ratios must be positive and sum to 1.0")


def preprocess_image(path, size=(224, 224), mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225), normalize=True):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, ValueError) as error:
        raise PreprocessingError(f"Unable to preprocess image {path}: {error}") from error
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if normalize:
        tensor = (tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    return tensor