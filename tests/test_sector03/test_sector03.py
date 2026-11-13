import json

from PIL import Image
import pytest

from data.preprocessing import (
    MetadataRecord,
    PreprocessingError,
    load_audit_metadata,
    preprocess_image,
)
from data.splitting import split_records


def image(path, mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (12, 8), 128).save(path, format="PNG")


def audit_fixture(tmp_path):
    rows = []
    for label in ("Normal", "Benign", "Malignant"):
        path = tmp_path / label / f"{label}.png"
        image(path)
        rows.append({
            "relative_path": f"{label}/{label}.png",
            "absolute_path": str(path),
            "class_name": label,
            "sha256": label,
            "validation_status": "valid",
            "width": 12, "height": 8, "mode": "RGB",
        })
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"image_metadata": rows, "duplicate_images": []}), encoding="utf-8")
    return audit


def test_metadata_loading_and_alias(tmp_path):
    audit = audit_fixture(tmp_path)
    data = json.loads(audit.read_text())
    data["image_metadata"][1]["class_name"] = "Bengin"
    audit.write_text(json.dumps(data), encoding="utf-8")
    records, groups = load_audit_metadata(audit, ["Normal", "Benign", "Malignant"], {"Bengin": "Benign"})
    assert [record.class_name for record in records] == ["Benign", "Malignant", "Normal"]
    assert groups == {}


def test_preprocess_supports_grayscale_rgba_and_normalizes(tmp_path):
    gray = tmp_path / "gray.png"
    rgba = tmp_path / "rgba.png"
    image(gray, "L")
    image(rgba, "RGBA")
    assert preprocess_image(gray).shape == (3, 224, 224)
    assert preprocess_image(rgba, normalize=False).min() >= 0
    assert preprocess_image(rgba, normalize=False).max() <= 1


def test_cross_class_duplicate_error_and_exclusion(tmp_path):
    records = [
        MetadataRecord("a.png", tmp_path / "a.png", "Normal", "same", 1),
        MetadataRecord("b.png", tmp_path / "b.png", "Benign", "same", 1),
    ]
    with pytest.raises(PreprocessingError, match="Unresolved"):
        split_records(records, .7, .15, .15, 42, {1: ["a.png", "b.png"]}, {1}, "error")
    result = split_records(records, .7, .15, .15, 42, {1: ["a.png", "b.png"]}, {1},
                           "exclude_conflicted_groups")
    assert result == []


def test_same_class_duplicate_group_stays_together_and_is_reproducible(tmp_path):
    records = [
        MetadataRecord("a.png", tmp_path / "a.png", "Normal", "same", 1),
        MetadataRecord("b.png", tmp_path / "b.png", "Normal", "same", 1),
        MetadataRecord("c.png", tmp_path / "c.png", "Normal", "c", None),
        MetadataRecord("d.png", tmp_path / "d.png", "Normal", "d", None),
    ]
    first = split_records(records, .5, .25, .25, 42, {1: ["a.png", "b.png"]})
    second = split_records(records, .5, .25, .25, 42, {1: ["a.png", "b.png"]})
    assert first == second
    assignments = {row["source_relative_path"]: row["split"] for row in first}
    assert assignments["a.png"] == assignments["b.png"]
