from hashlib import sha256
import json

from PIL import Image

from data.audit import audit_dataset, write_reports
from data.audit.discovery import discover_dataset
from data.audit.validation import investigate_audit


def _image(path, color="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path, format="PNG")


def test_discovery_is_case_insensitive_recursive_and_sorted(tmp_path):
    _image(tmp_path / "normal" / "nested" / "b.png")
    _image(tmp_path / "normal" / "a.png")
    (tmp_path / "Benign").mkdir()
    result = discover_dataset(tmp_path, ["Normal", "Benign"], {".png"})
    assert [item.relative_path for item in result.candidates] == ["normal/a.png", "normal/nested/b.png"]
    assert result.empty_classes == ["Benign"]


def test_configured_class_alias_maps_to_canonical_label(tmp_path):
    _image(tmp_path / "Bengin cases" / "image.png")
    result = discover_dataset(
        tmp_path,
        ["Normal", "Benign", "Malignant"],
        {".png"},
        class_aliases={"Bengin cases": "Benign"},
    )
    assert result.missing_classes == ["Normal", "Malignant"]
    assert result.candidates[0].class_name == "Benign"


def test_audit_metadata_duplicates_and_source_safety(tmp_path):
    first = tmp_path / "Normal" / "one.png"
    second = tmp_path / "Normal" / "two.png"
    cross = tmp_path / "Malignant" / "copy.png"
    _image(first, "red")
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(first.read_bytes())
    cross.parent.mkdir(parents=True, exist_ok=True)
    cross.write_bytes(first.read_bytes())
    before = {path: sha256(path.read_bytes()).hexdigest() for path in (first, second, cross)}

    result = audit_dataset(tmp_path, ["Normal", "Benign", "Malignant"], {".png"})

    assert result["valid_image_count"] == 3
    assert result["cross_class_duplicate_group_count"] == 1
    assert result["status"] == "FAIL"
    assert {path: sha256(path.read_bytes()).hexdigest() for path in (first, second, cross)} == before


def test_invalid_files_do_not_stop_audit_and_reports_have_stable_headers(tmp_path):
    _image(tmp_path / "Normal" / "valid.png")
    invalid = tmp_path / "Normal" / "broken.png"
    invalid.write_bytes(b"not an image")
    result = audit_dataset(tmp_path, ["Normal"], {".png"})
    output = write_reports(result, tmp_path / "reports", run_name="run")
    assert result["valid_image_count"] == 1
    assert result["invalid_image_count"] == 1
    assert json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))["status"] == "WARNING"
    assert (output / "invalid_images.csv").read_text(encoding="utf-8").startswith("relative_path,absolute_path")


def test_validation_reports_cross_class_groups_dimensions_and_distribution(tmp_path):
    first = tmp_path / "Normal" / "one.png"
    second = tmp_path / "Benign" / "copy.png"
    _image(first, "red")
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(first.read_bytes())
    audit = audit_dataset(tmp_path, ["Normal", "Benign"], {".png"})
    findings = investigate_audit(audit)
    assert findings["total_metadata_records"] == 2
    assert findings["metadata_hash_count"] == 2
    assert findings["cross_class_duplicate_group_count"] == 1
    assert findings["duplicate_recheck_matches_audit"] is True
    assert findings["dimension_distribution"] == [{"width": 8, "height": 6, "count": 2, "percentage": 100.0}]
    assert findings["source_files_unchanged"] is True
