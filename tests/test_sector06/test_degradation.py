import csv

import numpy as np
from PIL import Image
import torch

from data.degradation import (
    CONDITION_IDS,
    DegradedManifestDataset,
    build_condition,
    gaussian_blur,
    gaussian_noise,
    load_degradation_config,
    reduce_contrast,
    stable_corruption_seed,
)


def _config():
    return {
        "seed": 42,
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "conditions": {
            "clean": {},
            "gaussian_noise_s1": {"type": "gaussian_noise", "std": 0.03},
            "gaussian_noise_s2": {"type": "gaussian_noise", "std": 0.08},
            "gaussian_noise_s3": {"type": "gaussian_noise", "std": 0.15},
            "gaussian_blur_s1": {"type": "gaussian_blur", "sigma": 0.8, "kernel_size": 5},
            "gaussian_blur_s2": {"type": "gaussian_blur", "sigma": 1.5, "kernel_size": 9},
            "gaussian_blur_s3": {"type": "gaussian_blur", "sigma": 3.0, "kernel_size": 15},
            "contrast_reduction_s1": {"type": "contrast_reduction", "factor": 0.75},
            "contrast_reduction_s2": {"type": "contrast_reduction", "factor": 0.5},
            "contrast_reduction_s3": {"type": "contrast_reduction", "factor": 0.25},
        },
    }


def test_operators_preserve_shape_range_and_finite_values():
    image = torch.rand(3, 224, 224)
    generator = torch.Generator().manual_seed(7)
    for output in (
        gaussian_noise(image, 0.1, generator),
        gaussian_blur(image, 1.5, 9),
        reduce_contrast(image, 0.5),
    ):
        assert output.shape == image.shape
        assert output.isfinite().all()
        assert 0 <= output.min() and output.max() <= 1


def test_noise_is_reproducible_and_severity_increases():
    image = torch.full((3, 224, 224), 0.5)
    first = gaussian_noise(image, 0.1, torch.Generator().manual_seed(9))
    second = gaussian_noise(image, 0.1, torch.Generator().manual_seed(9))
    different = gaussian_noise(image, 0.1, torch.Generator().manual_seed(10))
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert (gaussian_noise(image, 0.15, torch.Generator().manual_seed(9)) - image).abs().mean() > (
        gaussian_noise(image, 0.03, torch.Generator().manual_seed(9)) - image
    ).abs().mean()


def test_all_conditions_preserve_sample_identity_and_labels(tmp_path):
    image_path = tmp_path / "sample.JPG"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(image_path)
    manifest = tmp_path / "test.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source_relative_path", "class_index"])
        writer.writeheader()
        writer.writerow({"sample_id": "sample-1", "source_relative_path": "sample.JPG", "class_index": 2})
    config = _config()
    outputs = {}
    for condition in CONDITION_IDS:
        dataset = DegradedManifestDataset(manifest, tmp_path, condition, config)
        image, label, sample_id = dataset[0]
        outputs[condition] = image
        assert image.shape == (3, 224, 224)
        assert label == 2
        assert sample_id == "sample-1"
    assert torch.equal(outputs["gaussian_noise_s1"],
                       DegradedManifestDataset(manifest, tmp_path, "gaussian_noise_s1", config)[0][0])


def test_stable_seed_does_not_depend_on_iteration_order():
    first = stable_corruption_seed(42, "sample-1", "gaussian_noise_s1", 0.03)
    second = stable_corruption_seed(42, "sample-1", "gaussian_noise_s1", 0.03)
    other = stable_corruption_seed(42, "sample-2", "gaussian_noise_s1", 0.03)
    assert first == second
    assert first != other


def test_config_has_canonical_conditions(tmp_path):
    config_path = tmp_path / "degradation.yaml"
    config_path.write_text(
        "conditions:\n" + "\n".join(f"  {condition}: {{}}" for condition in CONDITION_IDS),
        encoding="utf-8",
    )
    config = load_degradation_config(config_path)
    assert tuple(config["conditions"]) == CONDITION_IDS
    assert build_condition("clean", config)(torch.rand(3, 4, 4)).shape == (3, 4, 4)
