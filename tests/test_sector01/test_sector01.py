import random

import numpy as np
import pytest
import torch

from utils.configuration import ConfigurationError, load_project_config, validate_project_config
from utils.device import inspect_device, select_device
from utils.reproducibility import seed_everything


ROOT = __file__


def test_project_config_loads_and_validates():
    config = load_project_config("configs/project.yaml")
    assert config["project_name"] == "lung_ct_robustness_xai_vlm"
    assert config["class_names"] == ["Normal", "Benign", "Malignant"]


def test_invalid_config_is_rejected():
    with pytest.raises(ConfigurationError, match="device"):
        validate_project_config({
            "project_name": "x", "python_min_version": "3.10",
            "random_seed": 1, "device": "tpu", "class_names": ["x"],
        })


def test_device_report_matches_runtime():
    info = inspect_device("auto")
    assert info.selected in {"cpu", "cuda"}
    assert info.cuda_available == torch.cuda.is_available()
    assert select_device("cpu").type == "cpu"


def test_seed_reproducibility():
    seed_everything(42)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(42)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
