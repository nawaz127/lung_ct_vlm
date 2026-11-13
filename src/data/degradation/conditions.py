"""Centralized Sector 06 condition definitions."""

from pathlib import Path
import yaml

from .operators import gaussian_blur, gaussian_noise, reduce_contrast

CONDITION_IDS = (
    "clean",
    "gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3",
    "gaussian_blur_s1", "gaussian_blur_s2", "gaussian_blur_s3",
    "contrast_reduction_s1", "contrast_reduction_s2", "contrast_reduction_s3",
)


def load_degradation_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if tuple(config["conditions"]) != CONDITION_IDS:
        raise ValueError("degradation conditions must match the canonical order")
    return config


def build_condition(condition_id, config):
    if condition_id not in CONDITION_IDS:
        raise ValueError(f"Unknown degradation condition: {condition_id}")
    parameters = config["conditions"][condition_id]
    if condition_id == "clean":
        return lambda image, generator=None: image
    operation = parameters["type"]
    if operation == "gaussian_noise":
        return lambda image, generator=None: gaussian_noise(image, parameters["std"], generator)
    if operation == "gaussian_blur":
        return lambda image, generator=None: gaussian_blur(
            image, parameters["sigma"], _kernel_alias(parameters["kernel_size"])
        )
    if operation == "contrast_reduction":
        return lambda image, generator=None: reduce_contrast(image, parameters["factor"])
    raise ValueError(f"Unsupported degradation operation: {operation}")


def _kernel_alias(value):
    aliases = {"five": 5, "nine": 9, "fifteen": 15}
    return aliases.get(value, value)
