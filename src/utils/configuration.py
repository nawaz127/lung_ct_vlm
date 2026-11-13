"""YAML configuration loading, validation, and project-relative path helpers."""

from pathlib import Path
import os
import re
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a project configuration is missing or invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return data


def resolve_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve environment substitutions and relative paths from project_root."""
    raw = str(value)
    match = re.fullmatch(r"\$\{([^:}]+):([^}]+)\}", raw)
    if match:
        raw = os.environ.get(match.group(1), match.group(2))
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project_root).resolve() / path).resolve()


def validate_project_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ("project_name", "python_min_version", "random_seed", "device", "class_names")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigurationError(f"Missing project configuration fields: {', '.join(missing)}")
    if not isinstance(config["project_name"], str) or not config["project_name"].strip():
        raise ConfigurationError("project_name must be a non-empty string")
    if not isinstance(config["random_seed"], int) or config["random_seed"] < 0:
        raise ConfigurationError("random_seed must be a non-negative integer")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("device must be one of: auto, cpu, cuda")
    classes = config["class_names"]
    if not isinstance(classes, list) or not classes or not all(isinstance(item, str) and item for item in classes):
        raise ConfigurationError("class_names must be a non-empty list of strings")
    return config


def load_project_config(path: str | Path) -> dict[str, Any]:
    return validate_project_config(load_yaml(path))