"""Centralized logging setup from the project YAML schema."""

import logging
from pathlib import Path
from typing import Any


def configure_logging(config: dict[str, Any], project_root: str | Path = ".") -> logging.Logger:
    settings = config.get("logging", config)
    level_name = str(settings.get("level", "INFO")).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown logging level: {level_name}")
    output_dir = Path(project_root) / settings.get("output_dir", "outputs/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(settings.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger