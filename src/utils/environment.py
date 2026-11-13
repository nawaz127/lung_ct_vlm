"""Runtime dependency, Python, PyTorch, and CUDA reporting."""

from dataclasses import asdict
from importlib import import_module, metadata
import platform
import sys
from typing import Any

from .device import inspect_device


def package_versions(packages: list[str]) -> dict[str, str | None]:
    result = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def environment_report(requested_device: str = "auto", packages: list[str] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(packages or ["numpy", "PyYAML", "torch", "pytest"]),
    }
    try:
        torch = import_module("torch")
        report["pytorch_version"] = torch.__version__
    except ImportError:
        report["pytorch_version"] = None
        report["device"] = None
        return report
    report["device"] = asdict(inspect_device(requested_device))
    return report


def validate_environment(min_python: tuple[int, int] = (3, 10), requested_device: str = "auto") -> dict[str, Any]:
    if sys.version_info[:2] < min_python:
        raise RuntimeError(f"Python {min_python[0]}.{min_python[1]} or newer is required")
    report = environment_report(requested_device)
    missing = [name for name, version in report["packages"].items() if version is None]
    if missing:
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")
    return report