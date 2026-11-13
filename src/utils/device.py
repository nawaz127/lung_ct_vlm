"""PyTorch device selection and CUDA diagnostics."""

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    cuda_available: bool
    cuda_build: str | None
    device_name: str | None
    total_memory_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("requested device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def inspect_device(requested: str = "auto") -> DeviceInfo:
    device = select_device(requested)
    available = torch.cuda.is_available()
    index = device.index if device.type == "cuda" and device.index is not None else 0
    name = torch.cuda.get_device_name(index) if available else None
    memory = torch.cuda.get_device_properties(index).total_memory if available else None
    return DeviceInfo(requested, str(device), available, torch.version.cuda, name, memory)