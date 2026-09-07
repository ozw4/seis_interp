"""Resolve explicitly requested model devices without a silent CPU fallback."""

from __future__ import annotations

import torch

from seis_interp.configuration import ConfigurationError


def resolve_device(value: str, *, config_key: str = "training.device") -> torch.device:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{config_key} must be a non-empty string")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {value!r} is unavailable")
    # Verify availability and resolve aliases such as cpu:0/cuda.
    return torch.empty(0, device=device).device
