"""Explicit CUDA training precision with no dtype or device fallback."""

import torch


def validate_mixed_precision(mode: str) -> str:
    """Require an explicit supported mode, including at direct trainer entry."""
    if not isinstance(mode, str) or mode not in ("off", "fp16", "bf16"):
        raise ValueError("training.mixed_precision must be off, fp16 or bf16")
    return mode


def mixed_precision_dtype(mode: str, device: torch.device | str) -> torch.dtype | None:
    """Validate the selected CUDA device before creating autocast or a scaler."""
    validate_mixed_precision(mode)
    if mode == "off":
        return None
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("training.mixed_precision requires an available CUDA device")
    with torch.cuda.device(selected):
        if mode == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError(
                "training.mixed_precision bf16 is unsupported on the selected CUDA device"
            )
    return torch.float16 if mode == "fp16" else torch.bfloat16
