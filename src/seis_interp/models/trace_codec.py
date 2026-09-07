"""Shared per-trace temporal encoder and decoder for gather models."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

_GROUP_COUNT = 8


class TraceNodeEncoder(nn.Module):
    """Encode one trace waveform into a time-downsampled latent sequence."""

    def __init__(
        self,
        width: int,
        *,
        stem_kernel_size: int,
        time_downsample_factor: int,
    ) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.stem_kernel_size = _odd_positive_integer(stem_kernel_size, "stem_kernel_size")
        self.time_downsample_factor = _positive_integer(
            time_downsample_factor,
            "time_downsample_factor",
        )
        self.stem = nn.Conv1d(
            1,
            self.width,
            kernel_size=self.stem_kernel_size,
            padding=self.stem_kernel_size // 2,
        )
        self.norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        factor = self.time_downsample_factor
        self.downsample = nn.Conv1d(
            self.width,
            self.width,
            kernel_size=2 * factor - 1,
            stride=factor,
            padding=factor - 1,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return latent frames with shape ``[nodes, width, time / factor]``."""
        waveforms = _require_floating_tensor(waveforms, "waveforms")
        if waveforms.ndim != 3 or waveforms.shape[1] != 1:
            raise ValueError(
                f"waveforms must have shape (nodes, 1, time), got {tuple(waveforms.shape)}"
            )
        if waveforms.shape[2] % self.time_downsample_factor != 0:
            raise ValueError(
                "waveforms time dimension must be divisible by "
                f"time_downsample_factor {self.time_downsample_factor}, "
                f"got {waveforms.shape[2]}"
            )
        return self.downsample(F.silu(self.norm(self.stem(waveforms))))


class TraceNodeDecoder(nn.Module):
    """Decode target-node latent sequences back to full-rate residual traces."""

    def __init__(self, width: int, *, time_downsample_factor: int) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.time_downsample_factor = _positive_integer(
            time_downsample_factor,
            "time_downsample_factor",
        )
        factor = self.time_downsample_factor
        self.upsample = nn.ConvTranspose1d(
            self.width,
            self.width,
            kernel_size=2 * factor - 1,
            stride=factor,
            padding=factor - 1,
            output_padding=factor - 1,
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_GROUP_COUNT, self.width),
            nn.SiLU(),
            nn.Conv1d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.width, 1, kernel_size=1),
        )
        final_projection = self.head[-1]
        if not isinstance(final_projection, nn.Conv1d):
            raise AssertionError("trace node decoder head must end with Conv1d")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return residual traces with shape ``[nodes, time]``."""
        latents = _require_floating_tensor(latents, "latents")
        if latents.ndim != 3 or latents.shape[1] != self.width:
            raise ValueError(
                f"latents must have shape (nodes, {self.width}, frames), got {tuple(latents.shape)}"
            )
        return self.head(self.upsample(latents))[:, 0]


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _require_floating_tensor(value: object, name: str) -> torch.Tensor:
    tensor = _require_tensor(value, name)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    return tensor


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validated_width(value: object) -> int:
    width = _positive_integer(value, "width")
    if width % _GROUP_COUNT != 0:
        raise ValueError(f"width must be divisible by {_GROUP_COUNT}, got {width}")
    return width


def _odd_positive_integer(value: object, name: str) -> int:
    converted = _positive_integer(value, name)
    if converted % 2 == 0:
        raise ValueError(f"{name} must be odd, got {converted}")
    return converted
