"""Temporal CNN for interpolating a trace from fixed geometry neighbors."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

TEMPORAL_DILATIONS = (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1)
"""Ordered dilation schedule used by :class:`NeighborTraceInpainter`."""

_GROUP_COUNT = 8
_BLOCK_KERNEL_SIZE = 7
_STEM_KERNEL_SIZE = 15
_TARGET_GEOMETRY_CHANNELS = 3


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validated_width(width: int) -> int:
    value = _positive_integer(width, "width")
    if value % _GROUP_COUNT != 0:
        raise ValueError(f"width must be divisible by {_GROUP_COUNT}, got {value}")
    return value


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _require_floating_tensor(value: object, name: str) -> torch.Tensor:
    tensor = _require_tensor(value, name)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    return tensor


class TemporalResidualBlock(nn.Module):
    """Gated depthwise temporal residual block with a fixed seven-sample kernel."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.dilation = _positive_integer(dilation, "dilation")
        self.kernel_size = _BLOCK_KERNEL_SIZE

        self.norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        self.depthwise = nn.Conv1d(
            self.width,
            self.width,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size // 2) * self.dilation,
            dilation=self.dilation,
            groups=self.width,
        )
        self.expand = nn.Conv1d(self.width, self.width * 2, kernel_size=1)
        self.contract = nn.Conv1d(self.width, self.width, kernel_size=1)

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        """Transform ``[batch, width, time]`` while preserving its shape."""
        traces = _require_floating_tensor(traces, "traces")
        if traces.ndim != 3:
            raise ValueError(
                f"traces must have shape (batch, width, time), got {tuple(traces.shape)}"
            )
        if traces.shape[1] != self.width:
            raise ValueError(f"traces must have {self.width} channels, got {traces.shape[1]}")
        if traces.shape[0] == 0 or traces.shape[2] == 0:
            raise ValueError("traces batch and time dimensions must be non-empty")

        transformed = self.depthwise(F.silu(self.norm(traces)))
        value, gate = self.expand(transformed).chunk(2, dim=1)
        return traces + self.contract(F.silu(value) * torch.sigmoid(gate))


class NeighborTraceInpainter(nn.Module):
    """Predict one trace from neighboring traces and its target geometry.

    Inputs are neighbor amplitudes ``[B, K, T]``, availability ``[B, K]``, and
    three normalized target geometry coordinates ``[B, 3]``. The availability
    and geometry values are broadcast over time and a ``[-1, 1]`` time channel
    is appended before temporal convolution. Boolean availability masks are
    accepted and converted to the neighbor amplitude dtype.
    """

    def __init__(self, neighbor_count: int, width: int = 128) -> None:
        super().__init__()
        self.neighbor_count = _positive_integer(neighbor_count, "neighbor_count")
        self.width = _validated_width(width)
        self.dilations = TEMPORAL_DILATIONS
        self.input_channels = 2 * self.neighbor_count + _TARGET_GEOMETRY_CHANNELS + 1

        self.stem = nn.Conv1d(
            self.input_channels,
            self.width,
            kernel_size=_STEM_KERNEL_SIZE,
            padding=_STEM_KERNEL_SIZE // 2,
        )
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(self.width, dilation) for dilation in self.dilations
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_GROUP_COUNT, self.width),
            nn.SiLU(),
            nn.Conv1d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.width, 1, kernel_size=1),
        )

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted amplitudes with shape ``[batch, time]``."""
        neighbors = _require_floating_tensor(neighbors, "neighbors")
        availability = _require_tensor(availability, "availability")
        target_coordinates = _require_floating_tensor(target_coordinates, "target_coordinates")

        if neighbors.ndim != 3:
            raise ValueError(
                "neighbors must have shape (batch, neighbor_count, time), "
                f"got {tuple(neighbors.shape)}"
            )
        batch_size, neighbor_count, time_count = neighbors.shape
        if neighbor_count != self.neighbor_count:
            raise ValueError(
                f"neighbors must contain {self.neighbor_count} channels, got {neighbor_count}"
            )
        if batch_size == 0 or time_count == 0:
            raise ValueError("neighbors batch and time dimensions must be non-empty")
        if availability.shape != (batch_size, self.neighbor_count):
            raise ValueError(
                "availability must have shape "
                f"({batch_size}, {self.neighbor_count}), got {tuple(availability.shape)}"
            )
        if target_coordinates.shape != (batch_size, _TARGET_GEOMETRY_CHANNELS):
            raise ValueError(
                "target_coordinates must have shape "
                f"({batch_size}, {_TARGET_GEOMETRY_CHANNELS}), "
                f"got {tuple(target_coordinates.shape)}"
            )
        if not (neighbors.device == availability.device == target_coordinates.device):
            raise ValueError("neighbors, availability, and target_coordinates must share a device")
        if availability.dtype == torch.bool:
            availability = availability.to(dtype=neighbors.dtype)
        elif not availability.is_floating_point():
            raise TypeError(
                "availability must have a boolean or floating-point dtype, "
                f"got {availability.dtype}"
            )
        elif availability.dtype != neighbors.dtype:
            raise TypeError(
                "floating-point availability must share the neighbors dtype, "
                f"got {availability.dtype} and {neighbors.dtype}"
            )
        if target_coordinates.dtype != neighbors.dtype:
            raise TypeError(
                "target_coordinates must share the neighbors dtype, "
                f"got {target_coordinates.dtype} and {neighbors.dtype}"
            )

        availability_channels = availability[..., None].expand(-1, -1, time_count)
        geometry_channels = target_coordinates[..., None].expand(-1, -1, time_count)
        time_channel = (
            torch.linspace(
                -1.0,
                1.0,
                time_count,
                device=neighbors.device,
                dtype=neighbors.dtype,
            )
            .view(1, 1, time_count)
            .expand(batch_size, 1, time_count)
        )
        features = torch.cat(
            (neighbors, availability_channels, geometry_channels, time_channel),
            dim=1,
        )
        hidden = self.stem(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)[:, 0]
