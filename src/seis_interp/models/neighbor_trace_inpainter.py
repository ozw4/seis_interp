"""Temporal CNN for interpolating a trace from fixed geometry neighbors."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

TEMPORAL_DILATIONS = (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1)
"""Ordered dilation schedule used by :class:`NeighborTraceInpainter`."""

_GROUP_COUNT = 8
DEFAULT_RESIDUAL_KERNEL_SIZE = 7
DEFAULT_STEM_KERNEL_SIZE = 15
DEFAULT_NEIGHBOR_ALIGNMENT_KERNEL_SIZE = 1
DEFAULT_TARGET_COORDINATE_COUNT = 3
DEFAULT_COORDINATE_CONDITIONING = "stem"
DEFAULT_NEIGHBOR_GATING = "none"
TARGET_COORDINATE_MASKED_SOFTMAX_GATING = "target_coordinate_masked_softmax"
DEFAULT_PREDICTION_REFERENCE = "none"
MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE = "masked_aligned_neighbor_mean"

_COORDINATE_CONDITIONING_MODES = frozenset((DEFAULT_COORDINATE_CONDITIONING, "film"))
_NEIGHBOR_GATING_MODES = frozenset(
    (DEFAULT_NEIGHBOR_GATING, TARGET_COORDINATE_MASKED_SOFTMAX_GATING)
)
_PREDICTION_REFERENCE_MODES = frozenset(
    (DEFAULT_PREDICTION_REFERENCE, MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE)
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validated_width(width: int) -> int:
    value = _positive_integer(width, "width")
    if value % _GROUP_COUNT != 0:
        raise ValueError(f"width must be divisible by {_GROUP_COUNT}, got {value}")
    return value


def _odd_positive_integer(value: int, name: str) -> int:
    validated = _positive_integer(value, name)
    if validated % 2 == 0:
        raise ValueError(f"{name} must be odd, got {validated}")
    return validated


def _validated_temporal_dilations(dilations: Iterable[int]) -> tuple[int, ...]:
    if isinstance(dilations, (str, bytes)):
        raise ValueError("temporal_dilations must be a non-empty iterable of positive integers")
    try:
        values = tuple(dilations)
    except TypeError as error:
        raise ValueError(
            "temporal_dilations must be a non-empty iterable of positive integers"
        ) from error
    if not values:
        raise ValueError("temporal_dilations must not be empty")
    validated = tuple(
        _positive_integer(value, f"temporal_dilations[{index}]")
        for index, value in enumerate(values)
    )
    if dilations is TEMPORAL_DILATIONS:
        return TEMPORAL_DILATIONS
    return validated


def _validated_coordinate_conditioning(value: str) -> str:
    if not isinstance(value, str) or value not in _COORDINATE_CONDITIONING_MODES:
        choices = ", ".join(repr(mode) for mode in sorted(_COORDINATE_CONDITIONING_MODES))
        raise ValueError(f"coordinate_conditioning must be one of {choices}, got {value!r}")
    return value


def _validated_neighbor_gating(value: str) -> str:
    if not isinstance(value, str) or value not in _NEIGHBOR_GATING_MODES:
        choices = ", ".join(repr(mode) for mode in sorted(_NEIGHBOR_GATING_MODES))
        raise ValueError(f"neighbor_gating must be one of {choices}, got {value!r}")
    return value


def _validated_prediction_reference(value: str) -> str:
    if not isinstance(value, str) or value not in _PREDICTION_REFERENCE_MODES:
        choices = ", ".join(repr(mode) for mode in sorted(_PREDICTION_REFERENCE_MODES))
        raise ValueError(f"prediction_reference must be one of {choices}, got {value!r}")
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


def _availability_masked_softmax_gates(
    logits: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    """Normalize available-neighbor logits to gates with mean one.

    Unavailable entries receive zero. A row without any available neighbor also
    receives all-zero, finite gates instead of the NaNs produced by a softmax of
    all negative infinity values.
    """
    available = availability > 0.0
    minimum = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~available, minimum)
    row_maximum = masked_logits.max(dim=1, keepdim=True).values
    shifted = (masked_logits - row_maximum).masked_fill(~available, 0.0)
    unnormalized = shifted.exp() * available.to(dtype=logits.dtype)
    denominator = unnormalized.sum(dim=1, keepdim=True).clamp_min(1.0)
    available_count = available.sum(dim=1, keepdim=True).to(dtype=logits.dtype)
    return unnormalized * available_count / denominator


def _identity_initialized_neighbor_alignment(
    neighbor_count: int,
    kernel_size: int,
) -> nn.Conv1d:
    """Build a trainable depthwise FIR without advancing the caller's RNG."""
    with torch.random.fork_rng(devices=[]):
        alignment = nn.Conv1d(
            neighbor_count,
            neighbor_count,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=neighbor_count,
            bias=False,
        )
    with torch.no_grad():
        alignment.weight.zero_()
        alignment.weight[:, 0, kernel_size // 2] = 1.0
    return alignment


class TemporalResidualBlock(nn.Module):
    """Gated depthwise temporal residual block with a configurable odd kernel."""

    def __init__(
        self,
        width: int,
        dilation: int,
        kernel_size: int = DEFAULT_RESIDUAL_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.dilation = _positive_integer(dilation, "dilation")
        self.kernel_size = _odd_positive_integer(kernel_size, "kernel_size")

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

    def forward(
        self,
        traces: torch.Tensor,
        modulation: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        normalized = self.norm(traces)
        if modulation is not None:
            modulation = _require_floating_tensor(modulation, "modulation")
            expected_shape = (traces.shape[0], self.width * 2)
            if modulation.shape != expected_shape:
                raise ValueError(
                    f"modulation must have shape {expected_shape}, got {tuple(modulation.shape)}"
                )
            if modulation.device != traces.device:
                raise ValueError("modulation and traces must share a device")
            if modulation.dtype != traces.dtype:
                raise TypeError(
                    "modulation must share the traces dtype, "
                    f"got {modulation.dtype} and {traces.dtype}"
                )
            scale_delta, shift = modulation.chunk(2, dim=1)
            normalized = normalized * (1.0 + scale_delta[..., None]) + shift[..., None]

        transformed = self.depthwise(F.silu(normalized))
        value, gate = self.expand(transformed).chunk(2, dim=1)
        return traces + self.contract(F.silu(value) * torch.sigmoid(gate))


class NeighborTraceInpainter(nn.Module):
    """Predict one trace from neighboring traces and its target geometry.

    Inputs are neighbor amplitudes ``[B, K, T]``, availability ``[B, K]``, and
    normalized target geometry coordinates ``[B, C]``. The availability and
    geometry values are broadcast over time and a ``[-1, 1]`` time channel is
    appended before temporal convolution. Boolean availability masks are accepted
    and converted to the neighbor amplitude dtype.
    """

    def __init__(
        self,
        neighbor_count: int,
        width: int = 128,
        target_coordinate_count: int = DEFAULT_TARGET_COORDINATE_COUNT,
        stem_kernel_size: int = DEFAULT_STEM_KERNEL_SIZE,
        residual_kernel_size: int = DEFAULT_RESIDUAL_KERNEL_SIZE,
        temporal_dilations: Iterable[int] = TEMPORAL_DILATIONS,
        coordinate_conditioning: str = DEFAULT_COORDINATE_CONDITIONING,
        neighbor_gating: str = DEFAULT_NEIGHBOR_GATING,
        neighbor_alignment_kernel_size: int = DEFAULT_NEIGHBOR_ALIGNMENT_KERNEL_SIZE,
        prediction_reference: str = DEFAULT_PREDICTION_REFERENCE,
    ) -> None:
        super().__init__()
        self.neighbor_count = _positive_integer(neighbor_count, "neighbor_count")
        self.width = _validated_width(width)
        self.target_coordinate_count = _positive_integer(
            target_coordinate_count, "target_coordinate_count"
        )
        self.stem_kernel_size = _odd_positive_integer(stem_kernel_size, "stem_kernel_size")
        self.residual_kernel_size = _odd_positive_integer(
            residual_kernel_size, "residual_kernel_size"
        )
        self.temporal_dilations = _validated_temporal_dilations(temporal_dilations)
        self.coordinate_conditioning = _validated_coordinate_conditioning(coordinate_conditioning)
        self.neighbor_gating = _validated_neighbor_gating(neighbor_gating)
        self.neighbor_alignment_kernel_size = _odd_positive_integer(
            neighbor_alignment_kernel_size,
            "neighbor_alignment_kernel_size",
        )
        self.prediction_reference = _validated_prediction_reference(prediction_reference)
        # Preserve the original public attribute while exposing the constructor field name.
        self.dilations = self.temporal_dilations
        self.input_channels = 2 * self.neighbor_count + self.target_coordinate_count + 1

        self.stem = nn.Conv1d(
            self.input_channels,
            self.width,
            kernel_size=self.stem_kernel_size,
            padding=self.stem_kernel_size // 2,
        )
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(
                self.width,
                dilation,
                kernel_size=self.residual_kernel_size,
            )
            for dilation in self.temporal_dilations
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_GROUP_COUNT, self.width),
            nn.SiLU(),
            nn.Conv1d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.width, 1, kernel_size=1),
        )
        self.coordinate_modulations = nn.ModuleList()
        if self.coordinate_conditioning == "film":
            for _ in self.temporal_dilations:
                projection = nn.Linear(self.target_coordinate_count, self.width * 2)
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                self.coordinate_modulations.append(projection)
        self.neighbor_gate_projection: nn.Linear | None = None
        if self.neighbor_gating == TARGET_COORDINATE_MASKED_SOFTMAX_GATING:
            self.neighbor_gate_projection = nn.Linear(
                self.target_coordinate_count,
                self.neighbor_count,
            )
            nn.init.zeros_(self.neighbor_gate_projection.weight)
            nn.init.zeros_(self.neighbor_gate_projection.bias)
        # Construct the optional module after every legacy module so their
        # initialization is unchanged. The helper also restores the CPU RNG.
        self.neighbor_alignment: nn.Conv1d | None = None
        if self.neighbor_alignment_kernel_size > 1:
            self.neighbor_alignment = _identity_initialized_neighbor_alignment(
                self.neighbor_count,
                self.neighbor_alignment_kernel_size,
            )
        if self.prediction_reference == MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE:
            # The reference supplies the initial waveform. The CNN starts as an
            # exact zero correction without consuming additional randomness.
            final_projection = self.head[-1]
            if not isinstance(final_projection, nn.Conv1d):
                raise AssertionError("neighbor inpainter head must end with Conv1d")
            nn.init.zeros_(final_projection.weight)
            nn.init.zeros_(final_projection.bias)

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
        if target_coordinates.shape != (batch_size, self.target_coordinate_count):
            raise ValueError(
                "target_coordinates must have shape "
                f"({batch_size}, {self.target_coordinate_count}), "
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

        gated_neighbors = neighbors
        if self.neighbor_gate_projection is not None:
            gate_logits = self.neighbor_gate_projection(target_coordinates)
            gates = _availability_masked_softmax_gates(gate_logits, availability)
            gated_neighbors = neighbors * gates[..., None]
        if self.neighbor_alignment is not None:
            # Gating is a time-invariant scalar per neighbor and therefore
            # commutes with a depthwise temporal FIR. Apply it first, then make
            # unavailable channels explicitly zero before alignment.
            available = (availability > 0.0).to(dtype=neighbors.dtype)
            gated_neighbors = self.neighbor_alignment(gated_neighbors * available[..., None])

        prediction_reference: torch.Tensor | None = None
        if self.prediction_reference == MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE:
            available = availability > 0.0
            available_float = available.to(dtype=neighbors.dtype)
            available_count = available_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            prediction_reference = (gated_neighbors * available_float[..., None]).sum(
                dim=1
            ) / available_count

        availability_channels = availability[..., None].expand(-1, -1, time_count)
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
        geometry_channels = target_coordinates[..., None].expand(-1, -1, time_count)
        features = torch.cat(
            (gated_neighbors, availability_channels, geometry_channels, time_channel),
            dim=1,
        )
        hidden = self.stem(features)
        if self.coordinate_conditioning == "film":
            for block, projection in zip(
                self.blocks,
                self.coordinate_modulations,
                strict=True,
            ):
                hidden = block(hidden, projection(target_coordinates))
        else:
            for block in self.blocks:
                hidden = block(hidden)
        prediction = self.head(hidden)[:, 0]
        if prediction_reference is not None:
            prediction = prediction + prediction_reference
        return prediction
