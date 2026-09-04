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
DEFAULT_COARSE_SHIFT_SAMPLES_PER_RELATIVE_RECEIVER_Y_INDEX = 0
DEFAULT_TARGET_COORDINATE_COUNT = 3
DEFAULT_COORDINATE_CONDITIONING = "stem"
DEFAULT_NEIGHBOR_GATING = "none"
TARGET_COORDINATE_MASKED_SOFTMAX_GATING = "target_coordinate_masked_softmax"
DEFAULT_PREDICTION_REFERENCE = "none"
MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE = "masked_aligned_neighbor_mean"
SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE = "same_line_exact_receiver_linear_bracketing"
SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE = (
    "same_line_exact_receiver_linear_bracketing_channels"
)

_COORDINATE_CONDITIONING_MODES = frozenset((DEFAULT_COORDINATE_CONDITIONING, "film"))
_NEIGHBOR_GATING_MODES = frozenset(
    (DEFAULT_NEIGHBOR_GATING, TARGET_COORDINATE_MASKED_SOFTMAX_GATING)
)
_PREDICTION_REFERENCE_MODES = frozenset(
    (
        DEFAULT_PREDICTION_REFERENCE,
        MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    )
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
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


def _validated_coarse_alignment_offsets(
    value: Iterable[Iterable[int]] | None,
    *,
    neighbor_count: int,
    samples_per_relative_receiver_y_index: int,
) -> tuple[tuple[int, int, int, int], ...] | None:
    if samples_per_relative_receiver_y_index == 0:
        if value is not None:
            raise ValueError(
                "neighbor_offsets must be omitted when "
                "coarse_shift_samples_per_relative_receiver_y_index is 0"
            )
        return None
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(
            "neighbor_offsets must contain the exact four-axis multiline offsets when "
            "coarse_shift_samples_per_relative_receiver_y_index is positive"
        )
    try:
        raw_offsets = tuple(tuple(offset) for offset in value)
    except TypeError as error:
        raise ValueError(
            "neighbor_offsets must contain the exact four-axis multiline offsets"
        ) from error
    if len(raw_offsets) != neighbor_count:
        raise ValueError(
            f"neighbor_offsets must contain {neighbor_count} offsets, got {len(raw_offsets)}"
        )
    offsets: list[tuple[int, int, int, int]] = []
    for index, offset in enumerate(raw_offsets):
        if len(offset) != 4:
            raise ValueError(f"neighbor_offsets[{index}] must contain four integers")
        if any(
            isinstance(component, bool) or not isinstance(component, Integral)
            for component in offset
        ):
            raise ValueError(f"neighbor_offsets[{index}] must contain integers")
        offsets.append(tuple(int(component) for component in offset))
    if len(set(offsets)) != len(offsets):
        raise ValueError("neighbor_offsets must not contain duplicates")
    if (0, 0, 0, 0) in offsets:
        raise ValueError("neighbor_offsets must exclude the target center offset")
    return tuple(offsets)


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
        coarse_shift_samples_per_relative_receiver_y_index: int = (
            DEFAULT_COARSE_SHIFT_SAMPLES_PER_RELATIVE_RECEIVER_Y_INDEX
        ),
        neighbor_offsets: Iterable[Iterable[int]] | None = None,
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
        self.reference_neighbor_count = {
            SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE: 1,
            SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE: 2,
        }.get(self.prediction_reference, 0)
        self.local_neighbor_count = self.neighbor_count - self.reference_neighbor_count
        if self.local_neighbor_count < 1:
            raise ValueError(
                "same-line bracketing reference requires at least one local neighbor channel"
            )
        self.coarse_shift_samples_per_relative_receiver_y_index = _nonnegative_integer(
            coarse_shift_samples_per_relative_receiver_y_index,
            "coarse_shift_samples_per_relative_receiver_y_index",
        )
        coarse_alignment_offsets = _validated_coarse_alignment_offsets(
            neighbor_offsets,
            neighbor_count=self.neighbor_count,
            samples_per_relative_receiver_y_index=(
                self.coarse_shift_samples_per_relative_receiver_y_index
            ),
        )
        self.register_buffer("neighbor_offsets", None)
        self.register_buffer("coarse_sample_shifts", None)
        if coarse_alignment_offsets is not None:
            offset_tensor = torch.tensor(coarse_alignment_offsets, dtype=torch.int64)
            self.neighbor_offsets = offset_tensor
            self.coarse_sample_shifts = (
                offset_tensor[:, 3] * self.coarse_shift_samples_per_relative_receiver_y_index
            )
        # Both names are public: `dilations` is the read API, `temporal_dilations` the
        # constructor field.
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
        if self.prediction_reference in {
            MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE,
            SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
            SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
        }:
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

        raw_bracketing_reference: torch.Tensor | None = None
        if self.prediction_reference == SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE:
            reference_available = (availability[:, -1] > 0.0).to(dtype=neighbors.dtype)
            raw_bracketing_reference = neighbors[:, -1] * reference_available[:, None]
        elif (
            self.prediction_reference
            == SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
        ):
            raw_bracketing_reference = (neighbors[:, -2:] * availability[:, -2:, None]).sum(dim=1)

        gated_neighbors = neighbors
        aligned_availability: torch.Tensor | None = None
        if self.coarse_shift_samples_per_relative_receiver_y_index > 0:
            gated_neighbors, aligned_availability = self._coarse_align_neighbors(
                neighbors,
                availability,
            )
        if self.neighbor_gate_projection is not None:
            gate_logits = self.neighbor_gate_projection(target_coordinates)
            if aligned_availability is None:
                gates = _availability_masked_softmax_gates(gate_logits, availability)
                gated_neighbors = neighbors * gates[..., None]
            else:
                time_dependent_logits = gate_logits[..., None].expand(-1, -1, time_count)
                gates = _availability_masked_softmax_gates(
                    time_dependent_logits,
                    aligned_availability,
                )
                gated_neighbors = gated_neighbors * gates
        if self.neighbor_alignment is not None:
            # Gating is a time-invariant scalar per neighbor and therefore
            # commutes with a depthwise temporal FIR. Apply it first, then make
            # unavailable channels explicitly zero before alignment.
            if aligned_availability is None:
                available = (availability > 0.0).to(dtype=neighbors.dtype)
                alignment_input = gated_neighbors * available[..., None]
            else:
                alignment_input = gated_neighbors * aligned_availability.to(dtype=neighbors.dtype)
            gated_neighbors = self.neighbor_alignment(alignment_input)

        prediction_reference: torch.Tensor | None = None
        if self.prediction_reference == MASKED_ALIGNED_NEIGHBOR_MEAN_REFERENCE:
            if aligned_availability is None:
                available = availability > 0.0
                available_float = available.to(dtype=neighbors.dtype)
                available_count = available_float.sum(dim=1, keepdim=True).clamp_min(1.0)
                prediction_reference = (gated_neighbors * available_float[..., None]).sum(
                    dim=1
                ) / available_count
            else:
                available_float = aligned_availability.to(dtype=neighbors.dtype)
                available_count = available_float.sum(dim=1).clamp_min(1.0)
                prediction_reference = (gated_neighbors * available_float).sum(
                    dim=1
                ) / available_count
        elif raw_bracketing_reference is not None:
            prediction_reference = raw_bracketing_reference

        availability_channels = (
            availability[..., None].expand(-1, -1, time_count)
            if aligned_availability is None
            else aligned_availability.to(dtype=neighbors.dtype)
        )
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

    def _coarse_align_neighbors(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.coarse_sample_shifts is None:
            raise AssertionError("coarse alignment requires configured sample shifts")
        time_count = neighbors.shape[2]
        output_indices = torch.arange(time_count, device=neighbors.device)
        source_indices = output_indices[None, :] - self.coarse_sample_shifts[:, None]
        valid_samples = (source_indices >= 0) & (source_indices < time_count)
        safe_indices = source_indices.clamp(0, time_count - 1)
        aligned = torch.gather(
            neighbors,
            dim=2,
            index=safe_indices[None, :, :].expand(neighbors.shape[0], -1, -1),
        )
        aligned_availability = valid_samples[None, :, :] & (availability > 0.0)[..., None]
        return (
            aligned * aligned_availability.to(dtype=neighbors.dtype),
            aligned_availability,
        )
