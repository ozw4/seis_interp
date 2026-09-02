"""Shared temporal neighbor encoder with geometry-conditioned masked attention."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral, Real

import torch
from torch import nn

from seis_interp.models.neighbor_trace_inpainter import (
    DEFAULT_RESIDUAL_KERNEL_SIZE,
    DEFAULT_STEM_KERNEL_SIZE,
    TEMPORAL_DILATIONS,
    TemporalResidualBlock,
)

MODEL_NAME = "shared_offset_attention_inpainter"
DEFAULT_NEIGHBOR_FEATURE_WIDTH = 8
DEFAULT_ATTENTION_WIDTH = 16
DEFAULT_COARSE_SHIFT_SAMPLES_PER_RELATIVE_RECEIVER_Y_INDEX = 3
DEFAULT_ATTENTION_GEOMETRY_PRIOR_SCALE = 1.0
DEFAULT_TARGET_COORDINATE_COUNT = 4
DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE = "distance_prior_shifted_neighbor_mean"
OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING = "offset_target_time_masked_softmax"
OFFSET_ORDER_AXES = (
    "relative_receiver_x_index",
    "source_x_line_index",
    "source_y_half_shot_index",
    "relative_receiver_y_index",
)

_GROUP_COUNT = 8
_OFFSET_COORDINATE_COUNT = len(OFFSET_ORDER_AXES)


class SharedOffsetAttentionInpainter(nn.Module):
    """Predict a trace using shared neighbor features and ``O(B*K*T)`` attention.

    ``neighbor_offsets`` is the exact geometry order corresponding to the input
    neighbor axis. Each offset contains ``(relative_rx, source_x_line,
    source_y_half_shot, relative_ry)`` integer indices. Before encoding, every
    neighbor is shifted by ``samples_per_relative_ry * relative_ry`` using zero
    padding. No circular samples are introduced.

    At initialization, the residual decoder is exactly zero and the prediction
    is the masked, distance-prior-weighted mean of the shifted raw neighbors.
    The shared encoder, offset FiLM, target/time query, and content score can
    then learn a correction without allocating a neighbor-by-neighbor matrix.
    """

    model_name = MODEL_NAME

    def __init__(
        self,
        neighbor_offsets: Iterable[Iterable[int]],
        width: int = 128,
        neighbor_feature_width: int = DEFAULT_NEIGHBOR_FEATURE_WIDTH,
        attention_width: int = DEFAULT_ATTENTION_WIDTH,
        target_coordinate_count: int = DEFAULT_TARGET_COORDINATE_COUNT,
        stem_kernel_size: int = DEFAULT_STEM_KERNEL_SIZE,
        residual_kernel_size: int = DEFAULT_RESIDUAL_KERNEL_SIZE,
        temporal_dilations: Iterable[int] = TEMPORAL_DILATIONS,
        coarse_shift_samples_per_relative_receiver_y_index: int = (
            DEFAULT_COARSE_SHIFT_SAMPLES_PER_RELATIVE_RECEIVER_Y_INDEX
        ),
        attention_geometry_prior_scale: float = DEFAULT_ATTENTION_GEOMETRY_PRIOR_SCALE,
    ) -> None:
        super().__init__()
        offsets = _validated_neighbor_offsets(neighbor_offsets)
        self.neighbor_count = len(offsets)
        self.width = _validated_width(width)
        self.neighbor_feature_width = _positive_integer(
            neighbor_feature_width,
            "neighbor_feature_width",
        )
        self.attention_width = _positive_integer(attention_width, "attention_width")
        self.target_coordinate_count = _positive_integer(
            target_coordinate_count,
            "target_coordinate_count",
        )
        self.stem_kernel_size = _odd_positive_integer(stem_kernel_size, "stem_kernel_size")
        self.residual_kernel_size = _odd_positive_integer(
            residual_kernel_size,
            "residual_kernel_size",
        )
        self.temporal_dilations = _validated_temporal_dilations(temporal_dilations)
        self.dilations = self.temporal_dilations
        self.coarse_shift_samples_per_relative_receiver_y_index = _nonnegative_integer(
            coarse_shift_samples_per_relative_receiver_y_index,
            "coarse_shift_samples_per_relative_receiver_y_index",
        )
        self.attention_geometry_prior_scale = _nonnegative_finite_float(
            attention_geometry_prior_scale,
            "attention_geometry_prior_scale",
        )
        self.prediction_reference = DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE
        self.coordinate_conditioning = "film"
        self.neighbor_gating = OFFSET_TARGET_TIME_MASKED_SOFTMAX_GATING
        self.input_channels = self.neighbor_feature_width + self.target_coordinate_count + 2

        offset_tensor = torch.tensor(offsets, dtype=torch.int64)
        offset_scale = offset_tensor.abs().amax(dim=0).clamp_min(1).to(dtype=torch.float32)
        normalized_offsets = offset_tensor.to(dtype=torch.float32) / offset_scale
        distance_squared = (
            offset_tensor[:, 0].square()
            + 16 * offset_tensor[:, 1].square()
            + offset_tensor[:, 2].square()
            + offset_tensor[:, 3].square()
        )
        shifts = offset_tensor[:, 3] * self.coarse_shift_samples_per_relative_receiver_y_index
        self.register_buffer("neighbor_offsets", offset_tensor)
        self.register_buffer("normalized_neighbor_offsets", normalized_offsets)
        self.register_buffer(
            "attention_geometry_prior",
            -self.attention_geometry_prior_scale * distance_squared.to(dtype=torch.float32),
        )
        self.register_buffer("coarse_sample_shifts", shifts)

        self.shared_encoder = nn.Sequential(
            nn.Conv1d(
                1,
                self.neighbor_feature_width,
                kernel_size=self.stem_kernel_size,
                padding=self.stem_kernel_size // 2,
            ),
            nn.SiLU(),
            nn.Conv1d(
                self.neighbor_feature_width,
                self.neighbor_feature_width,
                kernel_size=self.residual_kernel_size,
                padding=self.residual_kernel_size // 2,
                groups=self.neighbor_feature_width,
            ),
            nn.SiLU(),
        )
        self.offset_feature_modulation = nn.Linear(
            _OFFSET_COORDINATE_COUNT,
            self.neighbor_feature_width * 2,
        )
        nn.init.zeros_(self.offset_feature_modulation.weight)
        nn.init.zeros_(self.offset_feature_modulation.bias)

        self.offset_key_projection = nn.Linear(
            _OFFSET_COORDINATE_COUNT,
            self.attention_width,
            bias=False,
        )
        nn.init.zeros_(self.offset_key_projection.weight)
        self.target_query_projection = nn.Linear(
            self.target_coordinate_count,
            self.attention_width,
        )
        self.time_query_projection = nn.Linear(2, self.attention_width, bias=False)
        self.content_score_projection = nn.Conv1d(
            self.neighbor_feature_width,
            1,
            kernel_size=1,
        )
        nn.init.zeros_(self.content_score_projection.weight)
        nn.init.zeros_(self.content_score_projection.bias)

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
        self.coordinate_modulations = nn.ModuleList()
        for _ in self.temporal_dilations:
            projection = nn.Linear(self.target_coordinate_count, self.width * 2)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.coordinate_modulations.append(projection)
        self.head = nn.Sequential(
            nn.GroupNorm(_GROUP_COUNT, self.width),
            nn.SiLU(),
            nn.Conv1d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.width, 1, kernel_size=1),
        )
        final_projection = self.head[-1]
        if not isinstance(final_projection, nn.Conv1d):
            raise AssertionError("shared offset attention head must end with Conv1d")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted amplitudes with shape ``[batch, time]``."""
        availability_float = self._validated_inputs(
            neighbors,
            availability,
            target_coordinates,
        )
        aligned, aligned_availability = self._coarse_align(neighbors, availability_float)
        encoded = self._encode_neighbors(aligned)
        attention = self._masked_attention_weights(
            encoded,
            aligned_availability,
            target_coordinates,
        )
        prediction_reference = (attention * aligned).sum(dim=1)
        fused = torch.einsum("bkt,bkft->bft", attention, encoded)

        batch_size = encoded.shape[0]
        time_count = encoded.shape[3]
        time = _normalized_time(
            time_count,
            device=neighbors.device,
            dtype=neighbors.dtype,
        )
        geometry_channels = target_coordinates[..., None].expand(-1, -1, time_count)
        time_channel = time.view(1, 1, time_count).expand(batch_size, 1, -1)
        availability_fraction = (
            (availability_float > 0.0)
            .to(dtype=neighbors.dtype)
            .mean(dim=1, keepdim=True)[..., None]
            .expand(-1, -1, time_count)
        )
        decoder_input = torch.cat(
            (fused, geometry_channels, time_channel, availability_fraction),
            dim=1,
        )
        hidden = self.stem(decoder_input)
        for block, projection in zip(
            self.blocks,
            self.coordinate_modulations,
            strict=True,
        ):
            hidden = block(hidden, projection(target_coordinates))
        residual = self.head(hidden)[:, 0]
        return prediction_reference + residual

    def _validated_inputs(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        neighbors = _require_floating_tensor(neighbors, "neighbors")
        availability = _require_tensor(availability, "availability")
        target_coordinates = _require_floating_tensor(
            target_coordinates,
            "target_coordinates",
        )
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
        expected_target_shape = (batch_size, self.target_coordinate_count)
        if target_coordinates.shape != expected_target_shape:
            raise ValueError(
                f"target_coordinates must have shape {expected_target_shape}, "
                f"got {tuple(target_coordinates.shape)}"
            )
        if not (neighbors.device == availability.device == target_coordinates.device):
            raise ValueError("neighbors, availability, and target_coordinates must share a device")
        if target_coordinates.dtype != neighbors.dtype:
            raise TypeError(
                "target_coordinates must share the neighbors dtype, "
                f"got {target_coordinates.dtype} and {neighbors.dtype}"
            )
        if availability.dtype == torch.bool:
            return availability.to(dtype=neighbors.dtype)
        if not availability.is_floating_point():
            raise TypeError(
                "availability must have a boolean or floating-point dtype, "
                f"got {availability.dtype}"
            )
        if availability.dtype != neighbors.dtype:
            raise TypeError(
                "floating-point availability must share the neighbors dtype, "
                f"got {availability.dtype} and {neighbors.dtype}"
            )
        return availability

    def _coarse_align(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _encode_neighbors(self, aligned: torch.Tensor) -> torch.Tensor:
        batch_size, neighbor_count, time_count = aligned.shape
        encoded = self.shared_encoder(aligned.reshape(batch_size * neighbor_count, 1, time_count))
        encoded = encoded.reshape(
            batch_size,
            neighbor_count,
            self.neighbor_feature_width,
            time_count,
        )
        modulation = self.offset_feature_modulation(
            self.normalized_neighbor_offsets.to(dtype=aligned.dtype)
        )
        scale_delta, shift = modulation.chunk(2, dim=1)
        return encoded * (1.0 + scale_delta[None, :, :, None]) + shift[None, :, :, None]

    def _masked_attention_weights(
        self,
        encoded: torch.Tensor,
        aligned_availability: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, neighbor_count, _, time_count = encoded.shape
        time = _normalized_time(
            time_count,
            device=encoded.device,
            dtype=target_coordinates.dtype,
        )
        time_features = torch.stack((time, time.square()), dim=1)
        query = self.target_query_projection(target_coordinates)[:, None, :]
        query = query + self.time_query_projection(time_features)[None, :, :]
        key = self.offset_key_projection(
            self.normalized_neighbor_offsets.to(dtype=target_coordinates.dtype)
        )
        query_key_logits = torch.einsum("bta,ka->bkt", query, key) / math.sqrt(self.attention_width)
        content_logits = self.content_score_projection(
            encoded.reshape(
                batch_size * neighbor_count,
                self.neighbor_feature_width,
                time_count,
            )
        ).reshape(batch_size, neighbor_count, time_count)
        logits = (
            query_key_logits
            + content_logits
            + self.attention_geometry_prior.to(dtype=query_key_logits.dtype)[None, :, None]
        )
        return _masked_softmax(logits, aligned_availability)


def _masked_softmax(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    minimum = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~available, minimum)
    row_maximum = masked_logits.max(dim=1, keepdim=True).values
    shifted = (masked_logits - row_maximum).masked_fill(~available, 0.0)
    unnormalized = shifted.exp() * available.to(dtype=logits.dtype)
    denominator = unnormalized.sum(dim=1, keepdim=True).clamp_min(1.0)
    return unnormalized / denominator


def _normalized_time(
    time_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, time_count, device=device, dtype=dtype)


def _validated_neighbor_offsets(
    value: Iterable[Iterable[int]],
) -> tuple[tuple[int, int, int, int], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("neighbor_offsets must be a non-empty iterable of four-integer offsets")
    try:
        raw_offsets = tuple(tuple(offset) for offset in value)
    except TypeError as error:
        raise ValueError(
            "neighbor_offsets must be a non-empty iterable of four-integer offsets"
        ) from error
    if not raw_offsets:
        raise ValueError("neighbor_offsets must not be empty")
    offsets: list[tuple[int, int, int, int]] = []
    for index, offset in enumerate(raw_offsets):
        if len(offset) != _OFFSET_COORDINATE_COUNT:
            raise ValueError(
                f"neighbor_offsets[{index}] must contain {_OFFSET_COORDINATE_COUNT} integers"
            )
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in offset):
            raise ValueError(f"neighbor_offsets[{index}] must contain integers")
        converted = tuple(int(item) for item in offset)
        offsets.append(converted)
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


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
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


def _validated_temporal_dilations(value: Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("temporal_dilations must be a non-empty iterable")
    try:
        raw = tuple(value)
    except TypeError as error:
        raise ValueError("temporal_dilations must be a non-empty iterable") from error
    if not raw:
        raise ValueError("temporal_dilations must not be empty")
    return tuple(
        _positive_integer(item, f"temporal_dilations[{index}]") for index, item in enumerate(raw)
    )


def _nonnegative_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return converted
