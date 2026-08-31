"""Joint whole-shot interpolation on the fixed SEG C3 receiver grid."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

RECEIVER_X_COUNT = 8
RECEIVER_Y_COUNT = 68
DEFAULT_WIDTH = 32
DEFAULT_STEM_KERNEL_SIZE = 7
DEFAULT_RESIDUAL_KERNEL_SIZE = 3
DEFAULT_TEMPORAL_DILATIONS = (1, 2, 4, 8, 4, 2, 1)
DEFAULT_DISTANCE_EPSILON = 1.0e-6
SHOT_GATHER_INPUT_FEATURE_NAMES: tuple[str, ...] = (
    "inverse_distance_reference",
    "weighted_absolute_deviation",
    "source_direction_x_waveform_moment",
    "source_direction_y_waveform_moment",
    "availability_fraction",
    "weighted_source_direction_x",
    "weighted_source_direction_y",
    "target_coordinate_x",
    "target_coordinate_y",
    "receiver_coordinate_x",
    "receiver_coordinate_y",
)
"""Ordered channels presented to :attr:`ShotGatherInpainter.stem`."""

_GROUP_COUNT = 8
_SOURCE_COORDINATE_COUNT = 2
_TARGET_COORDINATE_COUNT = 2


def inverse_distance_reference(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    source_deltas: torch.Tensor,
    *,
    distance_epsilon: float = DEFAULT_DISTANCE_EPSILON,
) -> torch.Tensor:
    """Return a receiver-wise inverse-distance blend with shape ``[B, 8, 68, T]``.

    A receiver cell with no available source gather receives a zero reference.
    Available sources must have non-zero source deltas so the target shot cannot
    enter its own interpolation reference.
    """
    _validated_reference_inputs(neighbors, availability, source_deltas)
    epsilon = _positive_finite_float(distance_epsilon, "distance_epsilon")
    weights, _directions = _inverse_distance_weights(
        availability,
        source_deltas,
        distance_epsilon=epsilon,
    )
    return torch.sum(neighbors * weights[..., None], dim=1)


class FactorizedGatherResidualBlock(nn.Module):
    """Apply shared temporal and receiver-spatial depthwise filters residually."""

    def __init__(
        self,
        width: int,
        temporal_dilation: int,
        *,
        temporal_kernel_size: int = DEFAULT_RESIDUAL_KERNEL_SIZE,
        spatial_y_dilation: int = 1,
    ) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.temporal_dilation = _positive_integer(
            temporal_dilation,
            "temporal_dilation",
        )
        self.temporal_kernel_size = _odd_positive_integer(
            temporal_kernel_size,
            "temporal_kernel_size",
        )
        self.spatial_y_dilation = _positive_integer(
            spatial_y_dilation,
            "spatial_y_dilation",
        )

        self.norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        self.temporal = nn.Conv3d(
            self.width,
            self.width,
            kernel_size=(1, 1, self.temporal_kernel_size),
            padding=(0, 0, (self.temporal_kernel_size // 2) * self.temporal_dilation),
            dilation=(1, 1, self.temporal_dilation),
            groups=self.width,
        )
        self.spatial = nn.Conv3d(
            self.width,
            self.width,
            kernel_size=(3, 3, 1),
            padding=(1, self.spatial_y_dilation, 0),
            dilation=(1, self.spatial_y_dilation, 1),
            groups=self.width,
        )
        self.expand = nn.Conv3d(self.width, self.width * 2, kernel_size=1)
        self.contract = nn.Conv3d(self.width, self.width, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform ``[B, width, 8, 68, T]`` while preserving its shape."""
        features = _require_floating_tensor(features, "features")
        if features.ndim != 5:
            raise ValueError(
                "features must have shape "
                f"(batch, {self.width}, {RECEIVER_X_COUNT}, {RECEIVER_Y_COUNT}, time), "
                f"got {tuple(features.shape)}"
            )
        expected_prefix = (features.shape[0], self.width, RECEIVER_X_COUNT, RECEIVER_Y_COUNT)
        if features.shape[:4] != expected_prefix:
            raise ValueError(
                "features must have shape "
                f"(batch, {self.width}, {RECEIVER_X_COUNT}, {RECEIVER_Y_COUNT}, time), "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] == 0 or features.shape[4] == 0:
            raise ValueError("features batch and time dimensions must be non-empty")

        transformed = self.temporal(F.silu(self.norm(features)))
        transformed = self.spatial(F.silu(transformed))
        value, gate = self.expand(transformed).chunk(2, dim=1)
        return features + self.contract(F.silu(value) * torch.sigmoid(gate))


class ShotGatherInpainter(nn.Module):
    """Predict a complete ``8 x 68`` target shot gather from neighboring gathers.

    The input source count ``K`` is dynamic. Neighbor gathers are first reduced
    receiver-by-receiver to an inverse-distance reference, disagreement, and
    signed source-direction waveform moments. Deterministic receiver coordinates
    expose far-offset nonstationarity. A compact residual network then alternates
    temporal and receiver spatial filtering with weights shared over the grid.
    """

    def __init__(
        self,
        *,
        width: int = DEFAULT_WIDTH,
        temporal_dilations: Iterable[int] = DEFAULT_TEMPORAL_DILATIONS,
        spatial_y_dilations: Iterable[int] | None = None,
        stem_kernel_size: int = DEFAULT_STEM_KERNEL_SIZE,
        residual_kernel_size: int = DEFAULT_RESIDUAL_KERNEL_SIZE,
        distance_epsilon: float = DEFAULT_DISTANCE_EPSILON,
    ) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.temporal_dilations = _validated_temporal_dilations(temporal_dilations)
        self.spatial_y_dilations = _validated_spatial_y_dilations(
            spatial_y_dilations,
            block_count=len(self.temporal_dilations),
        )
        self.stem_kernel_size = _odd_positive_integer(stem_kernel_size, "stem_kernel_size")
        self.residual_kernel_size = _odd_positive_integer(
            residual_kernel_size,
            "residual_kernel_size",
        )
        self.distance_epsilon = _positive_finite_float(
            distance_epsilon,
            "distance_epsilon",
        )
        self.input_feature_names: tuple[str, ...] = SHOT_GATHER_INPUT_FEATURE_NAMES
        self.input_channels = len(self.input_feature_names)

        self.stem = nn.Conv3d(
            self.input_channels,
            self.width,
            kernel_size=(1, 1, self.stem_kernel_size),
            padding=(0, 0, self.stem_kernel_size // 2),
        )
        self.blocks = nn.ModuleList(
            FactorizedGatherResidualBlock(
                self.width,
                temporal_dilation,
                temporal_kernel_size=self.residual_kernel_size,
                spatial_y_dilation=spatial_y_dilation,
            )
            for temporal_dilation, spatial_y_dilation in zip(
                self.temporal_dilations,
                self.spatial_y_dilations,
                strict=True,
            )
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_GROUP_COUNT, self.width),
            nn.SiLU(),
            nn.Conv3d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(self.width, 1, kernel_size=1),
        )
        final_projection = self.head[-1]
        if not isinstance(final_projection, nn.Conv3d):
            raise AssertionError("shot-gather inpainter head must end with Conv3d")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        source_deltas: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return target amplitudes with shape ``[B, 8, 68, T]``."""
        batch_size, _source_count, time_count = _validated_inputs(
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
        )
        weights, directions = _inverse_distance_weights(
            availability,
            source_deltas,
            distance_epsilon=self.distance_epsilon,
        )
        reference = torch.sum(neighbors * weights[..., None], dim=1)
        weighted_centered_neighbors = (neighbors - reference[:, None]) * weights[..., None]
        disagreement = torch.sum(torch.abs(weighted_centered_neighbors), dim=1)
        directional_waveform_moments = _directional_waveform_moments(
            weighted_centered_neighbors,
            directions,
        )

        availability_fraction = availability.to(dtype=neighbors.dtype).mean(dim=1)
        weighted_direction = torch.einsum("bkxy,bkc->bcxy", weights, directions)
        target_grid = target_coordinates[:, :, None, None].expand(
            -1,
            -1,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
        )
        receiver_coordinates = _normalized_receiver_coordinates(
            dtype=neighbors.dtype,
            device=neighbors.device,
        )[None].expand(batch_size, -1, -1, -1)
        static_features = torch.cat(
            (
                availability_fraction[:, None],
                weighted_direction,
                target_grid,
                receiver_coordinates,
            ),
            dim=1,
        )
        static_time_features = static_features[..., None].expand(-1, -1, -1, -1, time_count)
        features = torch.cat(
            (
                reference[:, None],
                disagreement[:, None],
                directional_waveform_moments,
                static_time_features,
            ),
            dim=1,
        )

        hidden = self.stem(features)
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.head(hidden)[:, 0]
        if residual.shape != (batch_size, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, time_count):
            raise AssertionError("shot-gather residual head changed the target shape")
        return reference + residual


def _validated_inputs(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    source_deltas: torch.Tensor,
    target_coordinates: torch.Tensor,
) -> tuple[int, int, int]:
    batch_size, source_count, time_count = _validated_reference_inputs(
        neighbors,
        availability,
        source_deltas,
    )
    target_coordinates = _require_floating_tensor(target_coordinates, "target_coordinates")
    if target_coordinates.shape != (batch_size, _TARGET_COORDINATE_COUNT):
        raise ValueError(
            "target_coordinates must have shape "
            f"({batch_size}, {_TARGET_COORDINATE_COUNT}), got {tuple(target_coordinates.shape)}"
        )
    if target_coordinates.dtype != neighbors.dtype:
        raise TypeError(
            "target_coordinates must share the neighbors dtype, "
            f"got {target_coordinates.dtype} and {neighbors.dtype}"
        )
    if target_coordinates.device != neighbors.device:
        raise ValueError("neighbors and target_coordinates must share a device")
    if not bool(torch.isfinite(target_coordinates).all().item()):
        raise ValueError("target_coordinates must contain only finite values")
    return batch_size, source_count, time_count


def _validated_reference_inputs(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    source_deltas: torch.Tensor,
) -> tuple[int, int, int]:
    neighbors = _require_floating_tensor(neighbors, "neighbors")
    availability = _require_tensor(availability, "availability")
    source_deltas = _require_floating_tensor(source_deltas, "source_deltas")
    if neighbors.ndim != 5:
        raise ValueError(
            f"neighbors must have shape (batch, sources, 8, 68, time), got {tuple(neighbors.shape)}"
        )
    batch_size, source_count, receiver_x, receiver_y, time_count = neighbors.shape
    if (receiver_x, receiver_y) != (RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError(
            "neighbors receiver dimensions must be "
            f"({RECEIVER_X_COUNT}, {RECEIVER_Y_COUNT}), got ({receiver_x}, {receiver_y})"
        )
    if batch_size == 0 or source_count == 0 or time_count == 0:
        raise ValueError("neighbors batch, source, and time dimensions must be non-empty")
    if availability.shape != (batch_size, source_count, RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError(
            "availability must have shape "
            f"({batch_size}, {source_count}, {RECEIVER_X_COUNT}, {RECEIVER_Y_COUNT}), "
            f"got {tuple(availability.shape)}"
        )
    if availability.dtype != torch.bool:
        raise TypeError(f"availability must have dtype torch.bool, got {availability.dtype}")
    if source_deltas.shape != (batch_size, source_count, _SOURCE_COORDINATE_COUNT):
        raise ValueError(
            "source_deltas must have shape "
            f"({batch_size}, {source_count}, {_SOURCE_COORDINATE_COUNT}), "
            f"got {tuple(source_deltas.shape)}"
        )
    if source_deltas.dtype != neighbors.dtype:
        raise TypeError(
            "source_deltas must share the neighbors dtype, "
            f"got {source_deltas.dtype} and {neighbors.dtype}"
        )
    if not (neighbors.device == availability.device == source_deltas.device):
        raise ValueError("neighbors, availability, and source_deltas must share a device")
    if not bool(torch.isfinite(source_deltas).all().item()):
        raise ValueError("source_deltas must contain only finite values")
    return batch_size, source_count, time_count


def _inverse_distance_weights(
    availability: torch.Tensor,
    source_deltas: torch.Tensor,
    *,
    distance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    calculation_dtype = (
        torch.float32
        if source_deltas.dtype in (torch.float16, torch.bfloat16)
        else source_deltas.dtype
    )
    calculation_deltas = source_deltas.to(dtype=calculation_dtype)
    distance = torch.linalg.vector_norm(calculation_deltas, dim=2)
    source_is_available = availability.flatten(start_dim=2).any(dim=2)
    if bool(((distance == 0.0) & source_is_available).any().item()):
        raise ValueError("every available source gather must have a non-zero source delta")

    safe_distance = distance.clamp_min(distance_epsilon)
    log_inverse_distance = -torch.log(safe_distance)
    expanded_log_weights = log_inverse_distance[:, :, None, None].expand_as(availability)
    minimum = torch.finfo(calculation_dtype).min
    masked_log_weights = expanded_log_weights.masked_fill(~availability, minimum)
    receiver_has_source = availability.any(dim=1, keepdim=True)
    maximum = masked_log_weights.max(dim=1, keepdim=True).values
    safe_maximum = torch.where(receiver_has_source, maximum, torch.zeros_like(maximum))
    unnormalized = torch.exp(masked_log_weights - safe_maximum) * availability.to(
        dtype=calculation_dtype
    )
    denominator = unnormalized.sum(dim=1, keepdim=True)
    safe_denominator = torch.where(
        receiver_has_source,
        denominator,
        torch.ones_like(denominator),
    )
    weights = (unnormalized / safe_denominator).to(dtype=source_deltas.dtype)
    direction_denominator = torch.where(
        distance > 0.0,
        distance,
        torch.ones_like(distance),
    )
    directions = (calculation_deltas / direction_denominator[:, :, None]).to(
        dtype=source_deltas.dtype
    )
    return weights, directions


def _directional_waveform_moments(
    weighted_centered_neighbors: torch.Tensor,
    source_directions: torch.Tensor,
) -> torch.Tensor:
    """Project centered neighbor waveforms onto both signed source axes."""
    return torch.einsum(
        "bkxyt,bkc->bcxyt",
        weighted_centered_neighbors,
        source_directions,
    )


def _normalized_receiver_coordinates(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic receiver-axis coordinates with shape ``[2, 8, 68]``."""
    receiver_x = torch.linspace(
        -1.0,
        1.0,
        RECEIVER_X_COUNT,
        dtype=dtype,
        device=device,
    )
    receiver_y = torch.linspace(
        -1.0,
        1.0,
        RECEIVER_Y_COUNT,
        dtype=dtype,
        device=device,
    )
    return torch.stack(
        (
            receiver_x[:, None].expand(-1, RECEIVER_Y_COUNT),
            receiver_y[None, :].expand(RECEIVER_X_COUNT, -1),
        )
    )


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


def _validated_temporal_dilations(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("temporal_dilations must be a non-empty iterable")
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise ValueError("temporal_dilations must be a non-empty iterable") from error
    if not raw_values:
        raise ValueError("temporal_dilations must not be empty")
    return tuple(
        _positive_integer(value, f"temporal_dilations[{index}]")
        for index, value in enumerate(raw_values)
    )


def _validated_spatial_y_dilations(
    values: Iterable[int] | None,
    *,
    block_count: int,
) -> tuple[int, ...]:
    if values is None:
        return (1,) * block_count
    if isinstance(values, (str, bytes)):
        raise ValueError("spatial_y_dilations must be an iterable of positive integers")
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise ValueError("spatial_y_dilations must be an iterable of positive integers") from error
    if len(raw_values) != block_count:
        raise ValueError(
            "spatial_y_dilations length must equal temporal_dilations length "
            f"{block_count}, got {len(raw_values)}"
        )
    return tuple(
        _positive_integer(value, f"spatial_y_dilations[{index}]")
        for index, value in enumerate(raw_values)
    )


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return converted
