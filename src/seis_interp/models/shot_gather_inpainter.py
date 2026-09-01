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
DEFAULT_DISTANCE_POWER = 1.0
DEFAULT_DYNAMIC_ATTENTION_WIDTH = 8
DEFAULT_DYNAMIC_ATTENTION_KERNEL_SIZE = 5
INVERSE_DISTANCE_SOURCE_WEIGHTING = "inverse_distance"
DYNAMIC_ATTENTION_SOURCE_WEIGHTING = "dynamic_attention"
SOURCE_WEIGHTING_MODES = (
    INVERSE_DISTANCE_SOURCE_WEIGHTING,
    DYNAMIC_ATTENTION_SOURCE_WEIGHTING,
)
MOMENTS_SOURCE_FEATURE_MODE = "moments"
ORDERED_RAW_SOURCE_FEATURE_MODE = "ordered_raw"
SOURCE_FEATURE_MODES = (
    MOMENTS_SOURCE_FEATURE_MODE,
    ORDERED_RAW_SOURCE_FEATURE_MODE,
)
NO_RECEIVER_POSITION_CONDITIONING = "none"
LEARNED_FILM_RECEIVER_POSITION_CONDITIONING = "learned_film"
RECEIVER_POSITION_CONDITIONING_MODES = (
    NO_RECEIVER_POSITION_CONDITIONING,
    LEARNED_FILM_RECEIVER_POSITION_CONDITIONING,
)
DYNAMIC_ATTENTION_INPUT_FEATURE_NAMES: tuple[str, ...] = (
    "masked_neighbor_waveform",
    "normalized_source_direction_x",
    "normalized_source_direction_y",
    "normalized_source_distance",
    "target_coordinate_x",
    "target_coordinate_y",
    "receiver_coordinate_x",
    "receiver_coordinate_y",
)
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

_ORDERED_RAW_PER_SOURCE_FEATURE_NAMES = (
    "masked_raw_waveform",
    "availability",
    "normalized_direction_x",
    "normalized_direction_y",
    "normalized_distance",
)
_ORDERED_RAW_SHARED_TRAILING_FEATURE_NAMES = (
    "target_coordinate_x",
    "target_coordinate_y",
    "receiver_coordinate_x",
    "receiver_coordinate_y",
)

_GROUP_COUNT = 8
_SOURCE_COORDINATE_COUNT = 2
_TARGET_COORDINATE_COUNT = 2


def inverse_distance_reference(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    source_deltas: torch.Tensor,
    *,
    distance_epsilon: float = DEFAULT_DISTANCE_EPSILON,
    distance_power: float = DEFAULT_DISTANCE_POWER,
) -> torch.Tensor:
    """Return a receiver-wise inverse-distance blend with shape ``[B, 8, 68, T]``.

    A receiver cell with no available source gather receives a zero reference.
    Available sources must have non-zero source deltas so the target shot cannot
    enter its own interpolation reference.
    """
    _validated_reference_inputs(neighbors, availability, source_deltas)
    epsilon = _positive_finite_float(distance_epsilon, "distance_epsilon")
    power = _positive_finite_float(distance_power, "distance_power")
    weights, _directions = _inverse_distance_weights(
        availability,
        source_deltas,
        distance_epsilon=epsilon,
        distance_power=power,
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
        receiver_position_conditioning: str = NO_RECEIVER_POSITION_CONDITIONING,
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
        self.receiver_position_conditioning = _validated_receiver_position_conditioning(
            receiver_position_conditioning
        )

        self.norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        if self.receiver_position_conditioning == LEARNED_FILM_RECEIVER_POSITION_CONDITIONING:
            self.receiver_film = ReceiverPositionFiLM(self.width)
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

        normalized = self.norm(features)
        if self.receiver_position_conditioning == LEARNED_FILM_RECEIVER_POSITION_CONDITIONING:
            normalized = self.receiver_film(normalized)
        transformed = self.temporal(F.silu(normalized))
        transformed = self.spatial(F.silu(transformed))
        value, gate = self.expand(transformed).chunk(2, dim=1)
        return features + self.contract(F.silu(value) * torch.sigmoid(gate))


class ReceiverPositionFiLM(nn.Module):
    """Apply zero-initialized channel-and-cell FiLM on the fixed receiver grid."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = _validated_width(width)
        parameter_shape = (1, self.width, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 1)
        self.scale = nn.Parameter(torch.zeros(parameter_shape))
        self.shift = nn.Parameter(torch.zeros(parameter_shape))

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        """Return ``normalized * (1 + scale) + shift`` without changing shape."""
        normalized = _require_floating_tensor(normalized, "normalized")
        if normalized.ndim != 5 or normalized.shape[1:4] != (
            self.width,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
        ):
            raise ValueError(
                "normalized must have shape "
                f"(batch, {self.width}, {RECEIVER_X_COUNT}, {RECEIVER_Y_COUNT}, time), "
                f"got {tuple(normalized.shape)}"
            )
        if normalized.shape[0] == 0 or normalized.shape[4] == 0:
            raise ValueError("normalized batch and time dimensions must be non-empty")
        if normalized.dtype != self.scale.dtype:
            raise TypeError(
                "normalized dtype must match receiver FiLM parameters, "
                f"got {normalized.dtype} and {self.scale.dtype}"
            )
        if normalized.device != self.scale.device:
            raise ValueError("normalized and receiver FiLM parameters must share a device")
        return normalized * (1.0 + self.scale) + self.shift


class DynamicSourceAttention(nn.Module):
    """Predict shared per-source, receiver, and time corrections to IDW logits."""

    def __init__(
        self,
        *,
        width: int = DEFAULT_DYNAMIC_ATTENTION_WIDTH,
        temporal_kernel_size: int = DEFAULT_DYNAMIC_ATTENTION_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        self.width = _positive_integer(width, "width")
        self.temporal_kernel_size = _odd_positive_integer(
            temporal_kernel_size,
            "temporal_kernel_size",
        )
        self.input_feature_names = DYNAMIC_ATTENTION_INPUT_FEATURE_NAMES
        self.input_channels = len(self.input_feature_names)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                self.input_channels,
                self.width,
                kernel_size=(1, 1, self.temporal_kernel_size),
                padding=(0, 0, self.temporal_kernel_size // 2),
            ),
            nn.SiLU(),
            nn.Conv3d(self.width, 1, kernel_size=1),
        )
        final_projection = self.temporal[-1]
        if not isinstance(final_projection, nn.Conv3d):
            raise AssertionError("dynamic source attention must end with Conv3d")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        source_deltas: torch.Tensor,
        source_directions: torch.Tensor,
        target_coordinates: torch.Tensor,
        *,
        distance_epsilon: float,
    ) -> torch.Tensor:
        """Return logit corrections with shape ``[B, K, 8, 68, T]``."""
        epsilon = _positive_finite_float(distance_epsilon, "distance_epsilon")
        batch_size, source_count, time_count = _validated_inputs(
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
        )
        source_directions = _require_floating_tensor(
            source_directions,
            "source_directions",
        )
        if source_directions.shape != (batch_size, source_count, _SOURCE_COORDINATE_COUNT):
            raise ValueError(
                "source_directions must have shape "
                f"({batch_size}, {source_count}, {_SOURCE_COORDINATE_COUNT}), "
                f"got {tuple(source_directions.shape)}"
            )
        if source_directions.dtype != neighbors.dtype:
            raise TypeError(
                "source_directions must share the neighbors dtype, "
                f"got {source_directions.dtype} and {neighbors.dtype}"
            )
        if source_directions.device != neighbors.device:
            raise ValueError("neighbors and source_directions must share a device")
        if not bool(torch.isfinite(source_directions).all().item()):
            raise ValueError("source_directions must contain only finite values")
        first_projection = self.temporal[0]
        if not isinstance(first_projection, nn.Conv3d):
            raise AssertionError("dynamic source attention must start with Conv3d")
        if neighbors.dtype != first_projection.weight.dtype:
            raise TypeError(
                "neighbors dtype must match dynamic attention parameters, "
                f"got {neighbors.dtype} and {first_projection.weight.dtype}"
            )
        if neighbors.device != first_projection.weight.device:
            raise ValueError("neighbors and dynamic attention parameters must share a device")

        receiver_x = RECEIVER_X_COUNT
        receiver_y = RECEIVER_Y_COUNT
        availability_time = availability.to(dtype=neighbors.dtype)[..., None].expand(
            -1,
            -1,
            -1,
            -1,
            time_count,
        )
        masked_waveform = neighbors * availability_time
        direction_features = source_directions[:, :, :, None, None, None].expand(
            -1,
            -1,
            -1,
            receiver_x,
            receiver_y,
            time_count,
        )
        normalized_distance = _normalized_source_distances(
            source_deltas,
            distance_epsilon=epsilon,
        )[:, :, None, None, None, None].expand(
            -1,
            -1,
            -1,
            receiver_x,
            receiver_y,
            time_count,
        )
        target_features = target_coordinates[:, None, :, None, None, None].expand(
            -1,
            source_count,
            -1,
            receiver_x,
            receiver_y,
            time_count,
        )
        receiver_features = _normalized_receiver_coordinates(
            dtype=neighbors.dtype,
            device=neighbors.device,
        )[None, None, ..., None].expand(
            batch_size,
            source_count,
            -1,
            -1,
            -1,
            time_count,
        )
        features = torch.cat(
            (
                masked_waveform[:, :, None],
                direction_features,
                normalized_distance,
                target_features,
                receiver_features,
            ),
            dim=2,
        )
        expected_shape = (
            batch_size,
            source_count,
            self.input_channels,
            receiver_x,
            receiver_y,
            time_count,
        )
        if features.shape != expected_shape:
            raise AssertionError("dynamic attention features changed their declared schema")
        corrections = self.temporal(
            features.reshape(
                batch_size * source_count,
                self.input_channels,
                receiver_x,
                receiver_y,
                time_count,
            )
        )
        return corrections.reshape(
            batch_size,
            source_count,
            receiver_x,
            receiver_y,
            time_count,
        )


class ShotGatherInpainter(nn.Module):
    """Predict a complete ``8 x 68`` target shot gather from neighboring gathers.

    ``moments`` mode retains the original dynamic-source feature reduction.
    ``ordered_raw`` mode instead preserves every source waveform and requires a
    fixed source count. ``dynamic_attention`` source weighting learns a
    zero-initialized correction to receiver- and time-specific inverse-distance
    logits. Deterministic receiver coordinates expose far-offset nonstationarity.
    A compact residual network then alternates temporal and receiver spatial
    filtering with weights shared over the grid.
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
        distance_power: float = DEFAULT_DISTANCE_POWER,
        source_weighting: str = INVERSE_DISTANCE_SOURCE_WEIGHTING,
        dynamic_attention_width: int = DEFAULT_DYNAMIC_ATTENTION_WIDTH,
        dynamic_attention_kernel_size: int = DEFAULT_DYNAMIC_ATTENTION_KERNEL_SIZE,
        source_feature_mode: str = MOMENTS_SOURCE_FEATURE_MODE,
        source_gather_count: int | None = None,
        receiver_position_conditioning: str = NO_RECEIVER_POSITION_CONDITIONING,
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
        self.distance_power = _positive_finite_float(distance_power, "distance_power")
        self.source_weighting = _validated_source_weighting(source_weighting)
        self.dynamic_attention_width = _positive_integer(
            dynamic_attention_width,
            "dynamic_attention_width",
        )
        self.dynamic_attention_kernel_size = _odd_positive_integer(
            dynamic_attention_kernel_size,
            "dynamic_attention_kernel_size",
        )
        self.source_weighting_input_feature_names = (
            DYNAMIC_ATTENTION_INPUT_FEATURE_NAMES
            if self.source_weighting == DYNAMIC_ATTENTION_SOURCE_WEIGHTING
            else ()
        )
        self.source_feature_mode = _validated_source_feature_mode(source_feature_mode)
        self.source_gather_count = _validated_source_gather_count(
            source_gather_count,
            source_feature_mode=self.source_feature_mode,
        )
        self.receiver_position_conditioning = _validated_receiver_position_conditioning(
            receiver_position_conditioning
        )
        if self.source_feature_mode == MOMENTS_SOURCE_FEATURE_MODE:
            self.input_feature_names = SHOT_GATHER_INPUT_FEATURE_NAMES
        else:
            if self.source_gather_count is None:
                raise AssertionError("validated ordered_raw mode is missing its source count")
            self.input_feature_names = ordered_raw_input_feature_names(self.source_gather_count)
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
                receiver_position_conditioning=self.receiver_position_conditioning,
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
        if self.source_weighting == DYNAMIC_ATTENTION_SOURCE_WEIGHTING:
            self.dynamic_source_attention = DynamicSourceAttention(
                width=self.dynamic_attention_width,
                temporal_kernel_size=self.dynamic_attention_kernel_size,
            )

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        source_deltas: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return target amplitudes with shape ``[B, 8, 68, T]``."""
        batch_size, source_count, time_count = _validated_inputs(
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
        )
        if (
            self.source_feature_mode == ORDERED_RAW_SOURCE_FEATURE_MODE
            and source_count != self.source_gather_count
        ):
            raise ValueError(
                "ordered_raw neighbors source dimension must equal source_gather_count "
                f"{self.source_gather_count}, got {source_count}"
            )
        weights, directions = _inverse_distance_weights(
            availability,
            source_deltas,
            distance_epsilon=self.distance_epsilon,
            distance_power=self.distance_power,
        )
        if self.source_weighting == DYNAMIC_ATTENTION_SOURCE_WEIGHTING:
            logit_corrections = self.dynamic_source_attention(
                neighbors,
                availability,
                source_deltas,
                directions,
                target_coordinates,
                distance_epsilon=self.distance_epsilon,
            )
            weights, _directions = _inverse_distance_weights(
                availability,
                source_deltas,
                distance_epsilon=self.distance_epsilon,
                distance_power=self.distance_power,
                logit_corrections=logit_corrections,
            )
            reference = torch.sum(neighbors * weights, dim=1)
        else:
            reference = torch.sum(neighbors * weights[..., None], dim=1)
        if (
            self.source_feature_mode == MOMENTS_SOURCE_FEATURE_MODE
            and self.source_weighting == DYNAMIC_ATTENTION_SOURCE_WEIGHTING
        ):
            features = _dynamic_weighted_moment_features(
                neighbors,
                availability,
                weights,
                directions,
                target_coordinates,
                reference=reference,
            )
        elif self.source_feature_mode == MOMENTS_SOURCE_FEATURE_MODE:
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
            static_time_features = static_features[..., None].expand(
                -1,
                -1,
                -1,
                -1,
                time_count,
            )
            features = torch.cat(
                (
                    reference[:, None],
                    disagreement[:, None],
                    directional_waveform_moments,
                    static_time_features,
                ),
                dim=1,
            )
        else:
            _validate_ordered_raw_model_compatibility(self.stem, neighbors)
            features = _ordered_raw_features(
                neighbors,
                availability,
                source_deltas,
                target_coordinates,
                reference=reference,
                directions=directions,
                distance_epsilon=self.distance_epsilon,
            )

        hidden = self.stem(features)
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.head(hidden)[:, 0]
        if residual.shape != (batch_size, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, time_count):
            raise AssertionError("shot-gather residual head changed the target shape")
        return reference + residual


def ordered_raw_input_feature_names(source_gather_count: int) -> tuple[str, ...]:
    """Return the deterministic interleaved feature order for fixed ``K`` sources."""
    count = _positive_integer(source_gather_count, "source_gather_count")
    per_source = tuple(
        f"source_{source_index:03d}_{feature_name}"
        for source_index in range(count)
        for feature_name in _ORDERED_RAW_PER_SOURCE_FEATURE_NAMES
    )
    return (
        "inverse_distance_reference",
        *per_source,
        *_ORDERED_RAW_SHARED_TRAILING_FEATURE_NAMES,
    )


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
    distance_power: float,
    logit_corrections: torch.Tensor | None = None,
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
    log_inverse_distance = -distance_power * torch.log(safe_distance)
    if logit_corrections is None:
        expanded_availability = availability
        expanded_log_weights = log_inverse_distance[:, :, None, None].expand_as(availability)
    else:
        logit_corrections = _require_floating_tensor(
            logit_corrections,
            "logit_corrections",
        )
        expected_correction_prefix = (*availability.shape,)
        if (
            logit_corrections.ndim != 5
            or logit_corrections.shape[:4] != expected_correction_prefix
            or logit_corrections.shape[4] == 0
        ):
            raise ValueError(
                "logit_corrections must have shape "
                f"({', '.join(str(value) for value in availability.shape)}, time), "
                f"got {tuple(logit_corrections.shape)}"
            )
        if logit_corrections.device != source_deltas.device:
            raise ValueError("source_deltas and logit_corrections must share a device")
        if not bool(torch.isfinite(logit_corrections).all().item()):
            raise ValueError("logit_corrections must contain only finite values")
        expanded_availability = availability[..., None].expand_as(logit_corrections)
        expanded_log_weights = log_inverse_distance[:, :, None, None, None].expand_as(
            logit_corrections
        ) + logit_corrections.to(dtype=calculation_dtype)
    minimum = torch.finfo(calculation_dtype).min
    masked_log_weights = expanded_log_weights.masked_fill(~expanded_availability, minimum)
    receiver_has_source = expanded_availability.any(dim=1, keepdim=True)
    maximum = masked_log_weights.max(dim=1, keepdim=True).values
    safe_maximum = torch.where(receiver_has_source, maximum, torch.zeros_like(maximum))
    unnormalized = torch.exp(masked_log_weights - safe_maximum) * expanded_availability.to(
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


def _dynamic_weighted_moment_features(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    weights: torch.Tensor,
    source_directions: torch.Tensor,
    target_coordinates: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Build the original moment schema from receiver- and time-varying weights."""
    batch_size, _source_count, _receiver_x, _receiver_y, time_count = neighbors.shape
    weighted_centered_neighbors = (neighbors - reference[:, None]) * weights
    disagreement = torch.sum(torch.abs(weighted_centered_neighbors), dim=1)
    directional_waveform_moments = _directional_waveform_moments(
        weighted_centered_neighbors,
        source_directions,
    )
    availability_fraction = availability.to(dtype=neighbors.dtype).mean(dim=1)
    availability_time = availability_fraction[:, None, ..., None].expand(
        -1,
        -1,
        -1,
        -1,
        time_count,
    )
    weighted_direction = torch.einsum("bkxyt,bkc->bcxyt", weights, source_directions)
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
    coordinate_time = torch.cat((target_grid, receiver_coordinates), dim=1)[..., None].expand(
        -1,
        -1,
        -1,
        -1,
        time_count,
    )
    features = torch.cat(
        (
            reference[:, None],
            disagreement[:, None],
            directional_waveform_moments,
            availability_time,
            weighted_direction,
            coordinate_time,
        ),
        dim=1,
    )
    expected_shape = (
        batch_size,
        len(SHOT_GATHER_INPUT_FEATURE_NAMES),
        RECEIVER_X_COUNT,
        RECEIVER_Y_COUNT,
        time_count,
    )
    if features.shape != expected_shape:
        raise AssertionError("dynamic moment features changed the declared stem schema")
    return features


def _ordered_raw_features(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    source_deltas: torch.Tensor,
    target_coordinates: torch.Tensor,
    *,
    reference: torch.Tensor,
    directions: torch.Tensor,
    distance_epsilon: float,
) -> torch.Tensor:
    """Preserve each ordered source as one waveform plus four descriptor channels."""
    batch_size, source_count, receiver_x, receiver_y, time_count = neighbors.shape
    availability_feature = availability.to(dtype=neighbors.dtype)[..., None].expand(
        -1,
        -1,
        -1,
        -1,
        time_count,
    )
    masked_neighbors = neighbors * availability_feature
    direction_x = directions[:, :, 0, None, None, None].expand(
        -1,
        -1,
        receiver_x,
        receiver_y,
        time_count,
    )
    direction_y = directions[:, :, 1, None, None, None].expand_as(direction_x)
    normalized_distance = _normalized_source_distances(
        source_deltas,
        distance_epsilon=distance_epsilon,
    )[:, :, None, None, None].expand_as(direction_x)
    per_source_features = torch.stack(
        (
            masked_neighbors,
            availability_feature,
            direction_x,
            direction_y,
            normalized_distance,
        ),
        dim=2,
    ).flatten(start_dim=1, end_dim=2)

    target_grid = target_coordinates[:, :, None, None].expand(
        -1,
        -1,
        receiver_x,
        receiver_y,
    )
    receiver_coordinates = _normalized_receiver_coordinates(
        dtype=neighbors.dtype,
        device=neighbors.device,
    )[None].expand(batch_size, -1, -1, -1)
    shared_static_features = torch.cat((target_grid, receiver_coordinates), dim=1)
    shared_static_time_features = shared_static_features[..., None].expand(
        -1,
        -1,
        -1,
        -1,
        time_count,
    )
    features = torch.cat(
        (
            reference[:, None],
            per_source_features,
            shared_static_time_features,
        ),
        dim=1,
    )
    expected_channels = (
        1
        + source_count * len(_ORDERED_RAW_PER_SOURCE_FEATURE_NAMES)
        + len(_ORDERED_RAW_SHARED_TRAILING_FEATURE_NAMES)
    )
    if features.shape != (
        batch_size,
        expected_channels,
        receiver_x,
        receiver_y,
        time_count,
    ):
        raise AssertionError("ordered_raw feature construction changed its declared schema")
    return features


def _normalized_source_distances(
    source_deltas: torch.Tensor,
    *,
    distance_epsilon: float,
) -> torch.Tensor:
    """Scale each target's Euclidean source distances by its farthest source."""
    calculation_dtype = (
        torch.float32
        if source_deltas.dtype in (torch.float16, torch.bfloat16)
        else source_deltas.dtype
    )
    distances = torch.linalg.vector_norm(source_deltas.to(dtype=calculation_dtype), dim=2)
    scale = distances.amax(dim=1, keepdim=True).clamp_min(distance_epsilon)
    return (distances / scale).to(dtype=source_deltas.dtype)


def _validate_ordered_raw_model_compatibility(
    stem: nn.Conv3d,
    neighbors: torch.Tensor,
) -> None:
    if neighbors.dtype != stem.weight.dtype:
        raise TypeError(
            "ordered_raw neighbors dtype must match the model dtype, "
            f"got {neighbors.dtype} and {stem.weight.dtype}"
        )
    if neighbors.device != stem.weight.device:
        raise ValueError("ordered_raw neighbors and model parameters must share a device")


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


def _validated_source_weighting(value: object) -> str:
    if not isinstance(value, str) or value not in SOURCE_WEIGHTING_MODES:
        raise ValueError(f"source_weighting must be one of {SOURCE_WEIGHTING_MODES}, got {value!r}")
    return value


def _validated_source_feature_mode(value: object) -> str:
    if not isinstance(value, str) or value not in SOURCE_FEATURE_MODES:
        raise ValueError(
            f"source_feature_mode must be one of {SOURCE_FEATURE_MODES}, got {value!r}"
        )
    return value


def _validated_source_gather_count(
    value: object,
    *,
    source_feature_mode: str,
) -> int | None:
    if source_feature_mode == MOMENTS_SOURCE_FEATURE_MODE:
        if value is not None:
            raise ValueError("source_gather_count is only valid for ordered_raw mode")
        return None
    if value is None:
        raise ValueError("source_gather_count is required for ordered_raw mode")
    return _positive_integer(value, "source_gather_count")


def _validated_receiver_position_conditioning(value: object) -> str:
    if not isinstance(value, str) or value not in RECEIVER_POSITION_CONDITIONING_MODES:
        raise ValueError(
            "receiver_position_conditioning must be one of "
            f"{RECEIVER_POSITION_CONDITIONING_MODES}, got {value!r}"
        )
    return value


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
