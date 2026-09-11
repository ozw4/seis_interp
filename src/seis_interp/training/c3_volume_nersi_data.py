"""Build profile-wise NeRSI inputs from a leakage-safe C3 volume."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.training.amplitude_scaling import normalize_by_global_rms

PROFILE_COORDINATE_ORDER = (
    "source_line",
    "shot_in_line",
    "relative_receiver_x",
)
PROFILE_AXIS_ORDER = ("time", "relative_receiver_y")
ProfileCoordinateBounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class C3VolumeNersiData:
    """Profile coordinates, masked targets, and observed-only amplitude scale.

    Profiles and coordinates use C-order keys over ``(source_line,
    shot_in_line, relative_receiver_x)``.  ``observed_trace_mask`` stays at
    trace resolution so the trainer can select complete time traces.
    """

    normalized_coordinates: np.ndarray
    normalized_profiles: np.ndarray
    observed_trace_mask: np.ndarray
    training_profile_indices: np.ndarray
    amplitude_scale: float
    spatial_shape: tuple[int, int, int, int]
    profile_shape: tuple[int, int]

    @property
    def coordinate_bounds(self) -> ProfileCoordinateBounds:
        """Return the mask-independent local-index bounds for the full domain."""
        return profile_coordinate_bounds(self.spatial_shape)


def build_c3_volume_nersi_data(
    observed: ObservedC3Volume,
    *,
    amplitude_scale: float,
) -> C3VolumeNersiData:
    """Convert one observed-only volume into stable NeRSI profile arrays.

    ``amplitude_scale`` is the shared observed-only global RMS computed by the
    caller. Evaluation-target storage is discarded before targets are built.
    """
    values, observed_mask, _ = _validated_observed_volume(observed)
    scale = _positive_finite_float(amplitude_scale, "amplitude_scale")
    time_count, source_lines, shots, receiver_x, receiver_y = values.shape
    spatial_shape = (source_lines, shots, receiver_x, receiver_y)
    profile_count = source_lines * shots * receiver_x

    coordinates = _normalized_profile_coordinates((source_lines, shots, receiver_x))
    profile_mask = np.array(
        observed_mask.reshape(profile_count, receiver_y),
        dtype=np.bool_,
        order="C",
        copy=True,
    )
    training_indices = np.ascontiguousarray(
        np.flatnonzero(np.any(profile_mask, axis=1)), dtype=np.int64
    )

    normalized_dtype = np.result_type(values.dtype, np.float32)
    normalized_values = np.zeros(values.shape, dtype=normalized_dtype)
    normalized_values[:, observed_mask] = normalize_by_global_rms(
        values[:, observed_mask],
        scale,
    )
    normalized_profiles = volume_to_nersi_profiles(normalized_values)

    return C3VolumeNersiData(
        normalized_coordinates=coordinates,
        normalized_profiles=normalized_profiles,
        observed_trace_mask=profile_mask,
        training_profile_indices=training_indices,
        amplitude_scale=scale,
        spatial_shape=spatial_shape,
        profile_shape=(time_count, receiver_y),
    )


def volume_to_nersi_profiles(values: np.ndarray) -> np.ndarray:
    """Return ``(N, 1, T, Ry)`` profiles from a ``(T, Sx, Sy, Rx, Ry)`` volume."""
    _validate_real_array(values, name="values", dimensions=5)
    time_count, source_lines, shots, receiver_x, receiver_y = values.shape
    profile_count = source_lines * shots * receiver_x
    profiles = values.transpose(1, 2, 3, 0, 4).reshape(profile_count, 1, time_count, receiver_y)
    return np.ascontiguousarray(profiles)


def nersi_profiles_to_volume(
    profiles: np.ndarray,
    spatial_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Invert :func:`volume_to_nersi_profiles` without changing values or dtype."""
    _validate_real_array(profiles, name="profiles", dimensions=4)
    source_lines, shots, receiver_x, receiver_y = _validated_spatial_shape(spatial_shape)
    profile_count = source_lines * shots * receiver_x
    if profiles.shape[0] != profile_count:
        raise ValueError("profile count must equal Sx * Sy * Rx")
    if profiles.shape[1] != 1:
        raise ValueError("profiles must have exactly one amplitude channel")
    if profiles.shape[3] != receiver_y:
        raise ValueError("profile receiver-y size must match spatial_shape")

    time_count = profiles.shape[2]
    volume = profiles.reshape(
        source_lines,
        shots,
        receiver_x,
        time_count,
        receiver_y,
    ).transpose(3, 0, 1, 2, 4)
    return np.ascontiguousarray(volume)


def _validated_observed_volume(
    observed: ObservedC3Volume,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(observed, ObservedC3Volume):
        raise ValueError("observed must be an ObservedC3Volume")
    values = observed.values
    _validate_real_array_shape(values, name="observed values", dimensions=5)
    spatial_shape = values.shape[1:]
    observed_mask = _validated_trace_mask(
        observed.observed_trace_mask,
        name="observed_trace_mask",
        spatial_shape=spatial_shape,
    )
    target_mask = _validated_trace_mask(
        observed.evaluation_target_trace_mask,
        name="evaluation_target_trace_mask",
        spatial_shape=spatial_shape,
    )
    if np.any(observed_mask & target_mask) or not np.all(observed_mask | target_mask):
        raise ValueError(
            "observed and evaluation target trace masks must disjointly cover the volume"
        )
    if not np.any(observed_mask):
        raise ValueError("volume must contain at least one observed trace")
    if not np.all(np.isfinite(values[:, observed_mask])):
        raise ValueError("observed values must contain only finite amplitudes")
    return values, observed_mask, target_mask


def _validated_trace_mask(
    mask: np.ndarray,
    *,
    name: str,
    spatial_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != spatial_shape:
        raise ValueError(f"{name} must be boolean and match the volume spatial shape")
    return mask


def _validate_real_array(values: np.ndarray, *, name: str, dimensions: int) -> None:
    _validate_real_array_shape(values, name=name, dimensions=dimensions)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def _validate_real_array_shape(values: np.ndarray, *, name: str, dimensions: int) -> None:
    if not isinstance(values, np.ndarray) or values.ndim != dimensions or not values.size:
        raise ValueError(f"{name} must be a nonempty {dimensions}-dimensional NumPy array")
    if values.dtype.kind not in "fiu" or values.dtype.kind == "b":
        raise ValueError(f"{name} must contain real numeric values")


def _validated_spatial_shape(spatial_shape: Sequence[object]) -> tuple[int, int, int, int]:
    if isinstance(spatial_shape, (str, bytes)) or not isinstance(spatial_shape, Sequence):
        raise ValueError("spatial_shape must contain four positive integers")
    if len(spatial_shape) != 4 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in spatial_shape
    ):
        raise ValueError("spatial_shape must contain four positive integers")
    return tuple(int(value) for value in spatial_shape)  # type: ignore[return-value]


def _normalized_profile_coordinates(
    key_shape: tuple[int, int, int],
) -> np.ndarray:
    bounds = _coordinate_bounds_from_key_shape(key_shape)
    axes = tuple(
        np.zeros(length, dtype=np.float32)
        if lower == upper
        else (np.arange(length, dtype=np.float32) - np.float32(lower)) / np.float32(upper - lower)
        for length, (lower, upper) in zip(key_shape, bounds, strict=True)
    )
    grids = np.meshgrid(*axes, indexing="ij")
    return np.ascontiguousarray(np.stack(grids, axis=-1).reshape(-1, 3), dtype=np.float32)


def profile_coordinate_bounds(
    spatial_shape: Sequence[object],
) -> ProfileCoordinateBounds:
    """Return full-domain local-index bounds in profile-coordinate order."""
    source_lines, shots, receiver_x, _ = _validated_spatial_shape(spatial_shape)
    return _coordinate_bounds_from_key_shape((source_lines, shots, receiver_x))


def _coordinate_bounds_from_key_shape(
    key_shape: tuple[int, int, int],
) -> ProfileCoordinateBounds:
    return tuple((0, length - 1) for length in key_shape)  # type: ignore[return-value]


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be positive and finite")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted
