"""Build profile-wise NeRSI inputs from a leakage-safe C3 volume."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.processing.trace_time_alignment import (
    align_receiver_y_time,
    validate_time_alignment,
)
from seis_interp.training.amplitude_scaling import normalize_by_global_rms

PROFILE_COORDINATE_ORDER = (
    "source_line",
    "shot_in_line",
    "relative_receiver_x",
)
PROFILE_AXIS_ORDER = ("time", "relative_receiver_y")
SUPPORTED_PROFILE_AXES = ("relative_receiver_y", "shot_in_line")
SPATIAL_AXIS_ORDER = (
    "source_line",
    "shot_in_line",
    "relative_receiver_x",
    "relative_receiver_y",
)
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
    profile_axis: str = "relative_receiver_y"
    trace_amplitude_scale: np.ndarray | None = None
    time_alignment: dict[str, object] | None = None

    @property
    def coordinate_bounds(self) -> ProfileCoordinateBounds:
        """Return the mask-independent local-index bounds for the full domain."""
        return profile_coordinate_bounds(self.spatial_shape, profile_axis=self.profile_axis)

    @property
    def coordinate_order(self) -> tuple[str, str, str]:
        """Return the three spatial axes that identify one generated profile."""
        return profile_coordinate_order(self.profile_axis)

    @property
    def profile_axis_order(self) -> tuple[str, str]:
        """Return the time and generated spatial axis order."""
        return ("time", self.profile_axis)


def build_c3_volume_nersi_data(
    observed: ObservedC3Volume,
    *,
    amplitude_scale: float,
    trace_amplitude_scale: np.ndarray | None = None,
    time_alignment: dict[str, object] | None = None,
    profile_axis: str = "relative_receiver_y",
) -> C3VolumeNersiData:
    """Convert one observed-only volume into stable NeRSI profile arrays.

    ``amplitude_scale`` is the shared observed-only global RMS computed by the
    caller. With ``trace_amplitude_scale``, training instead uses each O trace's
    physical RMS (divisor 1 for zero energy); global RMS remains a reference.
    Evaluation-target storage is discarded before targets are built.
    """
    values, observed_mask, _ = _validated_observed_volume(observed)
    axis = validate_profile_axis(profile_axis)
    if time_alignment is not None and axis != "relative_receiver_y":
        raise ValueError("time_alignment requires relative_receiver_y profiles")
    scale = _positive_finite_float(amplitude_scale, "amplitude_scale")
    time_count, source_lines, shots, receiver_x, receiver_y = values.shape
    spatial_shape = (source_lines, shots, receiver_x, receiver_y)
    coordinate_shape = _profile_coordinate_shape(spatial_shape, axis)

    coordinates = _normalized_profile_coordinates(coordinate_shape)
    profile_mask = np.array(
        trace_mask_to_nersi_profiles(observed_mask, profile_axis=axis),
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
    if trace_amplitude_scale is not None:
        validate_trace_amplitude_scale(trace_amplitude_scale, spatial_shape)
        divisors = trace_amplitude_scale[observed_mask]
        normalized_values[:, observed_mask] = values[:, observed_mask] / np.where(
            divisors > 0, divisors, 1.0
        )
    alignment = None if time_alignment is None else validate_time_alignment(time_alignment)
    if alignment is not None:
        normalized_values = align_receiver_y_time(normalized_values, alignment)
    normalized_profiles = volume_to_nersi_profiles(normalized_values, profile_axis=axis)

    return C3VolumeNersiData(
        normalized_coordinates=coordinates,
        normalized_profiles=normalized_profiles,
        observed_trace_mask=profile_mask,
        training_profile_indices=training_indices,
        amplitude_scale=scale,
        spatial_shape=spatial_shape,
        profile_shape=(normalized_values.shape[0], spatial_shape[SPATIAL_AXIS_ORDER.index(axis)]),
        profile_axis=axis,
        trace_amplitude_scale=(
            None if trace_amplitude_scale is None else trace_amplitude_scale.copy()
        ),
        time_alignment=alignment,
    )


def validate_trace_amplitude_scale(value: np.ndarray, spatial_shape: tuple[int, ...]) -> None:
    """Validate an O-fitted, physical-unit trace RMS field including zero-energy traces."""
    if (
        not isinstance(value, np.ndarray)
        or value.shape != spatial_shape
        or value.dtype != np.float64
        or not np.isfinite(value).all()
        or np.any(value < 0)
    ):
        raise ValueError("trace_amplitude_scale must be finite nonnegative spatial float64 RMS")


def volume_to_nersi_profiles(
    values: np.ndarray,
    *,
    profile_axis: str = "relative_receiver_y",
) -> np.ndarray:
    """Return ``(N, 1, T, Ry)`` profiles from a ``(T, Sx, Sy, Rx, Ry)`` volume."""
    _validate_real_array(values, name="values", dimensions=5)
    axis = validate_profile_axis(profile_axis)
    time_count = values.shape[0]
    spatial_shape = tuple(values.shape[1:])
    profile_spatial_index = SPATIAL_AXIS_ORDER.index(axis)
    coordinate_indices = tuple(i for i in range(4) if i != profile_spatial_index)
    coordinate_shape = tuple(spatial_shape[i] for i in coordinate_indices)
    profile_count = math.prod(coordinate_shape)
    transpose_axes = tuple(i + 1 for i in coordinate_indices) + (0, profile_spatial_index + 1)
    profiles = values.transpose(transpose_axes).reshape(
        profile_count, 1, time_count, spatial_shape[profile_spatial_index]
    )
    return np.ascontiguousarray(profiles)


def nersi_profiles_to_volume(
    profiles: np.ndarray,
    spatial_shape: tuple[int, int, int, int],
    *,
    profile_axis: str = "relative_receiver_y",
) -> np.ndarray:
    """Invert :func:`volume_to_nersi_profiles` without changing values or dtype."""
    _validate_real_array(profiles, name="profiles", dimensions=4)
    shape = _validated_spatial_shape(spatial_shape)
    axis = validate_profile_axis(profile_axis)
    profile_spatial_index = SPATIAL_AXIS_ORDER.index(axis)
    coordinate_indices = tuple(i for i in range(4) if i != profile_spatial_index)
    coordinate_shape = tuple(shape[i] for i in coordinate_indices)
    profile_count = math.prod(coordinate_shape)
    if profiles.shape[0] != profile_count:
        raise ValueError("profile count must equal Sx * Sy * Rx")
    if profiles.shape[1] != 1:
        raise ValueError("profiles must have exactly one amplitude channel")
    if profiles.shape[3] != shape[profile_spatial_index]:
        label = "receiver-y" if axis == "relative_receiver_y" else axis
        raise ValueError(f"profile {label} size must match spatial_shape")

    time_count = profiles.shape[2]
    arranged = profiles.reshape(*coordinate_shape, time_count, shape[profile_spatial_index])
    current_axes = [*coordinate_indices, 4, profile_spatial_index]
    volume = arranged.transpose(
        tuple(current_axes.index(axis_index) for axis_index in (4, 0, 1, 2, 3))
    )
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
    *,
    profile_axis: str = "relative_receiver_y",
) -> ProfileCoordinateBounds:
    """Return full-domain local-index bounds in profile-coordinate order."""
    shape = _validated_spatial_shape(spatial_shape)
    return _coordinate_bounds_from_key_shape(_profile_coordinate_shape(shape, profile_axis))


def profile_coordinate_order(profile_axis: str) -> tuple[str, str, str]:
    """Return the stable coordinate order for one supported profile axis."""
    axis = validate_profile_axis(profile_axis)
    return tuple(name for name in SPATIAL_AXIS_ORDER if name != axis)  # type: ignore[return-value]


def validate_profile_axis(value: object) -> str:
    """Return a supported NeRSI generated-profile axis."""
    if value not in SUPPORTED_PROFILE_AXES:
        raise ValueError(f"profile_axis must be one of {SUPPORTED_PROFILE_AXES!r}")
    return str(value)


def _profile_coordinate_shape(
    spatial_shape: tuple[int, int, int, int], profile_axis: str
) -> tuple[int, int, int]:
    axis = validate_profile_axis(profile_axis)
    index = SPATIAL_AXIS_ORDER.index(axis)
    return tuple(value for i, value in enumerate(spatial_shape) if i != index)  # type: ignore[return-value]


def trace_mask_to_nersi_profiles(
    mask: np.ndarray,
    *,
    profile_axis: str = "relative_receiver_y",
) -> np.ndarray:
    """Arrange a four-dimensional spatial trace mask in NeRSI profile order."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 4 or mask.dtype != np.bool_:
        raise ValueError("mask must be a four-dimensional boolean array")
    axis = validate_profile_axis(profile_axis)
    profile_spatial_index = SPATIAL_AXIS_ORDER.index(axis)
    coordinate_indices = tuple(i for i in range(4) if i != profile_spatial_index)
    return mask.transpose(*coordinate_indices, profile_spatial_index).reshape(
        -1, mask.shape[profile_spatial_index]
    )


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
