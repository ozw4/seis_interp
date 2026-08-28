"""Sample normalized coordinate-amplitude points for SIREN training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.training.amplitude_scaling import (
    TRAIN_GLOBAL_RMS_SCALING,
    validated_amplitude_scaling,
)


def overlapping_patch_starts(
    sample_count: int,
    patch_size: int,
    overlap_fraction: float,
) -> tuple[int, ...]:
    """Return regular overlapping starts plus a final end-aligned patch."""
    count = _positive_integer(sample_count, "sample_count")
    size = _positive_integer(patch_size, "patch_size")
    if size > count:
        raise ValueError(f"patch_size must not exceed sample_count ({count})")
    if isinstance(overlap_fraction, bool) or not isinstance(overlap_fraction, Real):
        raise ValueError("overlap_fraction must be a finite number within [0, 1)")
    overlap = float(overlap_fraction)
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap_fraction must be a finite number within [0, 1)")
    stride_value = size * (1.0 - overlap)
    if not stride_value.is_integer() or stride_value <= 0.0:
        raise ValueError("patch_size * (1 - overlap_fraction) must be a positive integer")
    stride = int(stride_value)
    final_start = count - size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


class RandomPointSampler:
    """Uniformly sample trace and time indices from training traces."""

    def __init__(
        self,
        normalized_time: np.ndarray,
        normalized_spatial_by_array_row: np.ndarray,
        normalized_amplitudes: np.ndarray,
        training_array_rows: np.ndarray,
        *,
        random_seed: int,
        amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    ) -> None:
        time, spatial, amplitudes = _validated_point_arrays(
            normalized_time,
            normalized_spatial_by_array_row,
            normalized_amplitudes,
        )
        rows = _validated_array_rows(training_array_rows, spatial.shape[0], "training_array_rows")
        if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
            raise ValueError("random_seed must be an integer")
        if int(random_seed) < 0:
            raise ValueError("random_seed must be non-negative")
        scaling = validated_amplitude_scaling(amplitude_scaling)

        self._time = time
        self._spatial = spatial
        self._amplitudes = amplitudes
        self._training_array_rows = rows
        self._amplitude_scaling = scaling
        self._rng = np.random.default_rng(int(random_seed))

    @property
    def amplitude_scaling(self) -> str:
        """Return the target scaling applied before sampling."""
        return self._amplitude_scaling

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one uniformly sampled batch as NumPy arrays."""
        size = _positive_integer(batch_size, "batch_size")
        array_rows = self._rng.choice(self._training_array_rows, size=size, replace=True)
        time_indices = self._rng.integers(0, len(self._time), size=size)

        coordinates = np.empty((size, len(MODEL_COORDINATE_ORDER)), dtype=np.float64)
        coordinates[:, 0] = self._time[time_indices]
        coordinates[:, 1:] = self._spatial[array_rows]
        targets = self._amplitudes[array_rows, time_indices]
        return coordinates, targets


class RandomTraceBatchSampler:
    """Uniformly sample distinct training traces and keep every time sample."""

    def __init__(
        self,
        normalized_time: np.ndarray,
        normalized_spatial_by_array_row: np.ndarray,
        normalized_amplitudes: np.ndarray,
        training_array_rows: np.ndarray,
        *,
        random_seed: int,
    ) -> None:
        time, spatial, amplitudes = _validated_point_arrays(
            normalized_time,
            normalized_spatial_by_array_row,
            normalized_amplitudes,
        )
        rows = _validated_array_rows(training_array_rows, spatial.shape[0], "training_array_rows")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("training_array_rows must contain unique values")
        if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
            raise ValueError("random_seed must be an integer")
        if int(random_seed) < 0:
            raise ValueError("random_seed must be non-negative")

        self._time = time
        self._spatial = spatial
        self._amplitudes = amplitudes
        self._training_array_rows = rows
        self._rng = np.random.default_rng(int(random_seed))

    def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
        """Return all points from one random batch of distinct training traces."""
        trace_count = _positive_integer(traces_per_update, "traces_per_update")
        available_trace_count = len(self._training_array_rows)
        if trace_count > available_trace_count:
            raise ValueError(
                "traces_per_update must not exceed the number of available training rows "
                f"({available_trace_count})"
            )
        array_rows = self._rng.choice(
            self._training_array_rows,
            size=trace_count,
            replace=False,
        )
        return build_trace_points(
            self._time,
            self._spatial,
            self._amplitudes,
            array_rows,
        )


class RandomTracePatchSampler:
    """Sample distinct traces over one shared random contiguous time patch."""

    def __init__(
        self,
        normalized_time: np.ndarray,
        normalized_spatial_by_array_row: np.ndarray,
        normalized_amplitudes: np.ndarray,
        training_array_rows: np.ndarray,
        *,
        patch_size: int,
        patch_starts: Sequence[int] | np.ndarray,
        random_seed: int,
    ) -> None:
        time, spatial, amplitudes = _validated_point_arrays(
            normalized_time,
            normalized_spatial_by_array_row,
            normalized_amplitudes,
        )
        rows = _validated_array_rows(training_array_rows, spatial.shape[0], "training_array_rows")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("training_array_rows must contain unique values")
        size = _positive_integer(patch_size, "patch_size")
        starts = _validated_patch_starts(patch_starts, len(time), size)
        if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
            raise ValueError("random_seed must be an integer")
        if int(random_seed) < 0:
            raise ValueError("random_seed must be non-negative")

        self._time = time
        self._spatial = spatial
        self._amplitudes = amplitudes
        self._training_array_rows = rows
        self._patch_size = size
        self._patch_starts = starts
        self._rng = np.random.default_rng(int(random_seed))

    def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
        """Return complete shared-window patches from distinct training traces."""
        trace_count = _positive_integer(traces_per_update, "traces_per_update")
        available_trace_count = len(self._training_array_rows)
        if trace_count > available_trace_count:
            raise ValueError(
                "traces_per_update must not exceed the number of available training rows "
                f"({available_trace_count})"
            )
        array_rows = self._rng.choice(
            self._training_array_rows,
            size=trace_count,
            replace=False,
        )
        patch_start = int(self._rng.choice(self._patch_starts))
        patch_stop = patch_start + self._patch_size
        return build_trace_points(
            self._time[patch_start:patch_stop],
            self._spatial,
            self._amplitudes[:, patch_start:patch_stop],
            array_rows,
        )


def build_trace_points(
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    normalized_amplitudes: np.ndarray,
    array_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand all time samples for selected traces in trace-major order."""
    time, spatial, amplitudes = _validated_point_arrays(
        normalized_time,
        normalized_spatial_by_array_row,
        normalized_amplitudes,
    )
    rows = _validated_array_rows(array_rows, spatial.shape[0], "array_rows")
    time_count = len(time)

    coordinates = np.empty((len(rows) * time_count, len(MODEL_COORDINATE_ORDER)), dtype=np.float64)
    coordinates[:, 0] = np.tile(time, len(rows))
    coordinates[:, 1:] = np.repeat(spatial[rows], time_count, axis=0)
    targets = amplitudes[rows].reshape(-1)
    return coordinates, targets


def _validated_point_arrays(
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    normalized_amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(normalized_time)
    spatial = np.asarray(normalized_spatial_by_array_row)
    amplitudes = np.asarray(normalized_amplitudes)
    if time.ndim != 1 or time.size == 0:
        raise ValueError(
            f"normalized_time must be a non-empty one-dimensional array, got {time.shape}"
        )
    if spatial.ndim != 2:
        raise ValueError(
            f"normalized_spatial_by_array_row must be two-dimensional, got {spatial.shape}"
        )
    expected_features = len(MODEL_COORDINATE_ORDER) - 1
    if spatial.shape[1] != expected_features:
        raise ValueError(
            "normalized_spatial_by_array_row must have "
            f"{expected_features} features, got {spatial.shape[1]}"
        )
    if amplitudes.ndim != 2:
        raise ValueError(f"normalized_amplitudes must be two-dimensional, got {amplitudes.shape}")
    if amplitudes.shape != (spatial.shape[0], len(time)):
        raise ValueError(
            "normalized_amplitudes shape must match trace and time counts, got "
            f"{amplitudes.shape} and expected {(spatial.shape[0], len(time))}"
        )
    if time.dtype != np.float64:
        raise ValueError(f"normalized_time must be float64, got {time.dtype}")
    if spatial.dtype != np.float64:
        raise ValueError(f"normalized_spatial_by_array_row must be float64, got {spatial.dtype}")
    return time, spatial, amplitudes


def _validated_array_rows(values: np.ndarray, trace_count: int, name: str) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
        raise ValueError(f"{name} must have an integer dtype")
    if np.any(rows < 0) or np.any(rows >= trace_count):
        raise ValueError(f"{name} values must be within [0, {trace_count})")
    return rows.astype(np.int64, copy=False)


def _validated_patch_starts(
    values: Sequence[int] | np.ndarray,
    sample_count: int,
    patch_size: int,
) -> np.ndarray:
    if patch_size > sample_count:
        raise ValueError(
            f"patch_size must not exceed the number of available samples ({sample_count})"
        )
    starts = np.asarray(values)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("patch_starts must be a non-empty one-dimensional array")
    if starts.dtype.kind not in "iu" or starts.dtype.kind == "b":
        raise ValueError("patch_starts must have an integer dtype")
    if len(np.unique(starts)) != len(starts):
        raise ValueError("patch_starts must contain unique values")
    maximum_start = sample_count - patch_size
    if np.any(starts < 0) or np.any(starts > maximum_start):
        raise ValueError(f"patch_starts values must be within [0, {maximum_start}]")
    return starts.astype(np.int64, copy=False)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
