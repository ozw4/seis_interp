"""Sample normalized coordinate-amplitude points for SIREN training."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER


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

        self._time = time
        self._spatial = spatial
        self._amplitudes = amplitudes
        self._training_array_rows = rows
        self._rng = np.random.default_rng(int(random_seed))

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


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
