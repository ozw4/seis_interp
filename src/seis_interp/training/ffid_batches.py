"""Build and iterate complete-trace batches grouped by FFID."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.trace_splits import (
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from seis_interp.training.amplitude_scaling import (
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
    per_trace_rms_scaled_amplitudes,
    validated_amplitude_scaling,
)

_VALID_SPLITS = frozenset((TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT))


@dataclass(frozen=True)
class FullFfidBatch:
    """One FFID's complete training traces expanded into model points."""

    ffid: int
    coordinates: np.ndarray
    targets: np.ndarray
    trace_count: int
    point_count: int


def array_rows_by_ffid_for_split(
    trace_table: pd.DataFrame,
    split_table: pd.DataFrame,
    *,
    split: str,
) -> dict[int, np.ndarray]:
    """Group sorted ``array_row`` identifiers for one split by their source FFID.

    ``trace_table`` owns the FFID identity, while the deliberately minimal
    ``split_table`` owns split membership. They are joined by ``array_row``;
    neither DataFrame's index or row order has any meaning here.
    """
    requested_split = _validated_split_name(split)
    source_rows, ffid_by_array_row = _validated_trace_table_rows_and_ffids(trace_table)
    split_rows, split_values = _validated_complete_split_table(split_table, source_rows)

    selected_rows = split_rows[split_values == requested_split]
    if selected_rows.size == 0:
        return {}
    selected_ffids = ffid_by_array_row[selected_rows]

    grouped: dict[int, np.ndarray] = {}
    for ffid in np.sort(np.unique(selected_ffids)):
        rows = np.sort(selected_rows[selected_ffids == ffid]).astype(np.int64, copy=False)
        rows.setflags(write=False)
        grouped[int(ffid)] = rows
    return grouped


def validate_all_ffids_have_split_rows(
    trace_table: pd.DataFrame,
    rows_by_ffid: Mapping[int, np.ndarray],
    *,
    split: str,
) -> None:
    """Reject a grouped split unless every source FFID has at least one row."""
    requested_split = _validated_split_name(split)
    _, ffid_by_array_row = _validated_trace_table_rows_and_ffids(trace_table)
    canonical = _validated_rows_by_ffid(
        rows_by_ffid,
        trace_count=len(ffid_by_array_row),
        ffid_by_array_row=ffid_by_array_row,
    )
    expected_ffids = {int(value) for value in np.unique(ffid_by_array_row)}
    actual_ffids = set(canonical)
    missing = sorted(expected_ffids - actual_ffids)
    unexpected = sorted(actual_ffids - expected_ffids)
    if missing or unexpected:
        raise ValueError(
            f"{requested_split} rows do not cover every FFID: "
            f"missing_ffids={missing}, unexpected_ffids={unexpected}"
        )


def build_global_rms_trace_points(
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    amplitudes: np.ndarray,
    array_rows: np.ndarray,
    *,
    amplitude_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected traces and expand globally RMS-scaled points in trace-major order.

    ``amplitudes`` may be a read-only memmap. Advanced indexing materializes only
    the selected trace rows; that one-FFID copy is divided in place, so the full
    amplitude store is never copied or normalized.
    """
    time, spatial, amplitude_array = _validated_point_sources(
        normalized_time,
        normalized_spatial_by_array_row,
        amplitudes,
    )
    rows = _validated_selected_rows(array_rows, spatial.shape[0], "array_rows")
    rms = _positive_finite_float(amplitude_rms, "amplitude_rms")

    selected_spatial = spatial[rows]
    if not np.all(np.isfinite(selected_spatial)):
        raise ValueError("selected normalized spatial coordinates contain non-finite values")

    # Integer and float16 fixtures need promotion for a meaningful division;
    # production interim amplitudes remain float32 and incur no second copy.
    selected_amplitudes = np.asarray(amplitude_array[rows])
    target_dtype = np.result_type(selected_amplitudes.dtype, np.float32)
    if selected_amplitudes.dtype != target_dtype:
        selected_amplitudes = selected_amplitudes.astype(target_dtype)
    elif not selected_amplitudes.flags.writeable:
        selected_amplitudes = selected_amplitudes.copy()
    if not np.all(np.isfinite(selected_amplitudes)):
        raise ValueError("selected amplitudes contain non-finite values")
    np.divide(selected_amplitudes, rms, out=selected_amplitudes)

    return _expand_trace_points(time, selected_spatial, selected_amplitudes)


def build_per_trace_rms_trace_points(
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    amplitudes: np.ndarray,
    array_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected traces and scale each complete trace to unit RMS."""
    time, spatial, amplitude_array = _validated_point_sources(
        normalized_time,
        normalized_spatial_by_array_row,
        amplitudes,
    )
    rows = _validated_selected_rows(array_rows, spatial.shape[0], "array_rows")

    selected_spatial = spatial[rows]
    if not np.all(np.isfinite(selected_spatial)):
        raise ValueError("selected normalized spatial coordinates contain non-finite values")
    selected_amplitudes = per_trace_rms_scaled_amplitudes(
        amplitude_array[rows],
        array_rows=rows,
    )
    return _expand_trace_points(time, selected_spatial, selected_amplitudes)


def _expand_trace_points(
    time: np.ndarray,
    selected_spatial: np.ndarray,
    selected_amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand already selected traces into trace-major coordinate-target points."""
    time_count = len(time)
    coordinates = np.empty(
        (len(selected_spatial) * time_count, len(MODEL_COORDINATE_ORDER)),
        dtype=np.float64,
    )
    coordinate_grid = coordinates.reshape(
        len(selected_spatial),
        time_count,
        len(MODEL_COORDINATE_ORDER),
    )
    coordinate_grid[:, :, 0] = time[np.newaxis, :]
    coordinate_grid[:, :, 1:] = selected_spatial[:, np.newaxis, :]
    targets = selected_amplitudes.reshape(-1)
    return coordinates, targets


class FullFfidBatchSampler:
    """Yield every FFID once per epoch in a deterministic shuffled order."""

    def __init__(
        self,
        normalized_time: np.ndarray,
        normalized_spatial_by_array_row: np.ndarray,
        amplitudes: np.ndarray,
        rows_by_ffid: Mapping[int, np.ndarray],
        *,
        amplitude_rms: float,
        random_seed: int,
        amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    ) -> None:
        time, spatial, amplitude_array = _validated_point_sources(
            normalized_time,
            normalized_spatial_by_array_row,
            amplitudes,
        )
        rms = _positive_finite_float(amplitude_rms, "amplitude_rms")
        scaling = validated_amplitude_scaling(amplitude_scaling)
        canonical_rows = _validated_rows_by_ffid(
            rows_by_ffid,
            trace_count=spatial.shape[0],
        )
        seed = _validated_random_seed(random_seed)

        self._time = time
        self._spatial = spatial
        self._amplitudes = amplitude_array
        self._rows_by_ffid = canonical_rows
        self._ffids = tuple(sorted(canonical_rows))
        self._amplitude_rms = rms
        self._amplitude_scaling = scaling
        self._rng = np.random.default_rng(seed)

    @property
    def ffid_count(self) -> int:
        """Return the effective number of optimizer updates per complete epoch."""
        return len(self._ffids)

    @property
    def ffids(self) -> tuple[int, ...]:
        """Return the canonical sorted FFIDs covered by every epoch."""
        return self._ffids

    @property
    def amplitude_scaling(self) -> str:
        """Return the target scaling used to construct every FFID batch."""
        return self._amplitude_scaling

    def iter_epoch(self) -> Iterator[FullFfidBatch]:
        """Yield one complete batch for each FFID using the next RNG permutation."""
        shuffled_ffids = self._rng.permutation(np.asarray(self._ffids, dtype=np.int64))
        return map(self._build_batch, shuffled_ffids)

    def _build_batch(self, raw_ffid: np.integer) -> FullFfidBatch:
        ffid = int(raw_ffid)
        rows = self._rows_by_ffid[ffid]
        if self._amplitude_scaling == TRAIN_GLOBAL_RMS_SCALING:
            coordinates, targets = build_global_rms_trace_points(
                self._time,
                self._spatial,
                self._amplitudes,
                rows,
                amplitude_rms=self._amplitude_rms,
            )
        elif self._amplitude_scaling == PER_TRACE_RMS_SCALING:
            coordinates, targets = build_per_trace_rms_trace_points(
                self._time,
                self._spatial,
                self._amplitudes,
                rows,
            )
        else:  # pragma: no cover - constructor validation makes this unreachable.
            raise RuntimeError(f"unsupported amplitude scaling: {self._amplitude_scaling!r}")
        return FullFfidBatch(
            ffid=ffid,
            coordinates=coordinates,
            targets=targets,
            trace_count=len(rows),
            point_count=len(targets),
        )


def _validated_trace_table_rows_and_ffids(
    trace_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    if "ffid" not in trace_table.columns:
        raise ValueError("trace table is missing required column: ffid")
    source_rows = validated_array_rows(trace_table, require_contiguous=True)

    ffid_values = trace_table["ffid"]
    if ffid_values.isna().any():
        raise ValueError("ffid contains missing values")
    if is_bool_dtype(ffid_values.dtype) or not is_integer_dtype(ffid_values.dtype):
        raise ValueError(f"ffid must have an integer dtype, got {ffid_values.dtype}")
    int64_info = np.iinfo(np.int64)
    if int(ffid_values.min()) < 0 or int(ffid_values.max()) > int64_info.max:
        raise ValueError("ffid values must be non-negative and fit in int64")
    ffids = ffid_values.to_numpy(dtype=np.int64)

    ffid_by_array_row = np.empty(len(trace_table), dtype=np.int64)
    ffid_by_array_row[source_rows] = ffids
    return source_rows, ffid_by_array_row


def _validated_complete_split_table(
    split_table: pd.DataFrame,
    source_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(split_table, pd.DataFrame):
        raise TypeError(f"split_table must be a pandas DataFrame, got {type(split_table).__name__}")
    if SPLIT_COLUMN not in split_table.columns:
        raise ValueError(f"split table is missing required column: {SPLIT_COLUMN}")
    split_rows = validated_array_rows(split_table)
    missing_rows = np.setdiff1d(source_rows, split_rows).tolist()
    unexpected_rows = np.setdiff1d(split_rows, source_rows).tolist()
    if missing_rows or unexpected_rows:
        raise ValueError(
            "split table array_row values do not match the interim trace table: "
            f"missing_rows={missing_rows}, out_of_range_rows={unexpected_rows}"
        )

    split_values = split_table[SPLIT_COLUMN]
    valid_mask = split_values.isin(_VALID_SPLITS)
    if not bool(valid_mask.all()):
        invalid = split_values.loc[~valid_mask].unique().tolist()
        raise ValueError(f"split table contains invalid split values: {invalid}")
    return split_rows, split_values.to_numpy()


def _validated_rows_by_ffid(
    rows_by_ffid: Mapping[int, np.ndarray],
    *,
    trace_count: int,
    ffid_by_array_row: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    if not isinstance(rows_by_ffid, Mapping):
        raise TypeError("rows_by_ffid must be a mapping")
    if not rows_by_ffid:
        raise ValueError("rows_by_ffid must not be empty")

    canonical: dict[int, np.ndarray] = {}
    used_rows: set[int] = set()
    for raw_ffid, raw_rows in rows_by_ffid.items():
        ffid = _validated_ffid(raw_ffid)
        rows = _validated_selected_rows(raw_rows, trace_count, f"rows for FFID {ffid}")
        sorted_rows = np.sort(rows).copy()
        overlap = sorted(used_rows.intersection(int(row) for row in sorted_rows))
        if overlap:
            raise ValueError(f"array_row values occur in more than one FFID group: {overlap}")
        if ffid_by_array_row is not None and np.any(ffid_by_array_row[sorted_rows] != ffid):
            raise ValueError(f"rows for FFID {ffid} contain rows owned by another FFID")
        used_rows.update(int(row) for row in sorted_rows)
        sorted_rows.setflags(write=False)
        canonical[ffid] = sorted_rows
    return canonical


def _validated_point_sources(
    normalized_time: np.ndarray,
    normalized_spatial_by_array_row: np.ndarray,
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(normalized_time)
    spatial = np.asarray(normalized_spatial_by_array_row)
    amplitude_array = np.asarray(amplitudes)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("normalized_time must be a non-empty one-dimensional array")
    if time.dtype != np.float64:
        raise ValueError(f"normalized_time must be float64, got {time.dtype}")
    if not np.all(np.isfinite(time)):
        raise ValueError("normalized_time contains non-finite values")
    if spatial.ndim != 2:
        raise ValueError("normalized_spatial_by_array_row must be two-dimensional")
    expected_features = len(MODEL_COORDINATE_ORDER) - 1
    if spatial.shape[1] != expected_features:
        raise ValueError(
            f"normalized_spatial_by_array_row must have {expected_features} features, "
            f"got {spatial.shape[1]}"
        )
    if spatial.dtype != np.float64:
        raise ValueError(f"normalized_spatial_by_array_row must be float64, got {spatial.dtype}")
    if amplitude_array.ndim != 2:
        raise ValueError("amplitudes must be two-dimensional")
    if amplitude_array.shape != (spatial.shape[0], len(time)):
        raise ValueError(
            "amplitudes shape must match trace and time counts, got "
            f"{amplitude_array.shape} and expected {(spatial.shape[0], len(time))}"
        )
    if amplitude_array.dtype.kind not in "iuf" or amplitude_array.dtype.kind == "b":
        raise ValueError("amplitudes must contain real numeric values")
    return time, spatial, amplitude_array


def _validated_selected_rows(values: np.ndarray, trace_count: int, name: str) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
        raise ValueError(f"{name} must have an integer dtype")
    if np.any(rows < 0) or np.any(rows >= trace_count):
        raise ValueError(f"{name} values must be within [0, {trace_count})")
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"{name} must contain unique values")
    return rows.astype(np.int64, copy=False)


def _validated_split_name(value: str) -> str:
    if value not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {value!r}")
    return value


def _validated_ffid(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("FFID mapping keys must be non-negative integers")
    ffid = int(value)
    if ffid < 0 or ffid > np.iinfo(np.int64).max:
        raise ValueError("FFID mapping keys must be non-negative integers that fit in int64")
    return ffid


def _validated_random_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("random_seed must be a non-negative integer")
    seed = int(value)
    if seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    return seed


def _positive_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted
