"""Materialize leakage-safe observed C3 volumes from trace artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.benchmark_case_inputs import verify_benchmark_case_inputs
from seis_interp.data.benchmark_case_store import load_benchmark_case
from seis_interp.data.c3_volume_index_inputs import load_bound_benchmark_case
from seis_interp.data.c3_volume_index_store import (
    load_c3_volume_index,
    validate_c3_volume_index,
)
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.trace_store import AMPLITUDES_FILE_NAME, TIME_FILE_NAME
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
    validate_interpolation_mask,
)


@dataclass(frozen=True)
class ObservedC3Volume:
    """One zero-filled observed volume and its trace-level role mapping."""

    values: np.ndarray
    time_s: np.ndarray
    array_rows: np.ndarray
    observed_trace_mask: np.ndarray
    evaluation_target_trace_mask: np.ndarray


def materialize_observed_c3_volume(
    amplitudes: np.ndarray,
    time_s: np.ndarray,
    index_table: pd.DataFrame,
    volume_metadata: Mapping[str, object],
    mask_table: pd.DataFrame,
) -> ObservedC3Volume:
    """Read observed rows only and place them in a dense zero-filled volume."""
    validate_c3_volume_index(index_table, volume_metadata)
    validate_interpolation_mask(mask_table)
    _validate_source_arrays(amplitudes, time_s)

    array_rows_flat = index_table["array_row"].to_numpy(dtype=np.int64)
    if np.any(array_rows_flat < 0) or np.any(array_rows_flat >= amplitudes.shape[0]):
        raise ValueError("volume index array_row values are outside the amplitude array")

    selected_mask = index_table[["array_row"]].merge(
        mask_table,
        on="array_row",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if selected_mask[OBSERVATION_ROLE_COLUMN].isna().any():
        raise ValueError("volume index rows are not all present exactly once in the mask")
    roles = selected_mask[OBSERVATION_ROLE_COLUMN].to_numpy()
    observed_flat = roles == OBSERVED_ROLE
    target_flat = roles == EVALUATION_TARGET_ROLE
    if np.any(observed_flat & target_flat) or not np.all(observed_flat | target_flat):
        raise ValueError("observed and evaluation target roles must disjointly cover the volume")

    expected_counts = volume_metadata["role_counts"]
    actual_counts = {
        OBSERVED_ROLE: int(observed_flat.sum()),
        EVALUATION_TARGET_ROLE: int(target_flat.sum()),
    }
    if actual_counts != dict(expected_counts):  # type: ignore[arg-type]
        raise ValueError(
            f"selected mask role counts {actual_counts} do not match volume metadata "
            f"{dict(expected_counts)!r}"  # type: ignore[arg-type]
        )

    time_start, time_stop = volume_metadata["selection"]["time"]  # type: ignore[index]
    if time_stop > len(time_s) or time_stop > amplitudes.shape[1]:
        raise ValueError("volume time selection is outside the source arrays")

    spatial_shape = tuple(int(value) for value in volume_metadata["shape"][1:])  # type: ignore[index]
    time_count = int(time_stop - time_start)
    values_flat = np.zeros((len(index_table), time_count), dtype=amplitudes.dtype)
    observed_rows = array_rows_flat[observed_flat]
    observed_values = amplitudes[observed_rows, time_start:time_stop]
    if not np.all(np.isfinite(observed_values)):
        raise ValueError("observed amplitudes contain non-finite values")
    values_flat[observed_flat] = observed_values

    values = values_flat.T.reshape((time_count, *spatial_shape))
    array_rows = array_rows_flat.reshape(spatial_shape)
    observed_mask = observed_flat.reshape(spatial_shape)
    target_mask = target_flat.reshape(spatial_shape)
    return ObservedC3Volume(
        values=np.ascontiguousarray(values),
        time_s=np.array(time_s[time_start:time_stop], copy=True),
        array_rows=np.ascontiguousarray(array_rows, dtype=np.int64),
        observed_trace_mask=np.ascontiguousarray(observed_mask, dtype=np.bool_),
        evaluation_target_trace_mask=np.ascontiguousarray(target_mask, dtype=np.bool_),
    )


def load_observed_c3_volume(
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
) -> ObservedC3Volume:
    """Verify all bindings and materialize an observed C3 volume from disk."""
    case = load_benchmark_case(case_dir)
    verify_benchmark_case_inputs(
        case,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
    )
    index_table, volume_metadata = load_c3_volume_index(volume_dir)
    bound_case = load_bound_benchmark_case(volume_metadata, case_dir=case_dir)
    if bound_case["case_id"] != case["case_id"]:
        raise ValueError("loaded benchmark case ID does not match the volume-bound case")
    mask_table, _ = load_interpolation_mask(mask_dir)
    amplitudes = np.load(
        Path(interim_dir) / AMPLITUDES_FILE_NAME,
        mmap_mode="r",
        allow_pickle=False,
    )
    time_s = np.load(Path(interim_dir) / TIME_FILE_NAME, allow_pickle=False)
    return materialize_observed_c3_volume(
        amplitudes,
        time_s,
        index_table,
        volume_metadata,
        mask_table,
    )


def volume_to_trace_predictions(
    predicted_values: np.ndarray,
    array_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial-major trace predictions from a dense five-dimensional volume."""
    if not isinstance(predicted_values, np.ndarray) or predicted_values.ndim != 5:
        raise ValueError("predicted_values must be a five-dimensional NumPy array")
    if predicted_values.dtype.kind not in "fiu" or predicted_values.dtype.kind == "b":
        raise ValueError("predicted_values must contain real numeric values")
    if not isinstance(array_rows, np.ndarray) or array_rows.ndim != 4:
        raise ValueError("array_rows must be a four-dimensional NumPy array")
    if array_rows.dtype.kind not in "iu" or array_rows.dtype.kind == "b":
        raise ValueError("array_rows must contain integers")
    if predicted_values.shape[1:] != array_rows.shape:
        raise ValueError("predicted_values spatial shape must match array_rows")
    if np.any(array_rows < 0):
        raise ValueError("array_rows must be nonnegative")
    flat_rows = array_rows.reshape(-1)
    if len(np.unique(flat_rows)) != len(flat_rows):
        raise ValueError("array_rows must be unique")
    if len(flat_rows) and int(flat_rows.max()) > np.iinfo(np.int64).max:
        raise ValueError("array_rows values must fit in int64")

    trace_values = predicted_values.transpose(1, 2, 3, 4, 0).reshape(
        len(flat_rows), predicted_values.shape[0]
    )
    return flat_rows.astype(np.int64, copy=False), trace_values


def _validate_source_arrays(amplitudes: np.ndarray, time_s: np.ndarray) -> None:
    if not isinstance(amplitudes, np.ndarray) or amplitudes.ndim != 2:
        raise ValueError("amplitudes must be a two-dimensional NumPy array")
    if amplitudes.dtype.kind not in "fiu" or amplitudes.dtype.kind == "b":
        raise ValueError("amplitudes must contain real numeric values")
    if not isinstance(time_s, np.ndarray) or time_s.ndim != 1:
        raise ValueError("time_s must be a one-dimensional NumPy array")
    if time_s.dtype.kind not in "fiu" or time_s.dtype.kind == "b":
        raise ValueError("time_s must contain real numeric values")
    if amplitudes.shape[1] != len(time_s):
        raise ValueError("amplitude sample count must match time_s length")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s contains non-finite values")
