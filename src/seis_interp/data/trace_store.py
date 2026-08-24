"""Write a selected trace table and its amplitudes as an interim dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.trace_schema import PHYSICAL_COORDINATE_ORDER, PHYSICAL_COORDINATE_UNITS

TRACES_FILE_NAME = "traces.parquet"
AMPLITUDES_FILE_NAME = "amplitudes.npy"
TIME_FILE_NAME = "time_s.npy"
METADATA_FILE_NAME = "dataset.json"

OUTPUT_FILE_NAMES = (
    TRACES_FILE_NAME,
    AMPLITUDES_FILE_NAME,
    TIME_FILE_NAME,
    METADATA_FILE_NAME,
)

AZIMUTH_CONVENTION = "degrees(atan2(source_x-receiver_x, source_y-receiver_y)) wrapped to [0, 360)"

TIME_ORIGIN_S = 0.0

_REQUIRED_COLUMNS = (
    "trace_index",
    "ffid",
    "cmp_x_m",
    "cmp_y_m",
    "offset_m",
    "azimuth_deg",
    "sample_interval_s",
)
_FINITE_COLUMNS = ("cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg")
_SHA256_CHUNK_BYTES = 1024 * 1024


def write_interim_trace_dataset(
    output_dir: Path,
    trace_table: pd.DataFrame,
    amplitudes: np.ndarray,
    time_s: np.ndarray,
    source_path: Path,
    dataset_id: str,
    selection: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write ``traces.parquet``, ``amplitudes.npy``, ``time_s.npy`` and ``dataset.json``.

    An ``array_row`` column is added to the trace table so that row ``i`` of
    the Parquet table corresponds to ``amplitudes[i]``. ``selection`` records
    how the caller chose these traces and is stored under the ``selection``
    key, so that the dataset can be reproduced from ``dataset.json`` alone.

    Returns exactly the metadata that was written to ``dataset.json``.
    """
    directory = Path(output_dir)
    source = Path(source_path)

    _validate_arrays(trace_table, amplitudes, time_s)
    _validate_trace_table(trace_table)
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    _check_output_directory(directory, overwrite=overwrite)

    stored_table = trace_table.reset_index(drop=True).copy()
    stored_table.insert(0, "array_row", np.arange(len(stored_table), dtype=np.int64))

    sample_interval_s = _single_sample_interval_s(stored_table)
    metadata: dict[str, object] = {
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source.name,
        "source_sha256": _file_sha256(source),
        "trace_count": int(len(stored_table)),
        "sample_count": int(amplitudes.shape[1]),
        "sample_interval_s": sample_interval_s,
        "ffids": sorted(int(value) for value in stored_table["ffid"].unique()),
        "selection": _selection_metadata(selection),
        "coordinate_order": list(PHYSICAL_COORDINATE_ORDER),
        "coordinate_units": dict(PHYSICAL_COORDINATE_UNITS),
        "azimuth_convention": AZIMUTH_CONVENTION,
        "time_origin_s": TIME_ORIGIN_S,
        "files": {
            TRACES_FILE_NAME: {
                "row_count": int(len(stored_table)),
                "column_count": int(stored_table.shape[1]),
            },
            AMPLITUDES_FILE_NAME: {
                "dtype": str(amplitudes.dtype),
                "shape": [int(size) for size in amplitudes.shape],
            },
            TIME_FILE_NAME: {
                "dtype": str(time_s.dtype),
                "shape": [int(size) for size in time_s.shape],
            },
        },
    }

    directory.mkdir(parents=True, exist_ok=True)
    stored_table.to_parquet(directory / TRACES_FILE_NAME, index=False)
    np.save(directory / AMPLITUDES_FILE_NAME, amplitudes)
    np.save(directory / TIME_FILE_NAME, time_s)
    (directory / METADATA_FILE_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _validate_arrays(
    trace_table: pd.DataFrame,
    amplitudes: np.ndarray,
    time_s: np.ndarray,
) -> None:
    """Check the shape and dtype contract between the table and the arrays."""
    if amplitudes.ndim != 2:
        raise ValueError(f"amplitudes must be two-dimensional, got {amplitudes.ndim} dimensions")
    if time_s.ndim != 1:
        raise ValueError(f"time_s must be one-dimensional, got {time_s.ndim} dimensions")
    if len(trace_table) != amplitudes.shape[0]:
        raise ValueError(
            f"trace table has {len(trace_table)} rows but amplitudes has {amplitudes.shape[0]} rows"
        )
    if amplitudes.shape[1] != len(time_s):
        raise ValueError(
            f"amplitudes has {amplitudes.shape[1]} samples but time_s has {len(time_s)} values"
        )
    if amplitudes.dtype != np.float32:
        raise ValueError(f"amplitudes must be float32, got {amplitudes.dtype}")
    if time_s.dtype != np.float64:
        raise ValueError(f"time_s must be float64, got {time_s.dtype}")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes contain non-finite values")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s contains non-finite values")


def _validate_trace_table(trace_table: pd.DataFrame) -> None:
    """Check that the required trace-table columns exist and are usable."""
    missing = [column for column in _REQUIRED_COLUMNS if column not in trace_table.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")
    if trace_table.empty:
        raise ValueError("trace table is empty")
    if trace_table["trace_index"].duplicated().any():
        raise ValueError("trace table contains duplicate trace_index values")

    geometry = trace_table[list(_FINITE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(geometry)):
        raise ValueError("trace table geometry columns contain non-finite values")


def _selection_metadata(selection: Mapping[str, object] | None) -> dict[str, object]:
    """Return the selection provenance as a JSON-serialisable dictionary."""
    if selection is None:
        return {}
    if not isinstance(selection, Mapping):
        raise ValueError(f"selection must be a mapping, got {type(selection).__name__}")

    stored = dict(selection)
    non_string_keys = [key for key in stored if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError(f"selection keys must be strings, got {non_string_keys}")
    try:
        json.dumps(stored)
    except TypeError as error:
        raise ValueError(f"selection is not JSON serialisable: {error}") from error
    return stored


def _single_sample_interval_s(trace_table: pd.DataFrame) -> float:
    """Return the one sample interval shared by every selected trace."""
    intervals = trace_table["sample_interval_s"].unique()
    if len(intervals) != 1:
        raise ValueError(f"trace table has more than one sample interval: {intervals.tolist()}")
    return float(intervals[0])


def _check_output_directory(directory: Path, overwrite: bool) -> None:
    """Refuse to write into a non-empty directory unless overwrite is requested."""
    if overwrite or not directory.exists():
        return
    if any(directory.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {directory}; pass overwrite=True to replace "
            f"the generated files"
        )


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_SHA256_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
