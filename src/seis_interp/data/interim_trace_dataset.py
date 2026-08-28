"""Load and validate an interim trace dataset from disk."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from seis_interp.data.trace_schema import PHYSICAL_COORDINATE_ORDER, PHYSICAL_COORDINATE_UNITS
from seis_interp.data.trace_store import (
    AMPLITUDES_FILE_NAME,
    METADATA_FILE_NAME,
    OUTPUT_FILE_NAMES,
    TIME_FILE_NAME,
    TRACES_FILE_NAME,
    canonical_source_files,
    validate_trace_identity,
)
from seis_interp.data.trace_table import validated_array_rows


@dataclass(frozen=True)
class InterimTraceDataset:
    """The table, arrays, and metadata stored for one interim trace dataset."""

    trace_table: pd.DataFrame
    amplitudes: np.ndarray
    time_s: np.ndarray
    metadata: dict[str, object]


_AMPLITUDE_VALIDATION_ROW_CHUNK_SIZE = 4096


def load_interim_trace_dataset(
    directory: Path,
    *,
    memory_map_amplitudes: bool = False,
) -> InterimTraceDataset:
    """Load an interim trace dataset and validate its on-disk contract."""
    dataset_dir = Path(directory)
    paths = {file_name: dataset_dir / file_name for file_name in OUTPUT_FILE_NAMES}
    missing = [file_name for file_name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"interim trace dataset is missing required files in {dataset_dir}: {missing}"
        )

    metadata = _read_metadata(paths[METADATA_FILE_NAME])
    trace_table = pd.read_parquet(paths[TRACES_FILE_NAME])
    amplitudes = np.load(
        paths[AMPLITUDES_FILE_NAME],
        mmap_mode="r" if memory_map_amplitudes else None,
        allow_pickle=False,
    )
    time_s = np.load(paths[TIME_FILE_NAME], allow_pickle=False)

    validated_array_rows(trace_table, require_contiguous=True)
    validate_trace_identity(trace_table)
    _validate_numeric_table_values(trace_table)
    _validate_arrays(trace_table, amplitudes, time_s)
    _validate_metadata(metadata, trace_table, amplitudes, time_s)

    return InterimTraceDataset(
        trace_table=trace_table,
        amplitudes=amplitudes,
        time_s=time_s,
        metadata=metadata,
    )


def _read_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{METADATA_FILE_NAME} must contain a JSON object")
    return payload


def _validate_numeric_table_values(trace_table: pd.DataFrame) -> None:
    for column in trace_table.columns:
        if not is_numeric_dtype(trace_table[column].dtype):
            continue
        try:
            values = trace_table[column].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"trace table column {column!r} contains invalid values") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"trace table column {column!r} contains non-finite values")


def _validate_arrays(
    trace_table: pd.DataFrame,
    amplitudes: np.ndarray,
    time_s: np.ndarray,
) -> None:
    if amplitudes.ndim != 2:
        raise ValueError(
            f"{AMPLITUDES_FILE_NAME} must be two-dimensional, got {amplitudes.ndim} dimensions"
        )
    if time_s.ndim != 1:
        raise ValueError(f"{TIME_FILE_NAME} must be one-dimensional, got {time_s.ndim} dimensions")
    if amplitudes.dtype != np.float32:
        raise ValueError(f"{AMPLITUDES_FILE_NAME} must be float32, got {amplitudes.dtype}")
    if time_s.dtype != np.float64:
        raise ValueError(f"{TIME_FILE_NAME} must be float64, got {time_s.dtype}")
    if amplitudes.shape[0] != len(trace_table):
        raise ValueError(
            f"{AMPLITUDES_FILE_NAME} has {amplitudes.shape[0]} rows but "
            f"{TRACES_FILE_NAME} has {len(trace_table)} rows"
        )
    if amplitudes.shape[1] != len(time_s):
        raise ValueError(
            f"{AMPLITUDES_FILE_NAME} has {amplitudes.shape[1]} samples but "
            f"{TIME_FILE_NAME} has {len(time_s)} values"
        )
    for start in range(0, amplitudes.shape[0], _AMPLITUDE_VALIDATION_ROW_CHUNK_SIZE):
        stop = min(start + _AMPLITUDE_VALIDATION_ROW_CHUNK_SIZE, amplitudes.shape[0])
        if not np.all(np.isfinite(amplitudes[start:stop])):
            raise ValueError(f"{AMPLITUDES_FILE_NAME} contains non-finite values")
    if not np.all(np.isfinite(time_s)):
        raise ValueError(f"{TIME_FILE_NAME} contains non-finite values")


def _validate_metadata(
    metadata: Mapping[str, object],
    trace_table: pd.DataFrame,
    amplitudes: np.ndarray,
    time_s: np.ndarray,
) -> None:
    canonical_source_files(metadata)
    _validate_coordinate_schema(metadata)

    trace_count = _metadata_integer(metadata, "trace_count")
    sample_count = _metadata_integer(metadata, "sample_count")
    if trace_count != len(trace_table):
        raise ValueError(
            f"metadata trace_count is {trace_count} but {TRACES_FILE_NAME} has "
            f"{len(trace_table)} rows"
        )
    if sample_count != len(time_s):
        raise ValueError(
            f"metadata sample_count is {sample_count} but {TIME_FILE_NAME} has {len(time_s)} values"
        )

    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("metadata files must be an object")

    trace_record = _file_record(files, TRACES_FILE_NAME)
    _require_record_integer(trace_record, "row_count", len(trace_table), TRACES_FILE_NAME)
    _require_record_integer(
        trace_record,
        "column_count",
        trace_table.shape[1],
        TRACES_FILE_NAME,
    )

    amplitude_record = _file_record(files, AMPLITUDES_FILE_NAME)
    _require_record_dtype(amplitude_record, amplitudes.dtype, AMPLITUDES_FILE_NAME)
    _require_record_shape(amplitude_record, amplitudes.shape, AMPLITUDES_FILE_NAME)

    time_record = _file_record(files, TIME_FILE_NAME)
    _require_record_dtype(time_record, time_s.dtype, TIME_FILE_NAME)
    _require_record_shape(time_record, time_s.shape, TIME_FILE_NAME)


def _validate_coordinate_schema(metadata: Mapping[str, object]) -> None:
    expected_order = list(PHYSICAL_COORDINATE_ORDER)
    coordinate_order = metadata.get("coordinate_order")
    if coordinate_order != expected_order:
        raise ValueError(
            f"metadata coordinate_order is {coordinate_order!r} but expected {expected_order!r}"
        )

    expected_units = dict(PHYSICAL_COORDINATE_UNITS)
    coordinate_units = metadata.get("coordinate_units")
    if coordinate_units != expected_units:
        raise ValueError(
            f"metadata coordinate_units is {coordinate_units!r} but expected {expected_units!r}"
        )


def _metadata_integer(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"metadata {key} must be a non-negative integer")
    return value


def _file_record(files: Mapping[str, object], file_name: str) -> Mapping[str, object]:
    record = files.get(file_name)
    if not isinstance(record, Mapping):
        raise ValueError(f"metadata files entry for {file_name} must be an object")
    return record


def _require_record_integer(
    record: Mapping[str, object],
    key: str,
    expected: int,
    file_name: str,
) -> None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"metadata {file_name} {key} must be an integer")
    if value != expected:
        raise ValueError(f"metadata {file_name} {key} is {value} but the file contains {expected}")


def _require_record_dtype(
    record: Mapping[str, object],
    actual: np.dtype[object],
    file_name: str,
) -> None:
    expected = str(actual)
    if record.get("dtype") != expected:
        raise ValueError(
            f"metadata {file_name} dtype is {record.get('dtype')!r} but the file is {expected}"
        )


def _require_record_shape(
    record: Mapping[str, object],
    actual: Sequence[int],
    file_name: str,
) -> None:
    shape = record.get("shape")
    if (
        not isinstance(shape, list)
        or any(isinstance(size, bool) or not isinstance(size, int) for size in shape)
        or shape != [int(size) for size in actual]
    ):
        raise ValueError(
            f"metadata {file_name} shape is {shape!r} but the file shape is {tuple(actual)}"
        )
