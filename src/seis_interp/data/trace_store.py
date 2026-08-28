"""Write a selected trace table and its amplitudes as an interim dataset."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from seis_interp.data.file_checksums import file_sha256
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
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def canonical_source_files(metadata: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    """Return validated source provenance as an ordered tuple of records.

    Legacy datasets store one ``source_file``/``source_sha256`` pair. New
    multi-source datasets store an ordered ``source_files`` list. When both
    representations are present they must describe the same source records.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("source metadata must be an object")

    has_source_files = "source_files" in metadata
    has_legacy_name = "source_file" in metadata
    has_legacy_sha256 = "source_sha256" in metadata
    if has_legacy_name != has_legacy_sha256:
        raise ValueError(
            "source metadata must contain both source_file and source_sha256 or neither"
        )
    if not has_source_files and not has_legacy_name:
        raise ValueError("source metadata must contain source_files or source_file/source_sha256")

    canonical: tuple[dict[str, str], ...] | None = None
    if has_source_files:
        raw_source_files = metadata["source_files"]
        if not isinstance(raw_source_files, list) or not raw_source_files:
            raise ValueError("source_files must be a non-empty list")
        records: list[dict[str, str]] = []
        for index, raw_record in enumerate(raw_source_files):
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"source_files[{index}] must be an object")
            records.append(
                _canonical_source_record(
                    raw_record.get("name"),
                    raw_record.get("sha256"),
                    field_prefix=f"source_files[{index}]",
                )
            )
        names = [record["name"] for record in records]
        if len(names) != len(set(names)):
            raise ValueError("source_files contains duplicate names")
        canonical = tuple(records)

    if has_legacy_name:
        legacy = (
            _canonical_source_record(
                metadata["source_file"],
                metadata["source_sha256"],
                field_prefix="legacy source metadata",
            ),
        )
        if canonical is not None and canonical != legacy:
            raise ValueError(
                "source_files does not match legacy source_file/source_sha256 metadata"
            )
        canonical = legacy

    assert canonical is not None
    return canonical


def validate_trace_identity(trace_table: pd.DataFrame) -> None:
    """Require unique local trace identities, including source when available."""
    if "trace_index" not in trace_table.columns:
        raise ValueError("trace table is missing required column: trace_index")
    if "source_file" in trace_table.columns:
        if trace_table[["source_file", "trace_index"]].duplicated().any():
            raise ValueError("trace table contains duplicate (source_file, trace_index) values")
        return
    if trace_table["trace_index"].duplicated().any():
        raise ValueError("trace table contains duplicate trace_index values")


def build_interim_dataset_metadata(
    trace_table: pd.DataFrame,
    amplitudes: np.ndarray,
    time_s: np.ndarray,
    *,
    dataset_id: str,
    source_metadata: Mapping[str, object],
    selection: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the shared metadata payload for an already validated dataset."""
    source_files = canonical_source_files(source_metadata)
    metadata: dict[str, object] = {
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_count": int(len(trace_table)),
        "sample_count": int(amplitudes.shape[1]),
        "sample_interval_s": _single_sample_interval_s(trace_table),
        "ffids": sorted(int(value) for value in trace_table["ffid"].unique()),
        "selection": _selection_metadata(selection),
        "coordinate_order": list(PHYSICAL_COORDINATE_ORDER),
        "coordinate_units": dict(PHYSICAL_COORDINATE_UNITS),
        "azimuth_convention": AZIMUTH_CONVENTION,
        "time_origin_s": TIME_ORIGIN_S,
        "files": {
            TRACES_FILE_NAME: {
                "row_count": int(len(trace_table)),
                "column_count": int(trace_table.shape[1]),
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
    if "source_files" in source_metadata:
        metadata["source_files"] = [dict(record) for record in source_files]
        if "source_file" in source_metadata:
            metadata["source_file"] = source_files[0]["name"]
            metadata["source_sha256"] = source_files[0]["sha256"]
    else:
        metadata["source_file"] = source_files[0]["name"]
        metadata["source_sha256"] = source_files[0]["sha256"]
    return metadata


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

    metadata = build_interim_dataset_metadata(
        stored_table,
        amplitudes,
        time_s,
        dataset_id=dataset_id,
        source_metadata={
            "source_file": source.name,
            "source_sha256": file_sha256(source),
        },
        selection=selection,
    )

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
    validate_trace_identity(trace_table)

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


def _canonical_source_record(
    name: object,
    sha256: object,
    *,
    field_prefix: str,
) -> dict[str, str]:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(f"{field_prefix} name must be a non-empty basename")
    posix_name = PurePosixPath(name)
    windows_name = PureWindowsPath(name)
    if (
        name in {".", ".."}
        or posix_name.is_absolute()
        or windows_name.is_absolute()
        or windows_name.drive
        or windows_name.root
        or len(posix_name.parts) != 1
        or len(windows_name.parts) != 1
        or posix_name.name != name
        or windows_name.name != name
    ):
        raise ValueError(f"{field_prefix} name must be a basename without path components")
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"{field_prefix} sha256 must be 64 lowercase hexadecimal characters")
    return {"name": name, "sha256": sha256}


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
