"""Scan SEG-Y trace headers into a table with one row per trace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import segyio

from seis_interp.processing.geometry import apply_coordinate_scalar, compute_trace_geometry

TRACE_TABLE_COLUMNS = (
    "source_file",
    "trace_index",
    "ffid",
    "trace_index_in_ffid",
    "coordinate_scalar",
    "coordinate_units",
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
    "cmp_x_m",
    "cmp_y_m",
    "offset_m",
    "azimuth_deg",
    "sample_count",
    "sample_interval_s",
)

GEOMETRY_COLUMNS = (
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
    "cmp_x_m",
    "cmp_y_m",
    "offset_m",
    "azimuth_deg",
)

_LENGTH_COORDINATE_UNIT = 1


def scan_segy_headers(path: Path) -> pd.DataFrame:
    """Read the trace headers of one SEG-Y file into a trace table.

    Every trace becomes one row. Coordinates are scaled with the per-trace
    SEG-Y coordinate scalar and CMP, offset and azimuth are derived from them.
    The file is opened without geometry inference, so no fixed shot-gather
    shape is assumed.
    """
    segy_path = Path(path)
    if not segy_path.is_file():
        raise FileNotFoundError(f"SEG-Y file not found: {segy_path}")

    with segyio.open(
        str(segy_path),
        mode="r",
        strict=False,
        ignore_geometry=True,
    ) as handle:
        trace_count = int(handle.tracecount)
        if trace_count == 0:
            raise ValueError(f"SEG-Y file contains no traces: {segy_path.name}")

        sample_count = len(handle.samples)
        if sample_count == 0:
            raise ValueError(f"SEG-Y file contains no samples: {segy_path.name}")

        sample_interval_s = _read_sample_interval_s(handle, segy_path.name)

        ffid = handle.attributes(segyio.TraceField.FieldRecord)[:]
        coordinate_scalar = handle.attributes(segyio.TraceField.SourceGroupScalar)[:]
        coordinate_units = handle.attributes(segyio.TraceField.CoordinateUnits)[:]
        raw_source_x = handle.attributes(segyio.TraceField.SourceX)[:]
        raw_source_y = handle.attributes(segyio.TraceField.SourceY)[:]
        raw_receiver_x = handle.attributes(segyio.TraceField.GroupX)[:]
        raw_receiver_y = handle.attributes(segyio.TraceField.GroupY)[:]

    _validate_coordinate_units(coordinate_units, segy_path.name)

    source_x_m = apply_coordinate_scalar(raw_source_x, coordinate_scalar)
    source_y_m = apply_coordinate_scalar(raw_source_y, coordinate_scalar)
    receiver_x_m = apply_coordinate_scalar(raw_receiver_x, coordinate_scalar)
    receiver_y_m = apply_coordinate_scalar(raw_receiver_y, coordinate_scalar)

    cmp_x_m, cmp_y_m, offset_m, azimuth_deg = compute_trace_geometry(
        source_x_m,
        source_y_m,
        receiver_x_m,
        receiver_y_m,
    )

    trace_table = pd.DataFrame(
        {
            "source_file": segy_path.name,
            "trace_index": np.arange(trace_count, dtype=np.int64),
            "ffid": ffid.astype(np.int64),
            "coordinate_scalar": coordinate_scalar.astype(np.int64),
            "coordinate_units": coordinate_units.astype(np.int64),
            "source_x_m": source_x_m,
            "source_y_m": source_y_m,
            "receiver_x_m": receiver_x_m,
            "receiver_y_m": receiver_y_m,
            "cmp_x_m": cmp_x_m,
            "cmp_y_m": cmp_y_m,
            "offset_m": offset_m,
            "azimuth_deg": azimuth_deg,
            "sample_count": np.int64(sample_count),
            "sample_interval_s": float(sample_interval_s),
        }
    )
    trace_table["trace_index_in_ffid"] = (
        trace_table.groupby("ffid", sort=False).cumcount().astype(np.int64)
    )

    _validate_geometry(trace_table, segy_path.name)
    return trace_table[list(TRACE_TABLE_COLUMNS)]


def _read_sample_interval_s(handle: segyio.SegyFile, file_name: str) -> float:
    """Return the sample interval in seconds from the SEG-Y binary header."""
    interval_us = int(handle.bin[segyio.BinField.Interval])
    if interval_us <= 0:
        raise ValueError(
            f"SEG-Y binary header has a non-positive sample interval "
            f"({interval_us} microseconds): {file_name}"
        )
    return interval_us / 1_000_000.0


def _validate_coordinate_units(coordinate_units: np.ndarray, file_name: str) -> None:
    """Accept only coordinate unit code 1 (length), which this POC treats as metres."""
    unsupported = np.unique(coordinate_units[coordinate_units != _LENGTH_COORDINATE_UNIT])
    if unsupported.size:
        raise ValueError(
            f"unsupported coordinate units {unsupported.tolist()} in {file_name}; "
            f"only unit code {_LENGTH_COORDINATE_UNIT} (length in metres) is supported"
        )


def _validate_geometry(trace_table: pd.DataFrame, file_name: str) -> None:
    """Reject NaN or Inf in any derived geometry column."""
    geometry = trace_table[list(GEOMETRY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(geometry)):
        raise ValueError(f"geometry columns contain non-finite values: {file_name}")
