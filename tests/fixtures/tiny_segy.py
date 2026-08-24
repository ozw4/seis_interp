"""Write tiny SEG-Y files for tests.

The generated files stay inside pytest temporary directories; no SEG-Y binary
is committed to the repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segyio


@dataclass(frozen=True)
class TinyTraceHeader:
    """Raw SEG-Y header values for one synthetic trace."""

    ffid: int
    coordinate_scalar: int
    source_x: int
    source_y: int
    receiver_x: int
    receiver_y: int


@dataclass(frozen=True)
class TinySegyFile:
    """A generated tiny SEG-Y file and the values it was written from."""

    path: Path
    headers: tuple[TinyTraceHeader, ...]
    amplitudes: np.ndarray
    sample_count: int
    sample_interval_s: float
    coordinate_units: int


# FFID 10 holds three traces and FFID 20 holds four, so a test can treat 4 as
# the expected complete-shot size and still see an incomplete FFID first.
# Positive, negative and zero coordinate scalars all appear.
DEFAULT_HEADERS: tuple[TinyTraceHeader, ...] = (
    TinyTraceHeader(
        ffid=10, coordinate_scalar=10, source_x=100, source_y=200, receiver_x=140, receiver_y=200
    ),
    TinyTraceHeader(
        ffid=10, coordinate_scalar=10, source_x=100, source_y=200, receiver_x=100, receiver_y=260
    ),
    TinyTraceHeader(
        ffid=10, coordinate_scalar=0, source_x=1000, source_y=2000, receiver_x=1300, receiver_y=2400
    ),
    TinyTraceHeader(
        ffid=20,
        coordinate_scalar=-100,
        source_x=100000,
        source_y=200000,
        receiver_x=100000,
        receiver_y=200000,
    ),
    TinyTraceHeader(
        ffid=20,
        coordinate_scalar=-100,
        source_x=100000,
        source_y=200000,
        receiver_x=140000,
        receiver_y=230000,
    ),
    TinyTraceHeader(
        ffid=20,
        coordinate_scalar=-100,
        source_x=100000,
        source_y=200000,
        receiver_x=60000,
        receiver_y=200000,
    ),
    TinyTraceHeader(
        ffid=20,
        coordinate_scalar=-100,
        source_x=100000,
        source_y=200000,
        receiver_x=100000,
        receiver_y=140000,
    ),
)

DEFAULT_SAMPLE_COUNT = 8
DEFAULT_SAMPLE_INTERVAL_S = 0.004


def build_amplitudes(trace_count: int, sample_count: int) -> np.ndarray:
    """Return deterministic float32 amplitudes that are exact in binary."""
    rows = np.arange(trace_count, dtype=np.float32).reshape(-1, 1) + 1.0
    columns = np.arange(sample_count, dtype=np.float32) * np.float32(0.125)
    return (rows + columns).astype(np.float32)


def write_tiny_segy(
    path: Path,
    headers: Sequence[TinyTraceHeader] = DEFAULT_HEADERS,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    coordinate_units: int = 1,
) -> TinySegyFile:
    """Write a tiny IEEE-float SEG-Y file and return its expected content."""
    interval_us = int(round(sample_interval_s * 1_000_000))
    amplitudes = build_amplitudes(len(headers), sample_count)

    spec = segyio.spec()
    spec.format = int(segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE)
    spec.samples = [index * interval_us / 1000.0 for index in range(sample_count)]
    spec.tracecount = len(headers)

    with segyio.create(str(path), spec) as handle:
        handle.bin[segyio.BinField.Interval] = interval_us
        handle.bin[segyio.BinField.Samples] = sample_count
        for index, header in enumerate(headers):
            handle.header[index].update(
                {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: index + 1,
                    segyio.TraceField.FieldRecord: header.ffid,
                    segyio.TraceField.SourceGroupScalar: header.coordinate_scalar,
                    segyio.TraceField.SourceX: header.source_x,
                    segyio.TraceField.SourceY: header.source_y,
                    segyio.TraceField.GroupX: header.receiver_x,
                    segyio.TraceField.GroupY: header.receiver_y,
                    segyio.TraceField.CoordinateUnits: coordinate_units,
                    segyio.TraceField.TRACE_SAMPLE_COUNT: sample_count,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: interval_us,
                }
            )
            handle.trace[index] = amplitudes[index]

    return TinySegyFile(
        path=path,
        headers=tuple(headers),
        amplitudes=amplitudes,
        sample_count=sample_count,
        sample_interval_s=interval_us / 1_000_000.0,
        coordinate_units=coordinate_units,
    )
