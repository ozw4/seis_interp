"""Read FORGE waveforms in audited order without rejecting nonfinite QC inputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import segyio


def read_forge_gather(path: Path, headers: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Validate trace count/time metadata before reading little-endian IEEE data."""
    with segyio.open(
        str(path), mode="r", strict=False, ignore_geometry=True, endian="little"
    ) as handle:
        if handle.bin[segyio.BinField.Format] != 5:
            raise ValueError("expected IEEE float32")
        if handle.tracecount != len(headers) or not np.array_equal(
            headers.trace_index.to_numpy(), np.arange(handle.tracecount)
        ):
            raise ValueError("header rows must cover this entire gather in file order")
        dt_us = int(handle.bin[segyio.BinField.Interval])
        if (
            not headers.sample_interval_us.eq(dt_us).all()
            or not headers.sample_count.eq(len(handle.samples)).all()
        ):
            raise ValueError("waveform time metadata differs from header audit")
        values = handle.trace.raw[:].copy()
    return values, dt_us / 1e6
