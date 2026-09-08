"""Connect mask-independent C3 geometry/crop QC to immutable artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seis_interp.data.c3_benchmark_artifacts import (
    benchmark_file_record,
    load_c3_geometry_inputs,
    write_benchmark_json,
)
from seis_interp.processing.c3_crop_selection import resolve_c3_crop
from seis_interp.processing.c3_crop_signal_qc import summarize_c3_crop_signal
from seis_interp.processing.c3_geometry_qc import summarize_c3_geometry


def qc_c3_geometry(
    interim_dir: Path,
    output_dir: Path,
    *,
    requested_inclusive: tuple[int, int] = (25, 40),
    time_range: tuple[int, int] = (0, 384),
    original_number_column: str | None = None,
) -> dict[str, object]:
    """Write measured line/FFID mapping and geometry statistics, without a mask."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    table, times = load_c3_geometry_inputs(interim_dir)
    mapping, summary = summarize_c3_geometry(
        table,
        times,
        requested_inclusive=requested_inclusive,
        time_range=time_range,
        original_number_column=original_number_column,
    )
    summary["inputs"] = {
        name: benchmark_file_record(interim_dir / name, output_dir)
        for name in ("traces.parquet", "time_s.npy")
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    mapping.to_parquet(output_dir / "sail_line_mapping.parquet", index=False)
    write_benchmark_json(output_dir / "geometry.json", summary)
    return summary


def qc_c3_crop(
    interim_dir: Path,
    output_dir: Path,
    *,
    source_line_range: tuple[int, int],
    time_range: tuple[int, int],
    spatial_lengths: tuple[int, int, int],
    explicit_ranges: dict[str, object] | None = None,
    plots: bool = False,
) -> dict[str, object]:
    """Resolve integer ranges and stream only the chosen crop's signal samples."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    table, times = load_c3_geometry_inputs(interim_dir)
    index, summary = resolve_c3_crop(
        table,
        times,
        time_range=time_range,
        source_line_range=source_line_range,
        spatial_lengths=spatial_lengths,
        explicit_ranges=explicit_ranges,
    )
    amplitudes = np.load(interim_dir / "amplitudes.npy", mmap_mode="r", allow_pickle=False)
    summary["signal"] = summarize_c3_crop_signal(
        amplitudes, index["array_row"].to_numpy(), times, time_range=time_range
    )
    summary["inputs"] = {
        name: benchmark_file_record(interim_dir / name, output_dir)
        for name in ("traces.parquet", "time_s.npy")
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    index.to_parquet(output_dir / "crop_index.parquet", index=False)
    if plots:
        from seis_interp.visualization.c3_crop_qc import plot_c3_crop_qc

        summary["plots"] = plot_c3_crop_qc(amplitudes, index, times, summary, output_dir)
    write_benchmark_json(output_dir / "crop.json", summary)
    if not summary["signal"]["ok"]:
        raise ValueError(f"crop signal QC failed; see {output_dir / 'crop.json'}")
    return summary
