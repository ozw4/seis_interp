"""Resolve disjoint pilot teacher regions inside the fixed suite's train pool."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seis_interp.data.c3_benchmark_artifacts import load_c3_geometry_inputs
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_supervised_source
from seis_interp.data.c3_benchmark_suite import (
    VerifiedC3BenchmarkSuite,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.processing.c3_crop_selection import resolve_c3_crop


def resolve_c3_first_results_ccnet_regions(
    suite_dir: Path,
    *,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
    fit_source_line_range: tuple[int, int] = (0, 8),
    selection_source_line_range: tuple[int, int] = (8, 10),
    spatial_lengths: tuple[int, int, int] = (32, 8, 32),
) -> dict[str, object]:
    """Use geometry to choose starts, then verify rows/time and fit-only RMS.

    No partition, evaluation crop, mask, or waveform artifact is regenerated.
    Alternate ranges are explicit caller choices, never an automatic fallback.
    """
    suite_dir = Path(suite_dir)
    verified = verified_suite
    suite = load_c3_benchmark_input_manifest(
        suite_dir, dimensions=dimensions, verified_suite=verified
    )
    table, time_s = load_c3_geometry_inputs(suite_path(suite_dir, suite["interim"]))
    allowed = np.load(suite_dir / suite["train_pool"]["file"], allow_pickle=False)
    resolved = {}
    for name, line_range in (
        ("fit", fit_source_line_range),
        ("selection", selection_source_line_range),
    ):
        index, record = resolve_c3_crop(
            table,
            time_s,
            time_range=dimensions.time_range,
            source_line_range=line_range,
            spatial_lengths=spatial_lengths,
        )
        if not np.isin(index["array_row"].to_numpy(), allowed).all():
            raise ValueError(f"CCNet {name} region is outside the authorized canonical train pool")
        resolved[name] = record
    source = load_c3_benchmark_supervised_source(
        suite_dir,
        fit_region=resolved["fit"]["selection"],
        selection_region=resolved["selection"]["selection"],
        dimensions=dimensions,
        verified_suite=verified,
    )
    return {
        "fit_region": source.fit.selection,
        "selection_region": source.selection.selection,
        "geometry_resolution": resolved,
        "authorized_train_trace_count": int(len(allowed)),
        "fit_trace_count": int(source.fit.array_rows.size),
        "selection_trace_count": int(source.selection.array_rows.size),
        "disjoint_physical_trace_sets": True,
        "time_samples": list(dimensions.time_range),
        "amplitude_rms": source.amplitude_rms,
        "amplitude_rms_scope": "fit_region_complete_labels",
        "source_inputs_lock": source.inputs_lock,
    }
