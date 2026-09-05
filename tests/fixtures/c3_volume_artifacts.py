"""Synthetic fixed-grid C3 geometry for volume-index tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.benchmark_case_store import load_benchmark_case
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import RANDOM_TRACE_MASK_KIND
from seis_interp.processing.trace_splits import (
    C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

RECEIVER_X_OFFSETS_M = np.arange(-140.0, 141.0, 40.0)
RECEIVER_Y_OFFSETS_M = np.arange(-2680.0, 1.0, 40.0)
SOURCE_LINE_RANGES = {
    TRAIN_SPLIT: (0, 1),
    VALIDATION_SPLIT: (1, 2),
    TEST_SPLIT: (2, 4),
}
VOLUME_SOURCE_LINE_RANGE = (2, 4)
VOLUME_SHOT_IN_LINE_RANGE = (0, 2)

_FFIDS_BY_PHYSICAL_LINE = (
    (91, 12, 77),
    (43, 8, 105),
    (68, 31, 97),
    (24, 113, 56),
    (84, 19, 102),
)


def make_c3_trace_table(
    *,
    physical_source_line_indices: tuple[int, ...] = (0, 1),
    omitted_shots: frozenset[tuple[int, int]] = frozenset(),
) -> pd.DataFrame:
    """Return physical C3 source lines with three fixed-grid shots each."""
    records: list[dict[str, float | int]] = []
    array_row = 0
    for physical_line in physical_source_line_indices:
        source_x_m = 1000.0 + 160.0 * physical_line
        source_y_start_m = 40.0 * (physical_line % 2)
        for shot in range(3):
            if (physical_line, shot) in omitted_shots:
                continue
            source_y_m = source_y_start_m + 80.0 * shot
            for relative_x_m in RECEIVER_X_OFFSETS_M:
                for relative_y_m in RECEIVER_Y_OFFSETS_M:
                    records.append(
                        {
                            "array_row": array_row,
                            "ffid": _FFIDS_BY_PHYSICAL_LINE[physical_line][shot],
                            "source_x_m": source_x_m,
                            "source_y_m": source_y_m,
                            "receiver_x_m": source_x_m + relative_x_m,
                            "receiver_y_m": source_y_m + relative_y_m,
                        }
                    )
                    array_row += 1
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class PreparedC3VolumeArtifacts:
    interim_dir: Path
    processed_dir: Path
    mask_dir: Path
    case_dir: Path
    source_line_range: tuple[int, int]
    shot_in_line_range: tuple[int, int]


def prepare_c3_volume_artifacts(
    tmp_path: Path,
    *,
    physical_source_line_indices: tuple[int, ...] = (0, 1, 2, 3),
    omitted_shots: frozenset[tuple[int, int]] = frozenset(),
) -> PreparedC3VolumeArtifacts:
    """Create a source-block partition, random mask, and case for a C3 crop."""
    geometry = make_c3_trace_table(
        physical_source_line_indices=physical_source_line_indices,
        omitted_shots=omitted_shots,
    )
    trace_count = len(geometry)
    source = tmp_path / "synthetic.sgy"
    source.write_bytes(b"synthetic SEG-Y placeholder")
    trace_table = geometry.assign(
        trace_index=np.arange(trace_count, dtype=np.int64),
        cmp_x_m=(geometry["source_x_m"] + geometry["receiver_x_m"]) / 2.0,
        cmp_y_m=(geometry["source_y_m"] + geometry["receiver_y_m"]) / 2.0,
        offset_m=np.hypot(
            geometry["source_x_m"] - geometry["receiver_x_m"],
            geometry["source_y_m"] - geometry["receiver_y_m"],
        ),
        azimuth_deg=np.zeros(trace_count, dtype=np.float64),
        sample_interval_s=np.full(trace_count, 0.008, dtype=np.float64),
    ).drop(columns="array_row")
    amplitudes = np.arange(trace_count * 4, dtype=np.float32).reshape(trace_count, 4)

    interim = tmp_path / "interim"
    write_interim_trace_dataset(
        interim,
        trace_table,
        amplitudes,
        np.arange(4, dtype=np.float64) * 0.008,
        source,
        "synthetic_c3",
        {"ffid_scope": "all"},
    )
    processed = tmp_path / "processed"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=None,
        validation_fraction_of_holdout=None,
        random_seed=42,
        split_scope=C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
        source_line_ranges=SOURCE_LINE_RANGES,
        config_source="studies/synthetic/config.yaml",
    )
    mask = processed / "masks" / "test-random-trace"
    prepare_interpolation_mask(
        interim,
        processed,
        mask,
        partition=TEST_SPLIT,
        kind=RANDOM_TRACE_MASK_KIND,
        missing_fraction=0.5,
        random_seed=42,
        config_source="studies/synthetic/config.yaml",
    )
    case = processed / "cases" / "synthetic-case"
    prepare_benchmark_case(
        interim,
        processed,
        mask,
        case,
        case_id="synthetic_case",
        config_source="studies/synthetic/config.yaml",
    )
    load_benchmark_case(case)
    return PreparedC3VolumeArtifacts(
        interim,
        processed,
        mask,
        case,
        VOLUME_SOURCE_LINE_RANGE,
        VOLUME_SHOT_IN_LINE_RANGE,
    )
