"""Synthetic fixed-grid C3 geometry for volume-index tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.benchmark_case_store import load_benchmark_case
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.prepared_partition import TRACE_SPLIT_FILE_NAME
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import RANDOM_TRACE_MASK_KIND
from seis_interp.processing.trace_splits import TEST_SPLIT, WHOLE_FFID_SPLIT_SCOPE

RECEIVER_X_OFFSETS_M = np.arange(-140.0, 141.0, 40.0)
RECEIVER_Y_OFFSETS_M = np.arange(-2680.0, 1.0, 40.0)


def make_c3_trace_table() -> pd.DataFrame:
    """Return two staggered source lines with three complete shots each."""
    records: list[dict[str, float | int]] = []
    ffids = ((91, 12, 77), (43, 8, 105))
    source_y_lines = ((0.0, 80.0, 160.0), (40.0, 120.0, 200.0))
    array_row = 0
    for line, source_x_m in enumerate((1000.0, 2000.0)):
        for shot, source_y_m in enumerate(source_y_lines[line]):
            for relative_x_m in RECEIVER_X_OFFSETS_M:
                for relative_y_m in RECEIVER_Y_OFFSETS_M:
                    records.append(
                        {
                            "array_row": array_row,
                            "ffid": ffids[line][shot],
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


def prepare_c3_volume_artifacts(tmp_path: Path) -> PreparedC3VolumeArtifacts:
    """Create current interim, partition, mask, and case artifacts for one dense shot."""
    geometry = make_c3_trace_table()
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
        holdout_fraction=0.5,
        validation_fraction_of_holdout=0.5,
        random_seed=42,
        split_scope=WHOLE_FFID_SPLIT_SCOPE,
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

    stored_traces = pd.read_parquet(interim / "traces.parquet")
    splits = pd.read_parquet(processed / TRACE_SPLIT_FILE_NAME)
    mask_table, _ = load_interpolation_mask(mask)
    test_rows = splits.loc[splits["split"].eq(TEST_SPLIT), "array_row"]
    candidates = stored_traces[stored_traces["array_row"].isin(test_rows)].merge(
        mask_table, on="array_row", validate="one_to_one"
    )
    for (source_x, source_y), shot in candidates.groupby(["source_x_m", "source_y_m"]):
        if shot["observation_role"].nunique() == 2:
            source_x_values = np.sort(stored_traces["source_x_m"].unique())
            source_line = int(np.searchsorted(source_x_values, source_x))
            source_y_values = np.sort(
                stored_traces.loc[stored_traces["source_x_m"].eq(source_x), "source_y_m"].unique()
            )
            shot_index = int(np.searchsorted(source_y_values, source_y))
            return PreparedC3VolumeArtifacts(
                interim,
                processed,
                mask,
                case,
                (source_line, source_line + 1),
                (shot_index, shot_index + 1),
            )
    raise RuntimeError("synthetic test partition did not produce a mixed-role dense shot")
