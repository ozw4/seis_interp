"""Small prepared C3 teacher data for CCNet source and pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.trace_splits import C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE
from tests.fixtures.c3_volume_artifacts import make_c3_trace_table

SOURCE_LINE_RANGES = {"train": (0, 2), "validation": (2, 3), "test": (3, 4)}


@dataclass(frozen=True)
class PreparedCCNet5DArtifacts:
    interim: Path
    processed: Path
    fit_region: dict[str, list[int]]
    selection_region: dict[str, list[int]]


def prepare_ccnet5d_artifacts(
    tmp_path: Path,
    *,
    shuffle_tables: bool = False,
    duplicate_fit_trace: bool = False,
    nonfit_value: float | None = None,
    nontrain_value: float | None = None,
) -> PreparedCCNet5DArtifacts:
    """Reuse fixed-grid geometry and optionally change labels before binding preparation."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    geometry = make_c3_trace_table(physical_source_line_indices=(0, 1, 2, 3))
    if duplicate_fit_trace:
        geometry = pd.concat([geometry, geometry.iloc[[0]]], ignore_index=True)
    count = len(geometry)
    table = geometry.drop(columns="array_row").assign(
        trace_index=np.arange(count, dtype=np.int64),
        cmp_x_m=(geometry["source_x_m"] + geometry["receiver_x_m"]) / 2,
        cmp_y_m=(geometry["source_y_m"] + geometry["receiver_y_m"]) / 2,
        offset_m=np.hypot(
            geometry["source_x_m"] - geometry["receiver_x_m"],
            geometry["source_y_m"] - geometry["receiver_y_m"],
        ),
        azimuth_deg=np.zeros(count, dtype=np.float64),
        sample_interval_s=np.full(count, 0.008, dtype=np.float64),
    )
    amplitudes = (
        np.arange(count, dtype=np.float32)[:, None] * 10
        + np.arange(5, dtype=np.float32)[None, :]
        - 2
    )
    source_line = np.searchsorted(np.sort(table["source_x_m"].unique()), table["source_x_m"])
    first_shot = table["source_y_m"].eq(table.groupby("source_x_m")["source_y_m"].transform("min"))
    relative_x = table["receiver_x_m"] - table["source_x_m"]
    relative_y = table["receiver_y_m"] - table["source_y_m"]
    fit_rows = (source_line < 2) & first_shot & relative_x.lt(-60) & relative_y.lt(-2560)
    if nonfit_value is not None:
        amplitudes[~fit_rows] = nonfit_value
    if nontrain_value is not None:
        amplitudes[source_line >= 2] = nontrain_value
    if duplicate_fit_trace:
        amplitudes[-1] = 999999.0
    source_path = tmp_path / "synthetic.sgy"
    source_path.write_bytes(b"synthetic CCNet C3 placeholder")
    interim = tmp_path / "interim"
    write_interim_trace_dataset(
        interim,
        table,
        amplitudes,
        np.arange(5, dtype=np.float64) * 0.008,
        source_path,
        "synthetic_c3_ccnet5d",
        {"ffid_scope": "all"},
    )
    if shuffle_tables:
        trace_path = interim / "traces.parquet"
        pd.read_parquet(trace_path).sample(frac=1.0, random_state=7).to_parquet(
            trace_path, index=False
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
    if shuffle_tables:
        split_path = processed / "trace_split.parquet"
        pd.read_parquet(split_path).sample(frac=1.0, random_state=11).to_parquet(
            split_path, index=False
        )
    fit = {
        "time": [1, 4],
        "source_line": [0, 2],
        "shot_in_line": [0, 1],
        "relative_receiver_x": [0, 2],
        "relative_receiver_y": [0, 3],
    }
    selection = {**fit, "shot_in_line": [1, 3]}
    return PreparedCCNet5DArtifacts(interim, processed, fit, selection)
