from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from seis_interp.processing.trace_splits import SPLIT_COLUMN, TEST_SPLIT


def _write_interim_survey(tmp_path: Path) -> Path:
    source_path = tmp_path / "synthetic.sgy"
    source_path.write_bytes(b"synthetic SEG-Y placeholder")
    ffid_count = 20
    traces_per_ffid = 2
    trace_count = ffid_count * traces_per_ffid
    trace_indices = np.arange(trace_count, dtype=np.int64)
    source_x_m = trace_indices.astype(np.float64) * 10.0
    source_y_m = trace_indices.astype(np.float64) * 20.0
    trace_table = pd.DataFrame(
        {
            "trace_index": trace_indices,
            "ffid": np.repeat(
                np.arange(100, 100 + ffid_count, dtype=np.int64),
                traces_per_ffid,
            ),
            "cmp_x_m": trace_indices.astype(np.float64),
            "cmp_y_m": trace_indices.astype(np.float64) * 2.0,
            "offset_m": trace_indices.astype(np.float64) + 100.0,
            "azimuth_deg": trace_indices.astype(np.float64) * 5.0,
            "sample_interval_s": np.full(trace_count, 0.008),
            "source_x_m": source_x_m,
            "source_y_m": source_y_m,
            "receiver_x_m": source_x_m + 100.0,
            "receiver_y_m": source_y_m + 200.0,
        }
    )
    amplitudes = np.arange(1, trace_count * 4 + 1, dtype=np.float32).reshape(trace_count, 4)
    interim_dir = tmp_path / "interim"
    write_interim_trace_dataset(
        output_dir=interim_dir,
        trace_table=trace_table,
        amplitudes=amplitudes,
        time_s=np.arange(4, dtype=np.float64) * 0.008,
        source_path=source_path,
        dataset_id="seg_c3_na",
        selection={"ffid_scope": "all"},
    )
    return interim_dir


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        name: file_sha256(directory / name)
        for name in ("trace_split.parquet", "normalization.json", "preparation.json")
    }


def test_partition_and_interpolation_masks_are_separate_artifacts(tmp_path: Path) -> None:
    interim_dir = _write_interim_survey(tmp_path)
    processed_dir = tmp_path / "processed"
    prepare_baseline_dataset(
        interim_dir,
        processed_dir,
        holdout_fraction=0.75,
        validation_fraction_of_holdout=0.25,
        random_seed=42,
        split_scope="whole_ffid",
        config_source="studies/synthetic/config.yaml",
    )
    partition_hashes = _artifact_hashes(processed_dir)
    split_table = pd.read_parquet(processed_dir / "trace_split.parquet")
    expected_rows = set(
        split_table.loc[split_table[SPLIT_COLUMN].eq(TEST_SPLIT), "array_row"].tolist()
    )

    first_dir = processed_dir / "masks" / "random-trace-seed-1"
    second_dir = processed_dir / "masks" / "random-trace-seed-2"
    prepare_interpolation_mask(
        interim_dir,
        processed_dir,
        first_dir,
        partition=TEST_SPLIT,
        kind=RANDOM_TRACE_MASK_KIND,
        missing_fraction=0.5,
        random_seed=1,
        config_source="studies/synthetic/config.yaml",
    )
    prepare_interpolation_mask(
        interim_dir,
        processed_dir,
        second_dir,
        partition=TEST_SPLIT,
        kind=RANDOM_TRACE_MASK_KIND,
        missing_fraction=0.5,
        random_seed=2,
        config_source="studies/synthetic/config.yaml",
    )
    first_mask, first_metadata = load_interpolation_mask(first_dir)
    second_mask, _ = load_interpolation_mask(second_dir)

    assert set(first_mask["array_row"]) == expected_rows
    assert first_mask.columns.tolist() == ["array_row", OBSERVATION_ROLE_COLUMN]
    assert split_table.columns.tolist() == ["array_row", SPLIT_COLUMN]
    assert set(first_mask[OBSERVATION_ROLE_COLUMN]) == {
        OBSERVED_ROLE,
        EVALUATION_TARGET_ROLE,
    }
    assert first_metadata["partition"] == TEST_SPLIT
    assert first_metadata["candidate_trace_count"] == len(expected_rows)
    assert not first_mask.equals(second_mask)
    assert _artifact_hashes(processed_dir) == partition_hashes

    whole_ffid_dir = processed_dir / "masks" / "whole-ffid"
    prepare_interpolation_mask(
        interim_dir,
        processed_dir,
        whole_ffid_dir,
        partition=TEST_SPLIT,
        kind=RANDOM_WHOLE_FFID_MASK_KIND,
        missing_fraction=0.5,
        random_seed=3,
        config_source="studies/synthetic/config.yaml",
    )
    whole_ffid_mask, _ = load_interpolation_mask(whole_ffid_dir)
    trace_table = pd.read_parquet(interim_dir / "traces.parquet")
    with_ffids = whole_ffid_mask.merge(
        trace_table[["array_row", "ffid"]],
        on="array_row",
        validate="one_to_one",
    )

    assert with_ffids.groupby("ffid")[OBSERVATION_ROLE_COLUMN].nunique().eq(1).all()
    assert _artifact_hashes(processed_dir) == partition_hashes
