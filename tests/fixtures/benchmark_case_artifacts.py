"""Synthetic prepared-partition and mask artifacts for benchmark-case tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import RANDOM_TRACE_MASK_KIND
from seis_interp.processing.trace_splits import TEST_SPLIT, WHOLE_FFID_SPLIT_SCOPE

DATASET_ID = "synthetic_benchmark"
CONFIG_SOURCE = "studies/synthetic/config.yaml"
MASK_MISSING_FRACTION = 0.5
RANDOM_SEED = 42


def prepare_benchmark_case_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create one tiny interim dataset, prepared partition, and mask artifact."""
    source_path = tmp_path / "synthetic.sgy"
    source_path.write_bytes(b"synthetic SEG-Y placeholder")
    ffid_count = 8
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
        dataset_id=DATASET_ID,
        selection={"ffid_scope": "all"},
    )

    processed_dir = tmp_path / "processed"
    prepare_baseline_dataset(
        interim_dir,
        processed_dir,
        holdout_fraction=0.5,
        validation_fraction_of_holdout=0.5,
        random_seed=RANDOM_SEED,
        split_scope=WHOLE_FFID_SPLIT_SCOPE,
        config_source=CONFIG_SOURCE,
    )

    mask_dir = processed_dir / "masks" / "test-random-trace"
    prepare_interpolation_mask(
        interim_dir,
        processed_dir,
        mask_dir,
        partition=TEST_SPLIT,
        kind=RANDOM_TRACE_MASK_KIND,
        missing_fraction=MASK_MISSING_FRACTION,
        random_seed=RANDOM_SEED,
        config_source=CONFIG_SOURCE,
    )
    return interim_dir, processed_dir, mask_dir
