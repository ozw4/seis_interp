from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.evaluation.baselines import (
    inverse_distance_weighted_predict,
    nearest_neighbor_predict,
)
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.normalization import (
    normalize_spatial_coordinates,
    read_normalization_parameters,
)
from seis_interp.processing.trace_splits import TRAIN_SPLIT


def _write_synthetic_interim_dataset(directory: Path, source_path: Path) -> None:
    trace_count = 20
    sample_count = 6
    trace_index = np.arange(trace_count)
    trace_table = pd.DataFrame(
        {
            "trace_index": trace_index,
            "ffid": np.full(trace_count, 7),
            "cmp_x_m": 100.0 + 12.0 * trace_index,
            "cmp_y_m": 500.0 + np.square(trace_index),
            "offset_m": 50.0 + 3.0 * trace_index,
            "azimuth_deg": np.mod(350.0 + 2.0 * trace_index, 360.0),
            "sample_interval_s": np.full(trace_count, 0.008),
        }
    )
    amplitudes = np.stack(
        [np.linspace(row + 1.0, row + 2.0, sample_count, dtype=np.float32) for row in trace_index]
    )
    time_s = np.arange(sample_count, dtype=np.float64) * 0.008
    source_path.write_bytes(b"synthetic SEG-Y placeholder")

    write_interim_trace_dataset(
        output_dir=directory,
        trace_table=trace_table,
        amplitudes=amplitudes,
        time_s=time_s,
        source_path=source_path,
        dataset_id="synthetic",
        selection={"ffid": 7, "expected_trace_count": trace_count},
    )


def test_pr3_preprocessing_outputs_feed_both_baselines(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    _write_synthetic_interim_dataset(interim_dir, tmp_path / "synthetic.sgy")

    prepare_baseline_dataset(
        interim_dir,
        processed_dir,
        holdout_fraction=0.2,
        validation_fraction_of_holdout=0.25,
        random_seed=42,
    )

    dataset = load_interim_trace_dataset(interim_dir)
    split_table = pd.read_parquet(processed_dir / "trace_split.parquet")
    parameters = read_normalization_parameters(processed_dir / "normalization.json")
    normalized = normalize_spatial_coordinates(dataset.trace_table, parameters)

    assert split_table["array_row"].is_unique
    assert split_table["split"].value_counts().to_dict() == {
        "train": 16,
        "validation": 1,
        "test": 3,
    }
    assert normalized.shape == (len(dataset.trace_table), 5)
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized[:, 3:] >= -1.0)
    assert np.all(normalized[:, 3:] <= 1.0)
    np.testing.assert_allclose(np.sum(normalized[:, 3:] ** 2, axis=1), 1.0)

    train_rows = split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"].to_numpy(
        dtype=np.int64
    )
    held_out_rows = split_table.loc[split_table["split"] != TRAIN_SPLIT, "array_row"].to_numpy(
        dtype=np.int64
    )

    assert set(train_rows).isdisjoint(held_out_rows)
    assert len(train_rows) + len(held_out_rows) == len(dataset.trace_table)

    coordinates_by_array_row = np.empty_like(normalized)
    coordinates_by_array_row[dataset.trace_table["array_row"].to_numpy(dtype=np.int64)] = normalized
    train_coordinates = coordinates_by_array_row[train_rows]
    held_out_coordinates = coordinates_by_array_row[held_out_rows]
    train_amplitudes = dataset.amplitudes[train_rows]
    expected_shape = dataset.amplitudes[held_out_rows].shape
    expected_rms = float(np.sqrt(np.mean(train_amplitudes.astype(np.float64) ** 2)))

    assert train_coordinates.shape == (len(train_rows), 5)
    assert held_out_coordinates.shape == (len(held_out_rows), 5)
    np.testing.assert_allclose(parameters.amplitude_rms, expected_rms)

    nearest = nearest_neighbor_predict(
        train_coordinates,
        train_amplitudes,
        held_out_coordinates,
    )
    idw = inverse_distance_weighted_predict(
        train_coordinates,
        train_amplitudes,
        held_out_coordinates,
    )

    assert nearest.shape == expected_shape
    assert idw.shape == expected_shape
    assert np.all(np.isfinite(nearest))
    assert np.all(np.isfinite(idw))
