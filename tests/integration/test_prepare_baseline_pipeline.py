from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.normalization import read_normalization_parameters

TRACE_COUNT = 20
SAMPLE_COUNT = 4
HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION_OF_HOLDOUT = 0.25
RANDOM_SEED = 42


def _write_interim_dataset(tmp_path: Path) -> Path:
    source_path = tmp_path / "source.sgy"
    source_path.write_bytes(b"synthetic SEG-Y placeholder for baseline preparation")
    interim_dir = tmp_path / "interim"
    trace_indices = np.arange(TRACE_COUNT, dtype=np.int64)
    trace_table = pd.DataFrame(
        {
            "trace_index": trace_indices,
            "ffid": np.full(TRACE_COUNT, 2348, dtype=np.int64),
            "cmp_x_m": trace_indices.astype(np.float64),
            "cmp_y_m": trace_indices.astype(np.float64) * 2.0 + 100.0,
            "offset_m": trace_indices.astype(np.float64) * 5.0 + 500.0,
            "azimuth_deg": trace_indices.astype(np.float64) * 10.0,
            "sample_interval_s": np.full(TRACE_COUNT, 0.008),
        }
    )
    amplitudes = np.repeat(
        np.arange(1, TRACE_COUNT + 1, dtype=np.float32)[:, np.newaxis],
        SAMPLE_COUNT,
        axis=1,
    )
    time_s = np.arange(SAMPLE_COUNT, dtype=np.float64) * 0.008
    write_interim_trace_dataset(
        output_dir=interim_dir,
        trace_table=trace_table,
        amplitudes=amplitudes,
        time_s=time_s,
        source_path=source_path,
        dataset_id="seg_c3_na",
        selection={"ffid": 2348, "expected_trace_count": TRACE_COUNT},
    )
    return interim_dir


def _prepare(interim_dir: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, object]:
    return prepare_baseline_dataset(
        interim_dir,
        output_dir,
        holdout_fraction=HOLDOUT_FRACTION,
        validation_fraction_of_holdout=VALIDATION_FRACTION_OF_HOLDOUT,
        random_seed=RANDOM_SEED,
        overwrite=overwrite,
    )


def test_writes_only_the_three_processed_dataset_files(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    _prepare(interim_dir, output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "normalization.json",
        "preparation.json",
        "trace_split.parquet",
    ]
    trace_split = pd.read_parquet(output_dir / "trace_split.parquet")
    assert trace_split.columns.tolist() == ["array_row", "split"]


def test_records_the_expected_split_counts(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    summary = _prepare(interim_dir, output_dir)

    assert summary["split_counts"] == {
        "train": 16,
        "validation": 1,
        "test": 3,
    }
    split_counts = (
        pd.read_parquet(output_dir / "trace_split.parquet")["split"].value_counts().to_dict()
    )
    assert split_counts == summary["split_counts"]


def test_same_seed_writes_identical_trace_splits(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    first_output = tmp_path / "processed_first"
    second_output = tmp_path / "processed_second"

    _prepare(interim_dir, first_output)
    _prepare(interim_dir, second_output)

    pd.testing.assert_frame_equal(
        pd.read_parquet(first_output / "trace_split.parquet"),
        pd.read_parquet(second_output / "trace_split.parquet"),
    )


def test_normalization_is_fit_from_training_rows_only(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    _prepare(interim_dir, output_dir)

    trace_table = pd.read_parquet(interim_dir / "traces.parquet")
    amplitudes = np.load(interim_dir / "amplitudes.npy")
    split_table = pd.read_parquet(output_dir / "trace_split.parquet")
    training_rows = split_table.loc[split_table["split"] == "train", "array_row"].to_numpy()
    training_table = trace_table.set_index("array_row").loc[training_rows]
    parameters = read_normalization_parameters(output_dir / "normalization.json")

    expected_min = (
        0.0,
        *(training_table[["cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg"]].min()),
    )
    expected_max = (
        (SAMPLE_COUNT - 1) * 0.008,
        *(training_table[["cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg"]].max()),
    )
    expected_rms = float(np.sqrt(np.mean(amplitudes[training_rows].astype(np.float64) ** 2)))

    np.testing.assert_allclose(parameters.coordinate_min, expected_min)
    np.testing.assert_allclose(parameters.coordinate_max, expected_max)
    assert parameters.amplitude_rms == pytest.approx(expected_rms)
    assert not np.isclose(
        expected_rms,
        np.sqrt(np.mean(amplitudes.astype(np.float64) ** 2)),
    )


def test_preparation_records_relative_provenance_and_metadata_hash(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"
    expected_metadata_hash = hashlib.sha256((interim_dir / "dataset.json").read_bytes()).hexdigest()

    summary = _prepare(interim_dir, output_dir)
    preparation_text = (output_dir / "preparation.json").read_text(encoding="utf-8")

    assert summary["source_file"] == "source.sgy"
    assert summary["input_dataset_metadata_sha256"] == expected_metadata_hash
    assert str(tmp_path) not in preparation_text


def test_rejects_a_non_empty_output_without_overwrite(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _prepare(interim_dir, output_dir)

    assert marker.read_text(encoding="utf-8") == "keep me"
    assert not (output_dir / "preparation.json").exists()


def test_overwrite_replaces_generated_files_and_preserves_unrelated_files(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"
    _prepare(interim_dir, output_dir)
    marker = output_dir / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")
    (output_dir / "preparation.json").write_text("stale", encoding="utf-8")
    (output_dir / "normalization.json").write_text("stale", encoding="utf-8")
    pd.DataFrame({"stale": [True]}).to_parquet(
        output_dir / "trace_split.parquet",
        index=False,
    )

    summary = _prepare(interim_dir, output_dir, overwrite=True)

    assert marker.read_text(encoding="utf-8") == "keep me"
    assert json.loads((output_dir / "preparation.json").read_text(encoding="utf-8")) == summary
    assert pd.read_parquet(output_dir / "trace_split.parquet").columns.tolist() == [
        "array_row",
        "split",
    ]
    read_normalization_parameters(output_dir / "normalization.json")


def test_return_value_matches_preparation_json(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    summary = _prepare(interim_dir, output_dir)

    stored = json.loads((output_dir / "preparation.json").read_text(encoding="utf-8"))
    assert stored == summary


def test_validation_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError):
        prepare_baseline_dataset(
            interim_dir,
            output_dir,
            holdout_fraction=0.0,
            validation_fraction_of_holdout=VALIDATION_FRACTION_OF_HOLDOUT,
            random_seed=RANDOM_SEED,
        )

    assert not output_dir.exists()
