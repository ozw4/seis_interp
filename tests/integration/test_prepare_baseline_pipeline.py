from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.cli import main
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines import prepare_baseline as prepare_baseline_pipeline
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.normalization import read_normalization_parameters
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig
from seis_interp.processing.trace_splits import EXCLUDED_SPLIT

TRACE_COUNT = 20
SAMPLE_COUNT = 4
HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION_OF_HOLDOUT = 0.25
RANDOM_SEED = 42
CONFIG_SOURCE = "studies/synthetic/config.yaml"
COORDINATE_NORMALIZATION_METHOD = "train_minmax_linear_plus_azimuth_sin_cos"
EXPECTED_TRAIN_ROWS = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 16, 17, 18, 19],
    dtype=np.int64,
)
EXPECTED_HELD_OUT_ROWS = np.asarray([7, 9, 14, 15], dtype=np.int64)
EXPECTED_TRAIN_RMS = 11.792476415070755


def test_prepared_partition_file_contract_remains_public_from_pipeline() -> None:
    assert prepare_baseline_pipeline.OUTPUT_FILE_NAMES == PREPARED_FILE_NAMES
    assert prepare_baseline_pipeline.TRACE_SPLIT_FILE_NAME == "trace_split.parquet"
    assert prepare_baseline_pipeline.NORMALIZATION_FILE_NAME == "normalization.json"
    assert prepare_baseline_pipeline.PREPARATION_FILE_NAME == "preparation.json"


def _write_interim_dataset(tmp_path: Path) -> Path:
    source_path = tmp_path / "source.sgy"
    source_path.write_bytes(b"synthetic SEG-Y placeholder for baseline preparation")
    interim_dir = tmp_path / "interim"
    trace_indices = np.arange(TRACE_COUNT, dtype=np.int64)
    cmp_x_m = trace_indices.astype(np.float64)
    cmp_y_m = trace_indices.astype(np.float64) * 2.0 + 100.0
    offset_m = trace_indices.astype(np.float64) * 5.0 + 500.0
    azimuth_deg = trace_indices.astype(np.float64) * 10.0
    cmp_x_m[EXPECTED_HELD_OUT_ROWS] = 1.0e6
    cmp_y_m[EXPECTED_HELD_OUT_ROWS] = -1.0e6
    offset_m[EXPECTED_HELD_OUT_ROWS] = 2.0e6
    azimuth_deg[EXPECTED_HELD_OUT_ROWS] = [0.0, 90.0, 180.0, 270.0]
    trace_table = pd.DataFrame(
        {
            "trace_index": trace_indices,
            "ffid": np.full(TRACE_COUNT, 2348, dtype=np.int64),
            "cmp_x_m": cmp_x_m,
            "cmp_y_m": cmp_y_m,
            "offset_m": offset_m,
            "azimuth_deg": azimuth_deg,
            "sample_interval_s": np.full(TRACE_COUNT, 0.008),
        }
    )
    amplitudes = np.repeat(
        np.arange(1, TRACE_COUNT + 1, dtype=np.float32)[:, np.newaxis],
        SAMPLE_COUNT,
        axis=1,
    )
    amplitudes[EXPECTED_HELD_OUT_ROWS] = 1.0e6
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


def _prepare(
    interim_dir: Path,
    output_dir: Path,
    *,
    split_scope: str = "global",
    trace_amplitude_filter: TraceAmplitudeFilterConfig | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    return prepare_baseline_dataset(
        interim_dir,
        output_dir,
        holdout_fraction=HOLDOUT_FRACTION,
        validation_fraction_of_holdout=VALIDATION_FRACTION_OF_HOLDOUT,
        random_seed=RANDOM_SEED,
        split_scope=split_scope,
        trace_amplitude_filter=trace_amplitude_filter,
        config_source=CONFIG_SOURCE,
        overwrite=overwrite,
    )


def _write_multi_ffid_interim_dataset(tmp_path: Path, *, ffid_count: int = 2) -> Path:
    source_path = tmp_path / "multi-source.sgy"
    source_path.write_bytes(b"synthetic multi-FFID SEG-Y placeholder")
    interim_dir = tmp_path / "multi-ffid-interim"
    traces_per_ffid = 20
    trace_count = ffid_count * traces_per_ffid
    trace_indices = np.arange(trace_count, dtype=np.int64)
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
        }
    )
    amplitudes = np.arange(1, trace_count * SAMPLE_COUNT + 1, dtype=np.float32).reshape(
        trace_count, SAMPLE_COUNT
    )
    write_interim_trace_dataset(
        output_dir=interim_dir,
        trace_table=trace_table,
        amplitudes=amplitudes,
        time_s=np.arange(SAMPLE_COUNT, dtype=np.float64) * 0.008,
        source_path=source_path,
        dataset_id="seg_c3_na",
        selection={"ffid_scope": "all", "include_incomplete_ffids": True},
    )
    return interim_dir


def _write_c3_source_line_interim_dataset(tmp_path: Path) -> Path:
    records: list[dict[str, float | int]] = []
    trace_index = 0
    for source_line, source_x_m in enumerate((0.0, 160.0, 320.0)):
        source_y_origin_m = 40.0 if source_line % 2 else 0.0
        for shot in range(2):
            source_y_m = source_y_origin_m + shot * 80.0
            ffid = 100 + source_line * 2 + shot
            for relative_receiver_x_m in (-140.0, -100.0):
                receiver_x_m = source_x_m + relative_receiver_x_m
                receiver_y_m = source_y_m - 2680.0
                records.append(
                    {
                        "trace_index": trace_index,
                        "ffid": ffid,
                        "source_x_m": source_x_m,
                        "source_y_m": source_y_m,
                        "receiver_x_m": receiver_x_m,
                        "receiver_y_m": receiver_y_m,
                        "cmp_x_m": (source_x_m + receiver_x_m) / 2.0,
                        "cmp_y_m": (source_y_m + receiver_y_m) / 2.0,
                        "offset_m": float(
                            np.hypot(
                                source_x_m - receiver_x_m,
                                source_y_m - receiver_y_m,
                            )
                        ),
                        "azimuth_deg": 0.0,
                        "sample_interval_s": 0.008,
                    }
                )
                trace_index += 1

    source_path = tmp_path / "c3-source-lines.sgy"
    source_path.write_bytes(b"synthetic C3 source-line SEG-Y placeholder")
    interim_dir = tmp_path / "c3-source-line-interim"
    trace_table = pd.DataFrame.from_records(records)
    amplitudes = np.arange(1, len(trace_table) * SAMPLE_COUNT + 1, dtype=np.float32).reshape(
        len(trace_table), SAMPLE_COUNT
    )
    write_interim_trace_dataset(
        output_dir=interim_dir,
        trace_table=trace_table,
        amplitudes=amplitudes,
        time_s=np.arange(SAMPLE_COUNT, dtype=np.float64) * 0.008,
        source_path=source_path,
        dataset_id="seg_c3_na",
        selection={"ffid_scope": "all", "include_incomplete_ffids": True},
    )
    return interim_dir


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
    assert summary["split_scope"] == "global"
    assert summary["ffid_count"] == 1


def test_per_ffid_scope_writes_each_split_for_every_ffid(tmp_path: Path) -> None:
    interim_dir = _write_multi_ffid_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    summary = _prepare(interim_dir, output_dir, split_scope="per_ffid")

    split_table = pd.read_parquet(output_dir / "trace_split.parquet")
    trace_table = pd.read_parquet(interim_dir / "traces.parquet")
    joined = trace_table[["array_row", "ffid"]].merge(
        split_table,
        on="array_row",
        validate="one_to_one",
    )
    counts = joined.groupby(["ffid", "split"]).size().unstack(fill_value=0)
    assert counts.to_dict(orient="index") == {
        100: {"test": 3, "train": 16, "validation": 1},
        101: {"test": 3, "train": 16, "validation": 1},
    }
    assert summary["split_scope"] == "per_ffid"
    assert summary["ffid_count"] == 2
    assert summary["split_counts"] == {"train": 32, "validation": 2, "test": 6}


def test_whole_ffid_scope_assigns_each_ffid_to_one_split(tmp_path: Path) -> None:
    interim_dir = _write_multi_ffid_interim_dataset(tmp_path, ffid_count=20)
    output_dir = tmp_path / "processed"

    summary = _prepare(interim_dir, output_dir, split_scope="whole_ffid")

    split_table = pd.read_parquet(output_dir / "trace_split.parquet")
    trace_table = pd.read_parquet(interim_dir / "traces.parquet")
    joined = trace_table[["array_row", "ffid"]].merge(
        split_table,
        on="array_row",
        validate="one_to_one",
    )
    assert joined.groupby("ffid")["split"].nunique().eq(1).all()
    assert joined.groupby("split")["ffid"].nunique().to_dict() == {
        "test": 3,
        "train": 16,
        "validation": 1,
    }
    assert summary["split_scope"] == "whole_ffid"
    assert summary["ffid_count"] == 20
    assert summary["ffid_split_counts"] == {"train": 16, "validation": 1, "test": 3}
    assert summary["split_counts"] == {"train": 320, "validation": 20, "test": 60}


def test_c3_source_line_blocks_assign_each_complete_line_to_configured_split(
    tmp_path: Path,
) -> None:
    interim_dir = _write_c3_source_line_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"
    source_line_ranges = {
        "train": (0, 1),
        "validation": (1, 2),
        "test": (2, 3),
    }

    summary = prepare_baseline_dataset(
        interim_dir,
        output_dir,
        holdout_fraction=None,
        validation_fraction_of_holdout=None,
        random_seed=RANDOM_SEED,
        split_scope="c3_source_line_blocks",
        source_line_ranges=source_line_ranges,
        config_source=CONFIG_SOURCE,
    )

    split_table = pd.read_parquet(output_dir / "trace_split.parquet")
    trace_table = pd.read_parquet(interim_dir / "traces.parquet")
    joined = trace_table.merge(split_table, on="array_row", validate="one_to_one")
    assert joined.groupby("source_x_m")["split"].unique().map(list).to_dict() == {
        0.0: ["train"],
        160.0: ["validation"],
        320.0: ["test"],
    }
    assert joined.groupby("ffid")["split"].nunique().eq(1).all()
    assert summary["source_line_ranges"] == {
        "train": [0, 1],
        "validation": [1, 2],
        "test": [2, 3],
    }
    assert summary["ffid_split_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert summary["split_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert "holdout_fraction" not in summary
    assert "validation_fraction_of_holdout" not in summary


def test_c3_source_line_ranks_are_computed_before_amplitude_filtering(tmp_path: Path) -> None:
    interim_dir = _write_c3_source_line_interim_dataset(tmp_path)
    trace_table = pd.read_parquet(interim_dir / "traces.parquet")
    amplitudes = np.load(interim_dir / "amplitudes.npy", allow_pickle=False)
    middle_line_rows = trace_table.loc[trace_table["source_x_m"].eq(160.0), "array_row"].to_numpy(
        dtype=np.int64
    )
    amplitudes[middle_line_rows] = 0.0
    np.save(interim_dir / "amplitudes.npy", amplitudes)
    output_dir = tmp_path / "processed"

    summary = prepare_baseline_dataset(
        interim_dir,
        output_dir,
        holdout_fraction=None,
        validation_fraction_of_holdout=None,
        random_seed=RANDOM_SEED,
        split_scope="c3_source_line_blocks",
        source_line_ranges={"train": (0, 1), "validation": (1, 2), "test": (2, 3)},
        trace_amplitude_filter=TraceAmplitudeFilterConfig(
            exclude_all_zero=True,
            max_abs_amplitude=1_000.0,
        ),
        config_source=CONFIG_SOURCE,
    )

    split_table = pd.read_parquet(output_dir / "trace_split.parquet")
    joined = trace_table.merge(split_table, on="array_row", validate="one_to_one")
    assert set(joined.loc[joined["source_x_m"].eq(160.0), "split"]) == {EXCLUDED_SPLIT}
    assert set(joined.loc[joined["source_x_m"].eq(320.0), "split"]) == {"test"}
    assert summary["split_counts"] == {"train": 4, "validation": 0, "test": 4}
    assert summary["ffid_split_counts"] == {"train": 2, "validation": 0, "test": 2}


@pytest.mark.parametrize("random_seed", [True, 1.5, "42", -1])
def test_c3_source_line_blocks_reject_invalid_random_seed(
    tmp_path: Path,
    random_seed: object,
) -> None:
    interim_dir = _write_c3_source_line_interim_dataset(tmp_path)

    with pytest.raises(ValueError, match="random_seed"):
        prepare_baseline_dataset(
            interim_dir,
            tmp_path / "processed",
            holdout_fraction=None,
            validation_fraction_of_holdout=None,
            random_seed=random_seed,  # type: ignore[arg-type]
            split_scope="c3_source_line_blocks",
            source_line_ranges={"train": (0, 1), "validation": (1, 2), "test": (2, 3)},
            config_source=CONFIG_SOURCE,
        )


@pytest.mark.parametrize(
    ("holdout_fraction", "validation_fraction_of_holdout"),
    [(0.2, None), (None, 0.25)],
)
def test_c3_source_line_blocks_reject_random_fraction_arguments(
    tmp_path: Path,
    holdout_fraction: float | None,
    validation_fraction_of_holdout: float | None,
) -> None:
    interim_dir = _write_c3_source_line_interim_dataset(tmp_path)

    with pytest.raises(ValueError, match="holdout fractions are not used"):
        prepare_baseline_dataset(
            interim_dir,
            tmp_path / "processed",
            holdout_fraction=holdout_fraction,
            validation_fraction_of_holdout=validation_fraction_of_holdout,
            random_seed=RANDOM_SEED,
            split_scope="c3_source_line_blocks",
            source_line_ranges={"train": (0, 1), "validation": (1, 2), "test": (2, 3)},
            config_source=CONFIG_SOURCE,
        )


def test_amplitude_filter_excludes_invalid_traces_before_split_and_normalization(
    tmp_path: Path,
) -> None:
    interim_dir = _write_multi_ffid_interim_dataset(tmp_path)
    amplitude_path = interim_dir / "amplitudes.npy"
    amplitudes = np.load(amplitude_path, allow_pickle=False)
    amplitudes[:10] = 0.0
    amplitudes[10:20] = 2_000.0
    np.save(amplitude_path, amplitudes)
    output_dir = tmp_path / "processed"
    trace_filter = TraceAmplitudeFilterConfig(
        exclude_all_zero=True,
        max_abs_amplitude=1_000.0,
    )

    summary = _prepare(
        interim_dir,
        output_dir,
        split_scope="per_ffid",
        trace_amplitude_filter=trace_filter,
    )

    split_table = pd.read_parquet(output_dir / "trace_split.parquet")
    assert len(split_table) == 40
    assert split_table.loc[split_table["array_row"] < 20, "split"].eq(EXCLUDED_SPLIT).all()
    assert not split_table.loc[split_table["array_row"] >= 20, "split"].eq(EXCLUDED_SPLIT).any()
    assert summary["split_counts"] == {"train": 16, "validation": 1, "test": 3}
    assert summary["ffid_count"] == 1
    assert summary["trace_amplitude_filter"] == trace_filter.to_dict()
    assert summary["trace_quality"] == {
        "input_trace_count": 40,
        "eligible_trace_count": 20,
        "excluded_trace_count": 20,
        "all_zero_trace_count": 10,
        "excess_amplitude_trace_count": 10,
        "excluded_array_rows": list(range(20)),
        "affected_ffids": [100],
        "fully_excluded_ffids": [100],
    }
    training_rows = split_table.loc[split_table["split"] == "train", "array_row"].to_numpy()
    expected_rms = float(np.sqrt(np.mean(amplitudes[training_rows].astype(np.float64) ** 2)))
    assert read_normalization_parameters(
        output_dir / "normalization.json"
    ).amplitude_rms == pytest.approx(expected_rms)


def test_preparation_opens_amplitudes_as_a_read_only_memmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    observed: dict[str, object] = {}
    actual_load = prepare_baseline_pipeline.load_interim_trace_dataset

    def recording_load(directory: Path, **kwargs: object):
        observed.update(kwargs)
        dataset = actual_load(directory, **kwargs)
        observed["is_memmap"] = isinstance(dataset.amplitudes, np.memmap)
        observed["writeable"] = dataset.amplitudes.flags.writeable
        return dataset

    monkeypatch.setattr(
        prepare_baseline_pipeline,
        "load_interim_trace_dataset",
        recording_load,
    )

    _prepare(interim_dir, tmp_path / "processed")

    assert observed == {
        "memory_map_amplitudes": True,
        "is_memmap": True,
        "writeable": False,
    }


def test_preparation_reuses_the_loader_amplitude_finiteness_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    observed: list[bool] = []
    actual_fit = prepare_baseline_pipeline.fit_normalization_parameters

    def recording_fit(*args: object, **kwargs: object):
        observed.append(kwargs.get("amplitudes_are_finite") is True)
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(
        prepare_baseline_pipeline,
        "fit_normalization_parameters",
        recording_fit,
    )

    _prepare(interim_dir, tmp_path / "processed")

    assert observed == [True]


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
    held_out_rows = split_table.loc[split_table["split"] != "train", "array_row"].to_numpy()
    training_table = trace_table.set_index("array_row").loc[training_rows]
    held_out_table = trace_table.set_index("array_row").loc[held_out_rows]
    parameters = read_normalization_parameters(output_dir / "normalization.json")

    expected_min = (
        0.0,
        *(training_table[["cmp_x_m", "cmp_y_m", "offset_m"]].min()),
        -1.0,
        -1.0,
    )
    expected_max = (
        (SAMPLE_COUNT - 1) * 0.008,
        *(training_table[["cmp_x_m", "cmp_y_m", "offset_m"]].max()),
        1.0,
        1.0,
    )
    expected_rms = float(np.sqrt(np.mean(amplitudes[training_rows].astype(np.float64) ** 2)))

    np.testing.assert_array_equal(training_rows, EXPECTED_TRAIN_ROWS)
    np.testing.assert_array_equal(held_out_rows, EXPECTED_HELD_OUT_ROWS)
    np.testing.assert_allclose(parameters.coordinate_min, expected_min)
    np.testing.assert_allclose(parameters.coordinate_max, expected_max)
    assert parameters.amplitude_rms == pytest.approx(expected_rms)
    assert parameters.amplitude_rms == pytest.approx(EXPECTED_TRAIN_RMS)
    assert not np.isclose(
        expected_rms,
        np.sqrt(np.mean(amplitudes.astype(np.float64) ** 2)),
    )
    assert held_out_table["cmp_x_m"].max() > training_table["cmp_x_m"].max()
    assert held_out_table["cmp_y_m"].min() < training_table["cmp_y_m"].min()
    assert held_out_table["offset_m"].max() > training_table["offset_m"].max()


def test_preparation_records_relative_provenance_and_input_hashes(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"
    expected_input_files = {
        file_name: {"sha256": hashlib.sha256((interim_dir / file_name).read_bytes()).hexdigest()}
        for file_name in (
            "traces.parquet",
            "amplitudes.npy",
            "time_s.npy",
            "dataset.json",
        )
    }

    summary = _prepare(interim_dir, output_dir)
    preparation_text = (output_dir / "preparation.json").read_text(encoding="utf-8")

    assert summary["source_file"] == "source.sgy"
    assert "source_files" not in summary
    assert summary["config_source"] == CONFIG_SOURCE
    assert summary["normalization"] == {
        "coordinates": COORDINATE_NORMALIZATION_METHOD,
        "amplitude": "train_global_rms",
    }
    assert summary["input_files"] == expected_input_files
    assert "input_dataset_metadata_sha256" not in summary
    assert str(tmp_path) not in preparation_text


def test_preparation_records_canonical_multi_source_provenance(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    metadata_path = interim_dir / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("source_file")
    metadata.pop("source_sha256")
    metadata["source_files"] = [
        {"name": "part-1.sgy", "sha256": "a" * 64},
        {"name": "part-2.sgy", "sha256": "b" * 64},
    ]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = _prepare(interim_dir, tmp_path / "processed")

    assert summary["source_files"] == metadata["source_files"]
    assert "source_file" not in summary
    assert "source_sha256" not in summary


def test_rejects_an_unknown_split_scope_before_writing_outputs(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError, match="split_scope"):
        _prepare(interim_dir, output_dir, split_scope="per_source")

    assert not output_dir.exists()


def test_input_hash_detects_array_changes_without_metadata_changes(tmp_path: Path) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    first_summary = _prepare(interim_dir, tmp_path / "processed_first")

    amplitude_path = interim_dir / "amplitudes.npy"
    amplitudes = np.load(amplitude_path, allow_pickle=False)
    amplitudes[0, 0] += 1.0
    np.save(amplitude_path, amplitudes)

    second_summary = _prepare(interim_dir, tmp_path / "processed_second")
    first_input_files = first_summary["input_files"]
    second_input_files = second_summary["input_files"]
    assert isinstance(first_input_files, dict)
    assert isinstance(second_input_files, dict)
    assert first_input_files["dataset.json"] == second_input_files["dataset.json"]
    assert first_input_files["amplitudes.npy"] != second_input_files["amplitudes.npy"]


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


@pytest.mark.parametrize(
    "config_source",
    [
        "/absolute/config.yaml",
        "../outside/config.yaml",
        r"C:\\config.yaml",
        "C:config.yaml",
        r"\config.yaml",
        r"studies\study\config.yaml",
        ".",
        " config.yaml",
        "",
    ],
)
def test_rejects_non_portable_config_source(tmp_path: Path, config_source: str) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError, match="config_source"):
        prepare_baseline_dataset(
            interim_dir,
            output_dir,
            holdout_fraction=HOLDOUT_FRACTION,
            validation_fraction_of_holdout=VALIDATION_FRACTION_OF_HOLDOUT,
            random_seed=RANDOM_SEED,
            config_source=config_source,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("coordinate_normalization", "zero_to_one"),
        ("amplitude_normalization", "per_trace_rms"),
    ],
)
def test_rejects_unsupported_normalization_methods(
    tmp_path: Path,
    keyword: str,
    value: str,
) -> None:
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"
    arguments: dict[str, object] = {
        "holdout_fraction": HOLDOUT_FRACTION,
        "validation_fraction_of_holdout": VALIDATION_FRACTION_OF_HOLDOUT,
        "random_seed": RANDOM_SEED,
        keyword: value,
    }

    with pytest.raises(ValueError, match=keyword):
        prepare_baseline_dataset(interim_dir, output_dir, **arguments)  # type: ignore[arg-type]

    assert not output_dir.exists()


def test_cli_writes_final_resolved_config_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    monkeypatch.setattr("seis_interp.commands.data.REPOSITORY_ROOT", repository)
    default_config = repository / "configs" / "default.yaml"
    study_config = repository / "studies" / "study" / "config.yaml"
    default_config.parent.mkdir(parents=True)
    study_config.parent.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    default_config.write_text(
        "project:\n"
        "  random_seed: 7\n"
        "normalization:\n"
        "  coordinates: train_minmax_linear_plus_azimuth_sin_cos\n"
        "  amplitude: train_global_rms\n",
        encoding="utf-8",
    )
    study_config.write_text(
        "extends: ../../configs/default.yaml\n"
        "project:\n"
        "  random_seed: 42\n"
        "sampling:\n"
        "  random_trace_holdout_fraction: 0.20\n"
        "  validation_fraction_of_holdout: 0.25\n",
        encoding="utf-8",
    )
    interim_dir = _write_interim_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    exit_code = main(
        [
            "data",
            "prepare-baseline",
            "--config",
            str(study_config),
            "--input",
            str(interim_dir),
            "--output",
            str(output_dir),
            "--holdout-fraction",
            "0.30",
            "--validation-fraction-of-holdout",
            "0.50",
            "--random-seed",
            "0",
            "--json",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    stored_text = (output_dir / "preparation.json").read_text(encoding="utf-8")
    assert json.loads(stored_text) == summary
    assert summary["config_source"] == "studies/study/config.yaml"
    assert summary["random_seed"] == 0
    assert summary["holdout_fraction"] == 0.3
    assert summary["validation_fraction_of_holdout"] == 0.5
    assert summary["normalization"] == {
        "coordinates": COORDINATE_NORMALIZATION_METHOD,
        "amplitude": "train_global_rms",
    }
    assert summary["split_counts"] == {"train": 14, "validation": 3, "test": 3}
    assert str(tmp_path) not in stored_text
