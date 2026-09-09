from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.c3_benchmark_artifacts import benchmark_file_record
from seis_interp.data.c3_benchmark_suite import (
    benchmark_suite_files,
    load_c3_benchmark_input_manifest,
    verify_c3_benchmark_suite,
)
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.processing.c3_benchmark_partition import audit_c3_benchmark_partition
from seis_interp.processing.c3_crop_selection import resolve_c3_crop
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 4), (1, 2), (4, 2, 2, 2, 4))
POLICY = {"exclude_all_zero": False, "max_abs_amplitude": 10000.0}


def _prepare(interim, output, *, filtered=True):
    config = synthetic_benchmark_config()
    if filtered:
        config["sampling"]["trace_amplitude_filter"] = POLICY.copy()
    return prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=config,
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )


def _rehash_candidate(output, suite):
    candidate = deepcopy(suite)
    candidate["files"] = [
        benchmark_file_record(path, output) for path in benchmark_suite_files(output, suite)
    ]
    return candidate


def _write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False))


def test_explicit_qc_preserves_zero_boundary_and_fixed_heldout_cases(tmp_path):
    interim = make_benchmark_interim(tmp_path / "source")
    values = np.load(interim / "amplitudes.npy")
    # QC scans all raw samples, including sample 7 outside model time [0,4).
    values[0, 7] = 1e30
    values[1] = 0
    values[2, :2] = [10000.0, -10000.0]
    values[3, 7] = np.nextafter(np.float32(10000), np.float32(np.inf))
    np.save(interim / "amplitudes.npy", values)
    before_bytes = (interim / "amplitudes.npy").read_bytes()

    original_dir, clean_dir = tmp_path / "original", tmp_path / "clean"
    original = _prepare(interim, original_dir, filtered=False)
    clean = _prepare(interim, clean_dir)
    assert verify_c3_benchmark_suite(clean_dir, dimensions=DIMENSIONS) == clean
    assert load_c3_benchmark_input_manifest(clean_dir, dimensions=DIMENSIONS) == clean
    assert "amplitude_qc" not in original["partition_summary"]
    original_preparation = json.loads((original_dir / "partition/preparation.json").read_text())
    assert "trace_amplitude_filter" not in original_preparation
    assert "trace_quality" not in original_preparation

    quality = clean["partition_summary"]["amplitude_qc"]
    assert quality["policy"] == POLICY
    assert quality["time_samples"] == [0, 8]
    assert quality["excluded_array_rows"] == [0, 3]
    assert quality["excluded_trace_count"] == quality["excess_amplitude_trace_count"] == 2
    assert quality["all_zero_trace_count"] == 0
    pool = np.load(clean_dir / "train_pool.npy")
    original_pool = np.load(original_dir / "train_pool.npy")
    np.testing.assert_array_equal(pool, np.setdiff1d(original_pool, [0, 3]))
    assert 1 in pool and 2 in pool
    for before, after in zip(original["cases"], clean["cases"], strict=True):
        assert before["effective_mask"] == after["effective_mask"]
        for key, name in (
            ("mask_dir", "observation_mask.parquet"),
            ("volume_dir", "volume_index.parquet"),
        ):
            pd.testing.assert_frame_equal(
                pd.read_parquet(original_dir / before[key] / name),
                pd.read_parquet(clean_dir / after[key] / name),
            )
    assert (interim / "amplitudes.npy").read_bytes() == before_bytes


@pytest.mark.parametrize("partition,lines", [("test", (1, 3)), ("validation", (3, 5))])
def test_qc_exclusion_inside_fixed_crop_is_rejected_without_shrinking(tmp_path, partition, lines):
    interim = make_benchmark_interim(tmp_path / "source")
    table = pd.read_parquet(interim / "traces.parquet")
    times = np.load(interim / "time_s.npy")
    index, _ = resolve_c3_crop(
        table,
        times,
        source_line_range=lines,
        time_range=DIMENSIONS.time_range,
        spatial_lengths=DIMENSIONS.shape[2:],
    )
    bad_row = int(index["array_row"].iloc[0])
    values = np.load(interim / "amplitudes.npy")
    values[bad_row, 7] = 1e30
    np.save(interim / "amplitudes.npy", values)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=f"{partition} crop contains rows outside"):
        _prepare(interim, output)
    saved_index = pd.read_parquet(output / "qc" / partition / "crop_index.parquet")
    pd.testing.assert_frame_equal(saved_index, index)
    assert not (output / "benchmark_suite.json").exists()


def test_qc_rejects_promotion_of_a_clean_alias_for_excluded_canonical_row(tmp_path):
    interim = make_benchmark_interim(tmp_path / "source")
    table = pd.read_parquet(interim / "traces.parquet").drop(columns="array_row")
    duplicate = table.iloc[[0]].copy()
    duplicate["trace_index"] = len(table)
    table = pd.concat([table, duplicate], ignore_index=True)
    values = np.load(interim / "amplitudes.npy")
    values = np.vstack([values, values[[0]]])
    values[0, 7] = 1e30
    write_interim_trace_dataset(
        interim,
        table,
        values,
        np.load(interim / "time_s.npy"),
        interim.parent / "synthetic.sgy",
        "synthetic_c3",
        {"ffid_scope": "all"},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="promote a noncanonical physical-cell alias"):
        _prepare(interim, tmp_path / "rejected")


@pytest.fixture
def filtered_suite(tmp_path):
    interim = make_benchmark_interim(tmp_path / "source")
    values = np.load(interim / "amplitudes.npy")
    values[0, 7] = 1e30
    np.save(interim / "amplitudes.npy", values)
    output = tmp_path / "suite"
    return interim, output, _prepare(interim, output)


@pytest.mark.parametrize("change", ["different_policy", "missing_policy", "null_policy"])
def test_rehashed_config_cannot_change_prepared_qc_policy(filtered_suite, change):
    _, output, suite = filtered_suite
    path = output / "config.resolved.yaml"
    config = yaml.safe_load(path.read_text())
    if change == "different_policy":
        config["sampling"]["trace_amplitude_filter"]["max_abs_amplitude"] = 9000
    elif change == "missing_policy":
        del config["sampling"]["trace_amplitude_filter"]
    else:
        config["sampling"]["trace_amplitude_filter"] = None
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="amplitude.filter"):
        verify_c3_benchmark_suite(
            output, candidate=_rehash_candidate(output, suite), dimensions=DIMENSIONS
        )


@pytest.mark.parametrize("change", ["declared_exclusions", "split_exclusions"])
def test_rehashed_exclusions_are_recomputed_from_raw_samples(filtered_suite, change):
    _, output, suite = filtered_suite
    if change == "declared_exclusions":
        path = output / "partition/preparation.json"
        preparation = json.loads(path.read_text())
        preparation["trace_quality"]["excluded_array_rows"] = [1]
        _write_json(path, preparation)
        message = "QC exclusions/counts differ from raw samples"
    else:
        path = output / "partition/trace_split.parquet"
        split = pd.read_parquet(path)
        split.loc[split.array_row.eq(0), "split"] = "train"
        split.loc[split.array_row.eq(1), "split"] = "excluded"
        split.to_parquet(path, index=False)
        message = "independently filtered source-line assignments"
    with pytest.raises(ValueError, match=message):
        verify_c3_benchmark_suite(
            output, candidate=_rehash_candidate(output, suite), dimensions=DIMENSIONS
        )


def test_absent_filter_still_requires_strict_unfiltered_partition(filtered_suite):
    interim, output, _ = filtered_suite
    preparation = json.loads((output / "partition/preparation.json").read_text())
    del preparation["trace_amplitude_filter"]
    del preparation["trace_quality"]
    with pytest.raises(ValueError, match="unfiltered source-line assignments"):
        audit_c3_benchmark_partition(
            pd.read_parquet(interim / "traces.parquet"),
            pd.read_parquet(output / "partition/trace_split.parquet"),
            preparation,
            {},
            time_range=DIMENSIONS.time_range,
        )


def test_explicit_qc_audit_rejects_model_time_subset_instead_of_full_raw_samples(filtered_suite):
    interim, output, _ = filtered_suite
    preparation = json.loads((output / "partition/preparation.json").read_text())
    values = np.load(interim / "amplitudes.npy", mmap_mode="r")
    with pytest.raises(ValueError, match="all raw rows and time samples"):
        audit_c3_benchmark_partition(
            pd.read_parquet(interim / "traces.parquet"),
            pd.read_parquet(output / "partition/trace_split.parquet"),
            preparation,
            {},
            time_range=DIMENSIONS.time_range,
            sampling={"trace_amplitude_filter": POLICY},
            amplitudes=values[:, : DIMENSIONS.time_range[1]],
        )
