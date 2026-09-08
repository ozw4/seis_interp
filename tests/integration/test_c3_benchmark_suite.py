from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.c3_benchmark_suite import verify_c3_benchmark_suite
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 4), (1, 2), (4, 2, 2, 2, 4))


@pytest.mark.parametrize("hole", [False, True])
def test_preparation_through_independent_manifest_verification(tmp_path, hole):
    interim = make_benchmark_interim(tmp_path / "source", hole=hole)
    output = tmp_path / "suite"
    suite = prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=synthetic_benchmark_config(),
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )
    assert suite["status"] == "locked"
    assert len(suite["cases"]) == 4
    assert verify_c3_benchmark_suite(output, dimensions=DIMENSIONS) == suite
    crop = json.loads((output / "qc/test/crop.json").read_text())
    assert crop["selection"]["time"] == [0, 4]
    assert crop["selection"]["source_line"] == [1, 3]
    assert crop["shape"] == [4, 2, 2, 2, 4]
    assert crop["time"]["time_start_s"] == 0.125
    assert bool(crop["adjustment_reason"]) == hole
    assert suite["partition_summary"]["source_line_ranges"] == {
        "train": [0, 1],
        "test": [1, 3],
        "validation": [3, 5],
    }
    pool = np.load(output / "train_pool.npy")
    test_rows = pd.read_parquet(output / "qc/test/crop_index.parquet")["array_row"].to_numpy()
    assert not np.intersect1d(pool, test_rows).size
    with pytest.raises(FileExistsError):
        prepare_c3_benchmark_artifacts(
            interim,
            output,
            config=synthetic_benchmark_config(),
            inputs={"cases": synthetic_benchmark_cases()},
            dimensions=DIMENSIONS,
        )


def test_partial_case_selection_does_not_write_a_lock(tmp_path):
    interim = make_benchmark_interim(tmp_path / "source")
    output = tmp_path / "partial"
    result = prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=synthetic_benchmark_config(),
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
        case_ids=["test_random_trace"],
    )
    assert result["status"] == "partial"
    assert len(result["missing_cases"]) == 3
    assert not (output / "benchmark_suite.json").exists()


def test_two_preparations_reproduce_semantic_tables_and_roles(tmp_path):
    interim = make_benchmark_interim(tmp_path / "source")
    suites = []
    for name in ("first", "second"):
        suites.append(
            prepare_c3_benchmark_artifacts(
                interim,
                tmp_path / name,
                config=synthetic_benchmark_config(),
                inputs={"cases": synthetic_benchmark_cases()},
                dimensions=DIMENSIONS,
            )
        )
    for first, second in zip(suites[0]["cases"], suites[1]["cases"], strict=True):
        assert first["effective_mask"] == second["effective_mask"]
        for key, file in (
            ("mask_dir", "observation_mask.parquet"),
            ("volume_dir", "volume_index.parquet"),
        ):
            pd.testing.assert_frame_equal(
                pd.read_parquet(tmp_path / "first" / first[key] / file),
                pd.read_parquet(tmp_path / "second" / second[key] / file),
            )


def test_manifest_rejects_missing_files_wrong_contract_and_modified_amplitudes(tmp_path):
    interim = make_benchmark_interim(tmp_path / "source")
    output = tmp_path / "suite"
    suite = prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=synthetic_benchmark_config(),
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )
    for key, value in (
        ("inference", {**suite["inference"], "outside_crop_context": True}),
        ("evaluation", {"domain": "all", "primary_metric": "physical_amplitude_global_snr_db"}),
    ):
        wrong = deepcopy(suite)
        wrong[key] = value
        with pytest.raises(ValueError):
            verify_c3_benchmark_suite(output, candidate=wrong, dimensions=DIMENSIONS)
    wrong = deepcopy(suite)
    wrong["files"] = wrong["files"][1:]
    with pytest.raises(ValueError, match="inventory"):
        verify_c3_benchmark_suite(output, candidate=wrong, dimensions=DIMENSIONS)
    wrong = deepcopy(suite)
    first, second = wrong["cases"][:2]
    first["volume_dir"], second["volume_dir"] = second["volume_dir"], first["volume_dir"]
    with pytest.raises(ValueError, match="volume binding"):
        verify_c3_benchmark_suite(output, candidate=wrong, dimensions=DIMENSIONS)
    pool = output / "train_pool.npy"
    hidden = output / "train_pool.hidden.npy"
    pool.rename(hidden)
    try:
        with pytest.raises(FileNotFoundError):
            verify_c3_benchmark_suite(output, dimensions=DIMENSIONS)
    finally:
        hidden.rename(pool)
    values = np.load(interim / "amplitudes.npy")
    values[0, 0] += 1
    np.save(interim / "amplitudes.npy", values)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_c3_benchmark_suite(output, dimensions=DIMENSIONS)
