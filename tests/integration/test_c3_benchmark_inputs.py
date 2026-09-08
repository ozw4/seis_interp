from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from seis_interp.data import c3_benchmark_suite as suite_store
from seis_interp.data.c3_benchmark_inputs import (
    load_c3_benchmark_graph_domain,
    load_c3_benchmark_supervised_source,
    load_c3_benchmark_training_graph_domain,
    load_c3_benchmark_volume_inputs,
)
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 4), (1, 2), (4, 2, 2, 2, 4))


@pytest.fixture
def prepared_suite(tmp_path):
    interim = make_benchmark_interim(tmp_path / "source")
    directory = tmp_path / "suite"
    prepare_c3_benchmark_artifacts(
        interim,
        directory,
        config=synthetic_benchmark_config(),
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )
    return interim, directory


@pytest.mark.parametrize("reuse_verified", [False, True])
def test_multiple_readers_do_not_repeat_full_verification(
    prepared_suite, monkeypatch, reuse_verified
):
    _, directory = prepared_suite
    verify = Mock(wraps=suite_store.verify_c3_benchmark_suite)
    monkeypatch.setattr(suite_store, "verify_c3_benchmark_suite", verify)
    verified = (
        suite_store.VerifiedC3BenchmarkSuite(directory, dimensions=DIMENSIONS)
        if reuse_verified
        else None
    )
    signal_qc = Mock(wraps=suite_store.summarize_c3_crop_signal)
    trace_masks = Mock(wraps=suite_store.make_random_trace_mask)
    ffid_masks = Mock(wraps=suite_store.make_random_whole_ffid_mask)
    monkeypatch.setattr(suite_store, "summarize_c3_crop_signal", signal_qc)
    monkeypatch.setattr(suite_store, "make_random_trace_mask", trace_masks)
    monkeypatch.setattr(suite_store, "make_random_whole_ffid_mask", ffid_masks)
    options = {"dimensions": DIMENSIONS, "verified_suite": verified}
    for case_id in ("test_random_trace", "test_random_whole_ffid"):
        volume = load_c3_benchmark_volume_inputs(directory, case_id, **options)
        graph = load_c3_benchmark_graph_domain(
            directory, case_id, volume_dir=directory / "volumes" / case_id, **options
        )
        assert volume.case["case_id"] == graph.inputs_lock["benchmark_case"]["case_id"]
        assert set(volume.observed_volume.array_rows.flat) == set(graph.array_rows)
    training = load_c3_benchmark_training_graph_domain(directory, **options)
    fit = {
        "time": [0, 4],
        "source_line": [0, 1],
        "shot_in_line": [0, 1],
        "relative_receiver_x": [3, 5],
        "relative_receiver_y": [32, 36],
    }
    teacher = load_c3_benchmark_supervised_source(
        directory, fit_region=fit, selection_region={**fit, "shot_in_line": [1, 2]}, **options
    )
    assert set(teacher.fit.array_rows.flat) <= set(training.array_rows)
    assert verify.call_count == int(reuse_verified)
    assert signal_qc.call_count == trace_masks.call_count == ffid_masks.call_count == 0


def test_single_case_and_train_pool_do_not_read_unrelated_case_files(prepared_suite):
    _, directory = prepared_suite
    unrelated = directory / "masks/validation_random_trace/observation_mask.parquet"
    unrelated.unlink()
    volume = load_c3_benchmark_volume_inputs(directory, "test_random_trace", dimensions=DIMENSIONS)
    assert volume.case["case_id"] == "test_random_trace"
    training = load_c3_benchmark_training_graph_domain(directory, dimensions=DIMENSIONS)
    assert training.pool == "all_train_traces"
    with pytest.raises(FileNotFoundError):
        suite_store.verify_c3_benchmark_suite(directory, dimensions=DIMENSIONS)


@pytest.mark.parametrize(
    "group,relative_path",
    [
        ("interim", "amplitudes.npy"),
        ("suite", "train_pool.npy"),
        ("suite", "masks/test_random_trace/observation_mask.parquet"),
        ("suite", "volumes/test_random_trace/volume.json"),
    ],
)
def test_reused_suite_and_complete_verify_reject_changed_required_files(
    prepared_suite, group, relative_path
):
    interim, directory = prepared_suite
    verified = suite_store.VerifiedC3BenchmarkSuite(directory, dimensions=DIMENSIONS)
    path = (interim if group == "interim" else directory) / relative_path
    path.write_bytes(path.read_bytes() + b"\n")
    for reused in (None, verified):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_c3_benchmark_volume_inputs(
                directory, "test_random_trace", dimensions=DIMENSIONS, verified_suite=reused
            )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        suite_store.verify_c3_benchmark_suite(directory, dimensions=DIMENSIONS)


def test_verified_suite_is_bound_to_directory_dimensions_and_manifest(prepared_suite):
    _, directory = prepared_suite
    verified = suite_store.VerifiedC3BenchmarkSuite(directory, dimensions=DIMENSIONS)
    for path, dimensions in (
        (directory / "another_suite", DIMENSIONS),
        (directory, C3BenchmarkDimensions((0, 3), (1, 2), (3, 2, 2, 2, 4))),
    ):
        with pytest.raises(ValueError, match="directory/dimensions"):
            load_c3_benchmark_volume_inputs(
                path, "test_random_trace", dimensions=dimensions, verified_suite=verified
            )
    manifest = directory / "benchmark_suite.json"
    changed = json.loads(manifest.read_text())
    changed["inference"]["outside_crop_context"] = True
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest changed"):
        load_c3_benchmark_volume_inputs(
            directory, "test_random_trace", dimensions=DIMENSIONS, verified_suite=verified
        )
    with pytest.raises(ValueError, match="same-crop observed"):
        suite_store.verify_c3_benchmark_suite(directory, dimensions=DIMENSIONS)
