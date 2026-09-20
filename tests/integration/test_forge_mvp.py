import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.pipelines.evaluate_forge_mvp import evaluate_forge_mvp
from seis_interp.pipelines.preflight_forge_mvp import preflight_forge_method
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.run_forge_mvp import run_forge_method
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.training.c3_poc_trace_graph_episodes import PocTraceGraphEpisodeGenerator
from seis_interp.training.forge_mvp_methods import graph_data
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    fixture = tiny_forge_study(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    import subprocess

    original = subprocess.check_output

    def git_in_real_repo(args, **kwargs):
        if args[:2] == ["git", "-C"]:
            args = [*args[:2], str(repo), *args[3:]]
        return original(args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", git_in_real_repo)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    run = prepare_forge_mvp(tmp_path, fixture[0])
    yield run, fixture
    torch.set_num_threads(previous)


def test_preparation_contains_observed_only_and_sealed_geometry(prepared):
    run, fixture = prepared
    values = fixture[-1]
    mask = pd.read_parquet(run / "masks/m1_random80_mvp_v1.parquet")
    seal = json.loads((run / "preparation_manifest.json").read_text())
    observed = mask.loc[mask.split.eq("observed"), "cell_id"].to_numpy()
    assert seal["global_rms"] == pytest.approx(
        np.sqrt(np.square(values[observed].astype(float)).mean())
    )
    for method in METHODS:
        inputs, config = load_model_inputs(run, method)
        np.testing.assert_array_equal(inputs.observed_values, values[observed])
        assert config["mask_hash"] == seal["mask_hash"]
        assert len(inputs.observed_values) == 7
        with pytest.raises(ValueError, match="outside outer observed"):
            inputs.read_observed(inputs.test_cells[:1])
        volume = inputs.volume()
        assert np.all(volume.values.reshape(9, -1)[:, inputs.test_cells] == 0)
        np.testing.assert_array_equal(volume.values.reshape(9, -1)[:, observed].T, values[observed])
        assert not hasattr(inputs, "dataset_root")
        assert "source_file" not in inputs.geometry
    path = run / "features/grid_geometry.parquet"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model_inputs(run, "regsi_grid")


def test_all_six_synthetic_methods_and_evaluation(prepared, tmp_path):
    preparation, _ = prepared
    output = tmp_path / "comparison"
    output.mkdir()
    write_json(
        output / "matrix_manifest.json",
        {
            "methods": list(METHODS),
            "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
            **execution_identity(tmp_path),
        },
    )
    for method in METHODS:
        run_forge_method(tmp_path, preparation, output / "runs" / method, method)
        assert not (output / "runs" / method / "metrics.json").exists()
    result = evaluate_forge_mvp(tmp_path, preparation, output)
    # This intentionally minimal kernel-1 CCNet has no spatial context and collapses.
    # The transport/evaluation succeeds, while the MVP correctly requires investigation.
    assert result["status"] == "investigation_required"
    assert result["findings"] == [
        {"method": "ccnet5d_grid", "issue": "all test predictions are constant"}
    ]
    metrics = pd.read_parquet(output / "aggregate/mvp_metrics.parquet")
    assert len(metrics) == 6
    assert metrics.status.eq("evaluated").all()
    records = {
        name: json.loads((output / "runs" / name / "run_manifest.json").read_text())
        for name in METHODS
    }
    assert (
        records["regsi_real"]["initialization_hash"] == records["regsi_grid"]["initialization_hash"]
    )
    assert records["regsi_real"]["parameter_count"] == records["regsi_grid"]["parameter_count"]
    for record in records.values():
        assert record["test_trace_count"] == 28
        assert record["independent_metrics_verified"]
        assert record["runtime_seconds"] > 0
        assert record["peak_cpu_rss_bytes"] > 0
    assert (output / "figures/common_source.png").is_file()


def test_failure_records_state_and_does_not_read_evaluation(prepared, tmp_path, monkeypatch):
    preparation, _ = prepared

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic OOM")

    monkeypatch.setattr("seis_interp.pipelines.run_forge_mvp.run_method", fail)
    output = tmp_path / "failed_method"
    with pytest.raises(RuntimeError, match="synthetic OOM"):
        run_forge_method(tmp_path, preparation, output, "regsi_real")
    record = json.loads((output / "run_manifest.json").read_text())
    assert record["status"] == "failed"
    assert record["prediction_hash"] is None
    assert record["failure_reason"] == "RuntimeError: synthetic OOM"
    assert record["end_time"]


def test_input_hash_failure_precedes_waveform_access(tmp_path, monkeypatch):
    fixture = tiny_forge_study(tmp_path)
    path = tmp_path / "candidate/source_stations.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")

    def forbidden(*args, **kwargs):
        raise AssertionError("waveform access before input validation")

    monkeypatch.setattr("seis_interp.pipelines.prepare_forge_mvp.read_selected_traces", forbidden)
    with pytest.raises(ValueError, match="input hash mismatch"):
        prepare_forge_mvp(tmp_path, fixture[0])


def test_observed_only_shape_checks_and_paired_episode_sequence(prepared, tmp_path):
    preparation, _ = prepared
    sequences = []
    records = {}
    for method in METHODS:
        output = tmp_path / f"{method}.json"
        preflight_forge_method(tmp_path, preparation, output, method)
        records[method] = json.loads(output.read_text())
        assert records[method]["status"] == "passed"
        assert records[method]["test_amplitudes_read"] == 0
        if method.startswith("regsi"):
            inputs, config = load_model_inputs(preparation, method)
            domain, training, _ = graph_data(inputs, config)
            assert set(training.domain.trace_ids) == set(inputs.observed_cells)
            np.testing.assert_array_equal(
                domain.source_xy_m, inputs.geometry[["source_x_m", "source_y_m"]]
            )
            generator = PocTraceGraphEpisodeGenerator(
                training,
                random_seed=config["model_seed"],
                missing_fraction=config["training"]["inner_mask_fraction"],
            )
            sequence = [generator.next_episode() for _ in range(3)]
            for episode in sequence:
                assert not set(episode.hidden_trace_ids) & set(inputs.test_cells)
            sequences.append(sequence)
    assert (
        records["regsi_real"]["initialization_hash"] == records["regsi_grid"]["initialization_hash"]
    )
    for real, grid in zip(*sequences, strict=True):
        np.testing.assert_array_equal(real.query_trace_ids, grid.query_trace_ids)
        np.testing.assert_array_equal(real.visible_mask, grid.visible_mask)
