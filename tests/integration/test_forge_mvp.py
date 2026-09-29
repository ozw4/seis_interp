import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    load_model_inputs,
    mvp_implementation_files,
    read_selected_traces,
    verify_hashes,
    verify_mvp_implementation,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.pipelines.evaluate_forge_mvp import evaluate_forge_mvp
from seis_interp.pipelines.preflight_forge_mvp import preflight_forge_method, preflight_forge_mvp
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.run_forge_mvp import run_forge_method, run_forge_mvp
from seis_interp.pipelines.visualize_forge_mvp import visualize_forge_mvp
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.processing.forge_time_sampling import resample_forge_traces
from seis_interp.training.c3_poc_trace_graph_episodes import (
    PocTraceGraphEpisodeGenerator,
    PocTraceGraphEpisodeLabelReader,
    build_poc_trace_graph_context_source,
)
from seis_interp.training.forge_mvp_methods import graph_data
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.fixture(params=["observed_global_rms", "observed_trace_rms_idw", "trace_oracle"])
def prepared(tmp_path, monkeypatch, request):
    fixture = tiny_forge_study(tmp_path)
    if request.param == "observed_trace_rms_idw":
        contract = fixture[1]
        contract["preprocessing"].update(
            amplitude_scaling=request.param,
            trace_rms=True,
            idw={
                "geometry": "projected_grid",
                "radius": 2,
                "power": 2.0,
                "distance": "physical",
                "restoration": "multiply_once",
            },
        )
        (fixture[0] / "config.yaml").write_text(yaml.safe_dump(contract))
    elif request.param == "trace_oracle":
        contract = fixture[1]
        contract["time_sampling"] = {
            "native_sample_count": 9,
            "native_sample_interval_us": 1000,
            "decimation_factor": 2,
        }
        contract.update(time_sample_count=5, sample_interval_us=2000)
        contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
        (fixture[0] / "config.yaml").write_text(yaml.safe_dump(contract))
        configs = fixture[3]
        configs["nersi_real"]["model"]["time_sample_count"] = 5
        configs["ccnet5d_grid"]["patches"]["shape"][0] = 5
        for method in ("nersi_real", "ccnet5d_grid"):
            (fixture[0] / f"{method}.yaml").write_text(yaml.safe_dump(configs[method]))
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
    values = resample_forge_traces(fixture[-1], fixture[1])
    mask = pd.read_parquet(run / "masks/m1_random80_mvp_v1.parquet")
    seal = json.loads((run / "preparation_manifest.json").read_text())
    observed = mask.loc[mask.split.eq("observed"), "cell_id"].to_numpy()
    gates = json.loads((run / "preparation_gates.json").read_text())
    if fixture[1]["preprocessing"]["amplitude_scaling"] == "trace_oracle":
        scales = pd.read_parquet(run / "input/trace_scales.parquet")
        np.testing.assert_array_equal(
            scales.rms, np.sqrt(np.mean(np.square(values[:35].astype(np.float64)), axis=1))
        )
        assert gates["test_waveforms_read"] == 28
        assert set(scales.scale_source) == {"observed", "oracle_test"}
        assert seal["amplitude_scaling"] == "trace_oracle"
        assert "idw" not in seal
    else:
        assert gates["test_waveforms_read"] == 0
    assert not (run / "input/test_waveforms.npy").exists()
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
        assert np.all(volume.values.reshape(values.shape[1], -1)[:, inputs.test_cells] == 0)
        np.testing.assert_array_equal(
            volume.values.reshape(values.shape[1], -1)[:, observed].T, values[observed]
        )
        np.testing.assert_array_equal(
            inputs.time_s, np.arange(values.shape[1]) * fixture[1]["sample_interval_us"] / 1e6
        )
        assert not hasattr(inputs, "dataset_root")
        assert "source_file" not in inputs.geometry
    path = run / "features/grid_geometry.parquet"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model_inputs(run, "regsi_grid")


@pytest.fixture
def prediction_matrix(prepared, tmp_path):
    preparation, _ = prepared
    output = tmp_path / "comparison"
    output.mkdir()
    write_json(
        output / "matrix_manifest.json",
        {
            "methods": list(METHODS),
            "preparation": str(preparation.relative_to(tmp_path)),
            "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
            **execution_identity(tmp_path),
        },
    )
    for method in METHODS:
        run_forge_method(tmp_path, preparation, output / "runs" / method, method)
        assert not (output / "runs" / method / "metrics.json").exists()
    return preparation, output


def test_all_six_synthetic_methods_and_evaluation(prediction_matrix, tmp_path):
    preparation, output = prediction_matrix
    with patch(
        "seis_interp.pipelines.evaluate_forge_mvp.read_selected_traces",
        wraps=read_selected_traces,
    ) as reader:
        result = evaluate_forge_mvp(tmp_path, preparation, output)
    reader.assert_called_once()
    assert result["status"] == "accepted"
    assert result["findings"] == []
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
    final_hash = sha256_file(output / "aggregate/final_mvp_manifest.json")
    with patch(
        "seis_interp.pipelines.visualize_forge_mvp.read_selected_traces",
        wraps=read_selected_traces,
    ) as reader:
        figures = visualize_forge_mvp(tmp_path, output)
    reader.assert_called_once()
    assert figures != output
    verify_hashes(output, result["artifacts"])
    assert sha256_file(output / "aggregate/final_mvp_manifest.json") == final_hash
    record = json.loads((figures / "manifest.json").read_text())
    assert record["status"] == "completed"
    assert record["source_manifest_hash"] == final_hash
    verify_hashes(figures, record["artifacts"])
    for role in ("source", "receiver"):
        for suffix in ("", "_errors"):
            name = f"figures/common_{role}{suffix}.png"
            # Saved predictions and native/resampled targets reconstruct the same panels.
            assert sha256_file(figures / name) == sha256_file(output / name)
    path = output / "runs/regsi_grid/prediction.npy"
    path.write_bytes(path.read_bytes() + b"tamper")
    with (
        patch("seis_interp.pipelines.visualize_forge_mvp.read_selected_traces") as reader,
        pytest.raises(ValueError, match="input hash mismatch"),
    ):
        visualize_forge_mvp(tmp_path, output)
    reader.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "running", "missing"])
def test_incomplete_matrix_does_not_open_test_amplitudes(prediction_matrix, tmp_path, status):
    preparation, output = prediction_matrix
    path = output / "runs/nersi_real/run_manifest.json"
    if status == "missing":
        path.unlink()
    else:
        record = json.loads(path.read_text())
        record.update(status=status, failure_reason="synthetic OOM" if status == "failed" else None)
        write_json(path, record)
    with (
        patch("seis_interp.pipelines.evaluate_forge_mvp.read_selected_traces") as reader,
        patch("seis_interp.pipelines.evaluate_forge_mvp.evaluate_traces") as evaluator,
    ):
        result = evaluate_forge_mvp(tmp_path, preparation, output)
    reader.assert_not_called()
    evaluator.assert_not_called()
    assert result["status"] == "incomplete"
    assert result["completed_methods"] == []
    assert set(result["valid_methods"]) == set(METHODS) - {"nersi_real"}
    assert result["test_amplitudes_read"] == 0
    assert not result["evaluation_started"]
    assert not list(output.rglob("metrics.json"))
    assert not (output / "aggregate/mvp_metrics.parquet").exists()
    assert not (output / "figures").exists()
    assert json.loads((output / "aggregate/final_mvp_manifest.json").read_text()) == result
    assert (output / "aggregate/runtime_summary.parquet").is_file()
    with (
        patch("seis_interp.pipelines.visualize_forge_mvp.read_selected_traces") as reader,
        pytest.raises(ValueError, match="requires an accepted six-method matrix"),
    ):
        visualize_forge_mvp(tmp_path, output)
    reader.assert_not_called()


@pytest.mark.parametrize(
    "fault", ["zero", "constant", "nan", "inf", "shape", "cell_id", "hash", "initialization"]
)
def test_invalid_prediction_matrix_never_opens_test_amplitudes(prediction_matrix, tmp_path, fault):
    preparation, output = prediction_matrix
    # Use the final method to check that all earlier successful methods remain blinded.
    path = output / "runs/regsi_grid"
    record = json.loads((path / "run_manifest.json").read_text())
    prediction = np.load(path / "prediction.npy")
    if fault == "cell_id":
        index = pd.read_parquet(path / "prediction_index.parquet")
        index.loc[0, "cell_id"] = index.loc[1, "cell_id"]
        index.to_parquet(path / "prediction_index.parquet", index=False)
        record["prediction_index_hash"] = sha256_file(path / "prediction_index.parquet")
    elif fault == "initialization":
        record["initialization_hash"] = "different initial weights"
    elif fault == "hash":
        record["prediction_hash"] = "invalid hash"
    else:
        if fault == "zero":
            prediction[:] = 0
        if fault == "constant":
            prediction[:] = np.arange(len(prediction))[:, None]
        if fault == "nan":
            prediction[-1, -1] = np.nan
        if fault == "inf":
            prediction[-1, -1] = np.inf
        if fault == "shape":
            prediction = prediction[:, :-1]
        np.save(path / "prediction.npy", prediction)
        record["prediction_hash"] = sha256_file(path / "prediction.npy")
    write_json(path / "run_manifest.json", record)
    with (
        patch("seis_interp.pipelines.evaluate_forge_mvp.read_selected_traces") as reader,
        patch("seis_interp.pipelines.evaluate_forge_mvp.evaluate_traces") as evaluator,
    ):
        if fault in ("zero", "constant"):
            result = evaluate_forge_mvp(tmp_path, preparation, output)
            assert result["status"] == "incomplete"
            assert result["findings"] == [
                {"method": "regsi_grid", "issue": "all test predictions are constant"}
            ]
        else:
            with pytest.raises(ValueError):
                evaluate_forge_mvp(tmp_path, preparation, output)
    reader.assert_not_called()
    evaluator.assert_not_called()
    assert not list(output.rglob("metrics.json"))


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
            domain, training, _ = graph_data(inputs.model_space(), config)
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
                context = build_poc_trace_graph_context_source(training, episode)
                with pytest.raises(ValueError):
                    context.read_observed_rows(episode.hidden_trace_ids[:1])
                if inputs.observed_rms is not None:
                    labels = (
                        PocTraceGraphEpisodeLabelReader(training, episode)
                        .read(episode.hidden_trace_ids)
                        .numpy()
                    )
                    np.testing.assert_allclose(
                        np.sqrt(np.mean(labels.astype(float) ** 2, axis=1)), 1.0, rtol=1e-7
                    )
            sequences.append(sequence)
    assert (
        records["regsi_real"]["initialization_hash"] == records["regsi_grid"]["initialization_hash"]
    )
    for real, grid in zip(*sequences, strict=True):
        np.testing.assert_array_equal(real.query_trace_ids, grid.query_trace_ids)
        np.testing.assert_array_equal(real.visible_mask, grid.visible_mask)


@pytest.mark.parametrize("entrypoint", ["run", "preflight"])
def test_modified_cli_stops_before_subprocess(prepared, tmp_path, entrypoint):
    preparation, _ = prepared
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    assert set(seal["implementation_hashes"]) == {
        str(path.relative_to(tmp_path)) for path in mvp_implementation_files(tmp_path)
    }
    relative = "scripts/forge_m1_mvp.py"
    script = tmp_path / relative
    snapshot = preparation / "source" / relative
    assert snapshot.read_bytes() == script.read_bytes()
    assert seal["implementation_hashes"][relative] == sha256_file(script)
    script.write_text(script.read_text() + "\n# changed after preparation\n")
    with (
        patch("subprocess.run") as launch,
        patch("subprocess.Popen") as start,
        pytest.raises(ValueError, match="input hash mismatch: scripts/forge_m1_mvp.py"),
    ):
        if entrypoint == "run":
            run_forge_mvp(tmp_path, preparation, tmp_path / "run", tmp_path / "preflight")
        else:
            preflight_forge_mvp(tmp_path, preparation, tmp_path / "preflight")
    launch.assert_not_called()
    start.assert_not_called()


DEVICES = {
    "pocs_grid": "cpu",
    "drr_grid": "cpu",
    "ccnet5d_grid": "cuda:0",
    "nersi_real": "cuda:1",
    "regsi_real": "cuda:1",
    "regsi_grid": "cuda:1",
}


def _fake_matrix(tmp_path, monkeypatch, worker):
    module = "seis_interp.pipelines.run_forge_mvp"
    monkeypatch.setattr(f"{module}.subprocess.Popen", worker)
    for name in ("verify_hashes", "verify_mvp_implementation"):
        monkeypatch.setattr(f"{module}.{name}", lambda *args: None)
    monkeypatch.setattr(f"{module}.validate_preflight", lambda *args: "preflight")
    monkeypatch.setattr(f"{module}.sha256_file", lambda path: "hash")
    monkeypatch.setattr(f"{module}.execution_identity", lambda repo: {"git_commit": "c"})
    monkeypatch.setattr(
        "seis_interp.pipelines.evaluate_forge_mvp.evaluate_forge_mvp",
        lambda *args: {"status": "accepted"},
    )
    preparation = tmp_path / "preparation"
    write_json(
        preparation / "preparation_manifest.json", {"artifacts": {}, "implementation_hashes": {}}
    )
    for method, device in DEVICES.items():
        (preparation / "configs").mkdir(exist_ok=True)
        (preparation / "configs" / f"{method}.yaml").write_text(yaml.safe_dump({"device": device}))
    return preparation


@pytest.mark.parametrize("concurrent", [False, True])
def test_matrix_runner_serializes_each_cuda_device(tmp_path, monkeypatch, concurrent):
    events = []

    class Worker:
        def __init__(self, args, **kwargs):
            self.method = args[args.index("--method") + 1]
            self.polls = 0
            events.append(("start", self.method))

        def poll(self):
            self.polls += 1
            return 0 if self.polls > 1 else None

        def wait(self, timeout=None):
            events.append(("finish", self.method))
            return 0

    monkeypatch.setattr("seis_interp.pipelines.run_forge_mvp.sleep", lambda seconds: None)
    preparation = _fake_matrix(tmp_path, monkeypatch, Worker)
    output = tmp_path / "run"
    run_forge_mvp(tmp_path, preparation, output, tmp_path / "preflight", concurrent=concurrent)
    position = {event: index for index, event in enumerate(events)}
    assert len(events) == 2 * len(METHODS)
    manifest = json.loads((output / "matrix_manifest.json").read_text())
    if not concurrent:
        assert events == [(kind, m) for m in METHODS for kind in ("start", "finish")]
        assert "execution" not in manifest
    else:
        assert events[:4] == [
            ("start", m) for m in ("pocs_grid", "drr_grid", "ccnet5d_grid", "nersi_real")
        ]
        assert position[("start", "regsi_real")] > position[("finish", "nersi_real")]
        assert position[("start", "regsi_grid")] > position[("finish", "regsi_real")]
        assert manifest["execution"] == "concurrent_by_device"
        assert manifest["execution_queues"]["cuda:1"] == ["nersi_real", "regsi_real", "regsi_grid"]
        assert manifest["runtime_comparable"] is False
    assert all((output / f"{method}.log").is_file() for method in METHODS)


def test_interrupted_concurrent_runner_stops_every_started_worker(tmp_path, monkeypatch):
    workers = []

    class Worker:
        def __init__(self, args, **kwargs):
            self.stopped = False
            workers.append(self)

        def poll(self):
            return -15 if self.stopped else None

        def wait(self, timeout=None):
            return -15

        def terminate(self):
            self.stopped = True

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("seis_interp.pipelines.run_forge_mvp.sleep", interrupt)
    preparation = _fake_matrix(tmp_path, monkeypatch, Worker)
    with pytest.raises(KeyboardInterrupt):
        run_forge_mvp(tmp_path, preparation, tmp_path / "run", tmp_path / "pf", concurrent=True)
    # One head per queue was running: two CPU methods, cuda:0 and cuda:1.
    assert len(workers) == 4
    assert all(worker.stopped for worker in workers)


def test_implementation_manifest_must_include_cli(prepared, tmp_path):
    preparation, _ = prepared
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    del seal["implementation_hashes"]["scripts/forge_m1_mvp.py"]
    with pytest.raises(ValueError, match="input hash mismatch: implementation file set"):
        verify_mvp_implementation(tmp_path, seal["implementation_hashes"])


def test_trace_scaling_is_shared_and_model_receives_unit_rms(prepared):
    run, _ = prepared
    scales = []
    for method in METHODS:
        inputs, config = load_model_inputs(run, method)
        model_inputs = inputs.model_space()
        if inputs.observed_rms is None:
            assert model_inputs is inputs
            continue
        scales.append(inputs.test_rms)
        assert model_inputs.global_rms == 1.0
        assert model_inputs.observed_rms is None and model_inputs.test_rms is None
        np.testing.assert_allclose(
            np.sqrt(np.mean(model_inputs.observed_values.astype(float) ** 2, axis=1)),
            1.0,
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            model_inputs.observed_values * inputs.observed_rms[:, None],
            inputs.observed_values,
            rtol=1e-7,
        )
        assert config["trace_scale_hash"] == inputs.binding["trace_scale_hash"]
    for scale in scales[1:]:
        np.testing.assert_array_equal(scale, scales[0])


def test_trace_scaling_preserves_locked_mask_across_studies(prepared, tmp_path):
    import shutil

    preparation, fixture = prepared
    original_study = fixture[0]
    study = tmp_path / "studies" / "followup"
    shutil.copytree(original_study, study)
    config = yaml.safe_load((study / "config.yaml").read_text())
    config.update(mask_study_id=config["study_id"], study_id="followup")
    (study / "config.yaml").write_text(yaml.safe_dump(config))
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    mask = preparation / "masks/m1_random80_mvp_v1.parquet"
    inputs.update(outer_mask=str(mask.relative_to(tmp_path)), outer_mask_sha256=sha256_file(mask))
    (study / "inputs.yaml").write_text(yaml.safe_dump(inputs))
    with patch(
        "seis_interp.pipelines.prepare_forge_mvp.read_selected_traces", wraps=read_selected_traces
    ) as reader:
        followup = prepare_forge_mvp(tmp_path, study)
    requested = reader.call_args_list[0].args[1].cell_id.to_numpy()
    expected = pd.read_parquet(mask).query("split == 'observed'").cell_id.to_numpy()
    np.testing.assert_array_equal(requested, expected)
    assert reader.call_count == (
        2 if config["preprocessing"]["amplitude_scaling"] == "trace_oracle" else 1
    )
    assert sha256_file(followup / "masks/m1_random80_mvp_v1.parquet") == sha256_file(mask)


def test_trace_scale_binding_failure_precedes_target_access(prediction_matrix, tmp_path):
    preparation, output = prediction_matrix
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    if "trace_scale_hash" not in seal:
        return
    path = output / "runs/regsi_grid/run_manifest.json"
    record = json.loads(path.read_text())
    record["trace_scale_hash"] = "different_scale_table"
    write_json(path, record)
    with (
        patch("seis_interp.pipelines.evaluate_forge_mvp.read_selected_traces") as reader,
        pytest.raises(ValueError, match="trace scale binding mismatch"),
    ):
        evaluate_forge_mvp(tmp_path, preparation, output)
    reader.assert_not_called()
