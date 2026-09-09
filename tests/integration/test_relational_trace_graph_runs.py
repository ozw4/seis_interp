"""CPU train, checkpoint and frozen benchmark commands with immutable run records."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_interp.cli import main
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines.interpolate_relational_trace_graph import (
    interpolate_relational_trace_graph_run,
)
from seis_interp.pipelines.train_relational_trace_graph import train_relational_trace_graph_run
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
)
from tests.fixtures.relational_trace_graph_runs import (
    prepare_trace_graph_run_artifacts,
    trace_graph_prediction_config,
    trace_graph_training_config,
    write_trace_graph_config,
)


def _train(artifacts, config, output, **kwargs):
    return train_relational_trace_graph_run(
        config_path=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        validation_mask_dir=artifacts.masks["validation"],
        validation_case_dir=artifacts.cases["validation"],
        output_dir=output,
        **kwargs,
    )


def _infer(artifacts, config, checkpoint, output, **kwargs):
    return interpolate_relational_trace_graph_run(
        config_path=config,
        checkpoint_path=checkpoint,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.masks["test"],
        case_dir=artifacts.cases["test"],
        output_dir=output,
        **kwargs,
    )


def _records(output):
    return {
        name: json.loads((output / f"{name}.json").read_text())
        for name in ("run", "metrics", "inputs.lock")
    }


def test_training_physical_bound_blocks_before_model_or_run_creation(tmp_path, monkeypatch):
    artifacts = prepare_trace_graph_run_artifacts(tmp_path / "data")
    config = trace_graph_training_config()
    config["training_data"]["max_abs_amplitude"] = 1e-12
    config_path = write_trace_graph_config(tmp_path / "training.yaml", config)
    output = tmp_path / "training"
    monkeypatch.setattr(
        "seis_interp.pipelines.train_relational_trace_graph.RelationalTraceGraphInterpolator",
        lambda **kwargs: pytest.fail("invalid training amplitudes must block model initialization"),
    )

    with pytest.raises(ValueError, match=r"max_abs_amplitude=1e-12.*trace_id.*array_row.*sample"):
        _train(artifacts, config_path, output)

    assert not output.exists()


@pytest.mark.parametrize("variant", ["plain_gcn_row_normalized", "untyped_edge_conditioned"])
def test_control_pipeline_records_variant_and_physical_diagnostics(variant, tmp_path):
    data = prepare_trace_graph_run_artifacts(tmp_path / "data", dense=True)
    config = trace_graph_training_config()
    config["model"].update(method_variant=variant, relation_fusion="mean")
    config["graph"]["common_distance_scales_m"] = [2000.0, 5000.0]
    if variant == "untyped_edge_conditioned":
        config["graph"]["topology"] = "single_4d"
    config["diagnostics"] = {
        "time_s": [0.002],
        "offset_m": [1000.0],
        "azimuth_deg": [90.0, 180.0, 270.0],
    }
    config_path = write_trace_graph_config(tmp_path / "train.yaml", config)
    output = tmp_path / "train"
    trained = _train(
        data,
        config_path,
        output,
        validation_volume_dir=data.volumes["validation"],
        train_mask_dir=data.masks["train"],
        train_case_dir=data.cases["train"],
        train_volume_dir=data.volumes["train"],
    )
    run = _records(output)["run"]
    assert run["method_variant"] == trained["method_variant"] == variant
    assert run["parameter_count"] > 0
    if variant == "untyped_edge_conditioned":
        assert run["graph"]["relation_names"] == ["untyped"]
    best = trained["best_validation_metrics"]["evaluation_target"]
    bands = trained["best_validation_metrics"]["diagnostic_bands"]
    assert bands["query_count"] == best["trace_count"]
    for name in ("sample_count", "reference_energy", "error_energy"):
        assert bands[name] == pytest.approx(best[name], rel=1e-12, abs=1e-12)
    for baseline in ("zero", "idw"):
        metrics = trained["baselines"][baseline]["evaluation_target"]
        for name in ("trace_count", "sample_count", "reference_energy"):
            assert metrics[name] == best[name]
    assert "timings" in run["prediction"]["diagnostics"]
    frozen = trace_graph_prediction_config()
    frozen["diagnostics"] = config["diagnostics"]
    frozen_path = write_trace_graph_config(tmp_path / "predict.yaml", frozen)
    inferred = _infer(
        data,
        frozen_path,
        output / "artifacts/best.pt",
        tmp_path / "infer",
        volume_dir=data.volumes["test"],
    )
    assert inferred["method_variant"] == variant
    assert (
        inferred["diagnostic_bands"]["query_count"] == inferred["evaluation_target"]["trace_count"]
    )
    for name in ("sample_count", "reference_energy", "error_energy"):
        assert inferred["diagnostic_bands"][name] == pytest.approx(
            inferred["evaluation_target"][name], rel=1e-12, abs=1e-12
        )


@pytest.mark.parametrize("dense", [False, True])
def test_train_best_checkpoint_frozen_prediction_and_metrics(dense, tmp_path, capsys):
    data = prepare_trace_graph_run_artifacts(tmp_path / "data", dense=dense)
    train_config = write_trace_graph_config(tmp_path / "train.yaml", trace_graph_training_config())
    output = tmp_path / "train"
    train_args = [
        "train",
        "relational-trace-graph",
        "--config",
        str(train_config),
        "--interim",
        str(data.interim),
        "--processed",
        str(data.processed),
        "--validation-mask",
        str(data.masks["validation"]),
        "--validation-case",
        str(data.cases["validation"]),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--json",
    ]
    if dense:
        train_args += ["--validation-volume", str(data.volumes["validation"])]
        train_args += [
            "--train-mask",
            str(data.masks["train"]),
            "--train-case",
            str(data.cases["train"]),
            "--train-volume",
            str(data.volumes["train"]),
        ]
    assert main(train_args) == 0
    captured = capsys.readouterr()
    metrics = json.loads(captured.out)
    assert "Verifying" in captured.err
    train_records = _records(output)
    assert train_records["metrics"] == metrics
    assert metrics["steps_completed"] == 2 and metrics["best_step"] > 0
    best_path = output / "artifacts/best.pt"
    loaded = load_relational_trace_graph_checkpoint(best_path)
    assert loaded.checkpoint_role == "best_validation"
    assert loaded.training_random_seed == 7
    assert loaded.training_provenance == train_records["inputs.lock"]
    assert train_records["run"]["random_seed"] == 42
    assert train_records["run"]["input"]["validation"]["mask"]["random_seed"] == 19
    assert train_records["run"]["training_data"]["additional_train_partition_access"]
    assert train_records["run"]["geometry_features"]["midpoint_origin_m"] == list(
        loaded.preprocessing.midpoint_origin_m
    )
    assert (output / "artifacts/final.pt").is_file()
    assert any(
        parameter.abs().max() > 0 for parameter in loaded.model.decoder.head[-1].parameters()
    )
    checkpoint_hash = file_sha256(best_path)
    infer_config = write_trace_graph_config(
        tmp_path / "infer.yaml", trace_graph_prediction_config()
    )
    infer_output = tmp_path / "infer"
    args = [
        "interpolate",
        "relational-trace-graph",
        "--checkpoint",
        str(best_path),
        "--config",
        str(infer_config),
        "--interim",
        str(data.interim),
        "--processed",
        str(data.processed),
        "--mask",
        str(data.masks["test"]),
        "--case",
        str(data.cases["test"]),
        "--output",
        str(infer_output),
        "--json",
    ]
    if dense:
        args += ["--volume", str(data.volumes["test"])]
    assert main(args) == 0
    captured = capsys.readouterr()
    inferred = json.loads(captured.out)
    assert "Evaluating" in captured.err
    records = _records(infer_output)
    assert records["metrics"] == inferred
    assert records["run"]["input"]["mask"]["random_seed"] == 23
    assert records["run"]["prediction"]["layout"] == (
        "dense_volume" if dense else "native_trace_list"
    )
    assert train_records["run"]["prediction"]["layout"] == records["run"]["prediction"]["layout"]
    assert (
        file_sha256(best_path) == checkpoint_hash == records["inputs.lock"]["checkpoint"]["sha256"]
    )
    prediction = np.load(infer_output / "artifacts/prediction.npy", allow_pickle=False)
    assert prediction.ndim == (5 if dense else 2) and np.isfinite(prediction).all()
    assert (infer_output / "artifacts/query_index.parquet").is_file()
    for value in (*records.values(), *train_records.values()):
        json.dumps(value, allow_nan=False)
    assert inferred["evaluation_target"]["sample_count"] > 0
    with pytest.raises(FileExistsError, match="already exists"):
        _infer(data, infer_config, best_path, infer_output)
    with pytest.raises(FileExistsError, match="already exists"):
        _train(data, train_config, output)


def test_training_seeds_reuse_fixed_partition_mask_and_observed_training_pool(tmp_path):
    data = prepare_trace_graph_run_artifacts(tmp_path / "data")
    states, locks = [], []
    for seed in (7, 8):
        config = trace_graph_training_config()
        config["training"]["random_seed"] = seed
        config["training_data"]["pool"] = "mask_observed"
        path = write_trace_graph_config(tmp_path / f"train{seed}.yaml", config)
        output = tmp_path / f"run{seed}"
        _train(
            data,
            path,
            output,
            train_mask_dir=data.masks["train"],
            train_case_dir=data.cases["train"],
        )
        checkpoint = load_relational_trace_graph_checkpoint(output / "artifacts/final.pt")
        states.append(checkpoint.model.state_dict())
        locks.append(checkpoint.training_provenance)
        assert checkpoint.training_random_seed == seed
        assert locks[-1]["training_input"]["mask"]["random_seed"] == 17
        assert not locks[-1]["training_data"]["additional_train_partition_access"]
    assert locks[0]["source_inputs_lock"] == locks[1]["source_inputs_lock"]
    assert locks[0]["validation_inputs_lock"] == locks[1]["validation_inputs_lock"]
    assert any(not torch.equal(states[0][name], states[1][name]) for name in states[0])


@pytest.mark.parametrize(
    "pool,kwargs",
    [
        ("mask_observed", {}),
        ("all_train_traces", {"train_mask_dir": Path("mask")}),
        ("all_train_traces", {"train_volume_dir": Path("volume")}),
    ],
)
def test_missing_training_case_arguments_fail_before_training(tmp_path, pool, kwargs):
    config = trace_graph_training_config()
    config["training_data"]["pool"] = pool
    path = write_trace_graph_config(tmp_path / "train.yaml", config)
    with pytest.raises(ValueError, match="mask|case"):
        train_relational_trace_graph_run(
            config_path=path,
            interim_dir=tmp_path / "missing",
            processed_dir=tmp_path / "missing",
            validation_mask_dir=tmp_path / "missing",
            validation_case_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
            **kwargs,
        )
    assert not (tmp_path / "output").exists()


@pytest.fixture(scope="module")
def trained_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("trace_graph_run_contracts")
    data = prepare_trace_graph_run_artifacts(directory / "data", dense=True)
    config = write_trace_graph_config(directory / "train.yaml", trace_graph_training_config())
    output = directory / "train"
    _train(data, config, output)
    return data, output / "artifacts/best.pt"


@pytest.mark.parametrize("corruption", ["model_type", "input_hash", "time_grid"])
def test_checkpoint_mismatches_are_rejected_before_prediction(
    trained_inputs, tmp_path, monkeypatch, corruption
):
    from seis_interp.pipelines import interpolate_relational_trace_graph as pipeline

    data, original_checkpoint = trained_inputs
    payload = torch.load(original_checkpoint, map_location="cpu", weights_only=True)
    if corruption == "model_type":
        payload["model_type"] = "trace_graph"
    elif corruption == "input_hash":
        files = payload["training_provenance"]["source_inputs_lock"]["input_files"]
        files["interim"]["amplitudes.npy"]["sha256"] = "0" * 64
    else:
        times = [value + 0.001 for value in payload["preprocessing"]["time_s"]]
        payload["preprocessing"]["time_s"] = times
        payload["time"]["time_s"] = times
    checkpoint = tmp_path / "invalid.pt"
    torch.save(payload, checkpoint)
    config = write_trace_graph_config(tmp_path / "infer.yaml", trace_graph_prediction_config())

    def no_prediction(*args, **kwargs):
        pytest.fail("input mismatch must fail before the prediction function")

    monkeypatch.setattr(pipeline, "predict_relational_trace_graph", no_prediction)
    with pytest.raises(ValueError, match="model|hashes|time grid"):
        _infer(data, config, checkpoint, tmp_path / "infer")
    assert not (tmp_path / "infer").exists()


def test_changed_case_bound_input_fails_before_prediction(trained_inputs, tmp_path, monkeypatch):
    from seis_interp.pipelines import interpolate_relational_trace_graph as pipeline

    data, checkpoint = trained_inputs
    interim = tmp_path / "changed_interim"
    shutil.copytree(data.interim, interim)
    with (interim / "amplitudes.npy").open("ab") as stream:
        stream.write(b"changed")
    config = write_trace_graph_config(tmp_path / "infer.yaml", trace_graph_prediction_config())
    monkeypatch.setattr(
        pipeline,
        "predict_relational_trace_graph",
        lambda *args, **kwargs: pytest.fail("must verify hashes before prediction"),
    )
    with pytest.raises(ValueError, match="input_files"):
        interpolate_relational_trace_graph_run(
            config_path=config,
            checkpoint_path=checkpoint,
            interim_dir=interim,
            processed_dir=data.processed,
            mask_dir=data.masks["test"],
            case_dir=data.cases["test"],
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_optional_volume_must_use_checkpoint_time_selection(trained_inputs, tmp_path):
    data, original = trained_inputs
    payload = torch.load(original, map_location="cpu", weights_only=True)
    payload["preprocessing"]["fit_domain"]["time_samples"] = [0, 3]
    checkpoint = tmp_path / "different_time.pt"
    torch.save(payload, checkpoint)
    config = write_trace_graph_config(tmp_path / "infer.yaml", trace_graph_prediction_config())
    with pytest.raises(ValueError, match="time_samples"):
        _infer(data, config, checkpoint, tmp_path / "infer", volume_dir=data.volumes["test"])


def test_partition_seed_and_validation_partition_fail_before_training(trained_inputs, tmp_path):
    data, _ = trained_inputs
    config = trace_graph_training_config()
    config["project"]["random_seed"] = 19  # Validation mask seed is not the partition seed.
    path = write_trace_graph_config(tmp_path / "wrong_seed.yaml", config)
    with pytest.raises(ValueError, match="partition seed"):
        _train(data, path, tmp_path / "bad_seed")
    assert not (tmp_path / "bad_seed").exists()
    path = write_trace_graph_config(tmp_path / "train.yaml", trace_graph_training_config())
    with pytest.raises(ValueError, match="validation partition"):
        train_relational_trace_graph_run(
            config_path=path,
            interim_dir=data.interim,
            processed_dir=data.processed,
            validation_mask_dir=data.masks["test"],
            validation_case_dir=data.cases["test"],
            output_dir=tmp_path / "bad_validation",
        )
    assert not (tmp_path / "bad_validation").exists()


@pytest.fixture(scope="module")
def persistence_inputs(tmp_path_factory):
    return prepare_trace_graph_run_artifacts(
        tmp_path_factory.mktemp("trace_graph_persistence") / "data", dense=True
    )


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_first_validation_preserves_start_records_and_best_on_interruption(
    persistence_inputs, tmp_path, monkeypatch, error_type
):
    from seis_interp.pipelines import train_relational_trace_graph as pipeline

    data = persistence_inputs
    config = trace_graph_training_config()
    config["training"].update(max_steps=3, validation_interval=1)
    path = write_trace_graph_config(tmp_path / "train.yaml", config)
    output = tmp_path / "run"
    original_train = pipeline.train_relational_trace_graph
    initial_files = {}

    def check_start_records(*args, **kwargs):
        records = _records(output)
        assert records["run"]["status"] == "running"
        assert records["run"]["phase"] == "training"
        assert records["run"]["finished_at_utc"] is None
        assert records["run"]["checkpoints"] == {}
        assert records["run"]["training"]["steps_completed"] == 0
        assert "best_step" not in records["metrics"]
        assert not list((output / "artifacts").iterdir())
        for name in ("config.resolved.yaml", "inputs.lock.json"):
            initial_files[name] = (output / name).read_bytes()
        assert yaml.safe_load(initial_files["config.resolved.yaml"]) == config
        return original_train(*args, **kwargs)

    def stop_after_validation(message):
        if "validation step 1:" in message:
            # These must already be readable before the trainer returns.
            records = _records(output)
            best = load_relational_trace_graph_checkpoint(output / "artifacts/best.pt")
            assert best.global_step == records["metrics"]["best_step"] == 1
            assert best.selection_metrics == records["metrics"]["best_validation_metrics"]
            assert best.training_provenance == records["inputs.lock"]
            assert records["run"]["status"] == "running"
            assert records["run"]["training"]["steps_completed"] == 1
            raise error_type("intentional stop after first validation")

    monkeypatch.setattr(pipeline, "train_relational_trace_graph", check_start_records)
    with pytest.raises(error_type, match="intentional stop"):
        _train(
            data,
            path,
            output,
            validation_volume_dir=data.volumes["validation"],
            progress_reporter=stop_after_validation,
        )
    records = _records(output)
    best = load_relational_trace_graph_checkpoint(output / "artifacts/best.pt")
    assert best.global_step == records["metrics"]["best_step"] == 1
    assert best.selection_metrics == records["metrics"]["best_validation_metrics"]
    assert best.training_provenance == records["inputs.lock"]
    assert best.training_random_seed == config["training"]["random_seed"]
    assert records["run"]["status"] == (
        "interrupted" if error_type is KeyboardInterrupt else "failed"
    )
    assert records["run"]["finished_at_utc"] is not None
    assert records["run"]["error"]["type"] == error_type.__name__
    assert records["run"]["phase"] == "training"
    assert records["run"]["checkpoints"]["best"]["step"] == 1
    assert not (output / "artifacts/final.pt").exists()
    assert not (output / "artifacts/prediction.npy").exists()
    for name, contents in initial_files.items():
        assert (output / name).read_bytes() == contents


@pytest.mark.parametrize(
    "failing_function", ["predict_relational_trace_graph", "save_trace_graph_prediction"]
)
def test_post_training_prediction_failure_preserves_final_metrics_and_checkpoints(
    persistence_inputs, tmp_path, monkeypatch, failing_function
):
    from seis_interp.pipelines import train_relational_trace_graph as pipeline

    data = persistence_inputs
    config = trace_graph_training_config()
    path = write_trace_graph_config(tmp_path / "train.yaml", config)
    output = tmp_path / "run"

    def fail_after_training(*args, **kwargs):
        records = _records(output)
        assert records["run"]["status"] == "running"
        assert records["run"]["phase"] == "validation_prediction"
        assert records["metrics"]["steps_completed"] == config["training"]["max_steps"]
        for name in ("best", "final"):
            checkpoint = load_relational_trace_graph_checkpoint(output / f"artifacts/{name}.pt")
            assert checkpoint.selection_metrics == records["metrics"][f"{name}_validation_metrics"]
        raise RuntimeError("intentional prediction artifact failure")

    monkeypatch.setattr(pipeline, failing_function, fail_after_training)
    with pytest.raises(RuntimeError, match="intentional prediction artifact failure"):
        _train(data, path, output, validation_volume_dir=data.volumes["validation"])
    records = _records(output)
    assert records["run"]["status"] == "failed"
    assert records["run"]["phase"] == "validation_prediction"
    assert records["run"]["finished_at_utc"] is not None
    assert records["metrics"]["steps_completed"] == config["training"]["max_steps"]
    assert "prediction" not in records["run"]


def test_infeasible_episode_mask_fails_before_creating_run(persistence_inputs, tmp_path):
    config = trace_graph_training_config()
    config["training_mask"]["missing_fractions"] = [0.000001]
    path = write_trace_graph_config(tmp_path / "train.yaml", config)
    with pytest.raises(ValueError, match="both visible and hidden units"):
        _train(persistence_inputs, path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_frozen_cli_configuration_override_is_a_clean_error(tmp_path, capsys):
    config = trace_graph_prediction_config()
    config["graph"] = {"neighbors_per_relation": 99}
    path = write_trace_graph_config(tmp_path / "infer.yaml", config)
    args = ["interpolate", "relational-trace-graph", "--config", str(path)]
    for option in ("checkpoint", "interim", "processed", "mask", "case", "output"):
        args += [f"--{option}", str(tmp_path / option)]
    assert main([*args, "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported" in captured.err
