from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from seis_interp import run_records
from seis_interp.cli import main
from seis_interp.data.c3_supervised_source import load_c3_supervised_source
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines import train_ccnet5d as train_ccnet5d_pipeline
from seis_interp.pipelines.train_ccnet5d import train_ccnet5d_run
from seis_interp.training.ccnet5d_checkpoints import load_ccnet5d_checkpoint
from tests.fixtures.ccnet5d_artifacts import prepare_ccnet5d_artifacts
from tests.fixtures.ccnet5d_runs import ccnet5d_training_config, write_ccnet5d_config


def _run(artifacts, config, output, **kwargs):
    return train_ccnet5d_run(
        config_path=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        output_dir=output,
        **kwargs,
    )


def test_training_records_checkpoints_cadence_and_final_timestamp(tmp_path, monkeypatch):
    artifacts = prepare_ccnet5d_artifacts(tmp_path / "data", shuffle_tables=True)
    config = ccnet5d_training_config(artifacts)
    path = write_ccnet5d_config(tmp_path / "config.yaml", config)
    output = tmp_path / "run"
    timestamps = []

    def timestamp():
        if timestamps:
            assert (output / "artifacts/best.pt").is_file()
            assert (output / "artifacts/final.pt").is_file()
        else:
            assert not output.exists()
        timestamps.append("2026-09-07T00:00:00Z")
        return timestamps[-1]

    monkeypatch.setattr(run_records, "utc_timestamp", timestamp)
    monkeypatch.setattr(
        run_records,
        "current_git_metadata",
        lambda: {"git_commit": "a" * 40, "git_worktree_dirty": False},
    )
    messages = []
    metrics = _run(
        artifacts, path, output, progress_reporter=messages.append, device_override="cpu"
    )
    assert len(timestamps) == 2
    assert messages
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()) == [
        "artifacts/best.pt",
        "artifacts/final.pt",
        "artifacts/patch_plan.json",
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    records = {
        name: json.loads((output / f"{name}.json").read_text())
        for name in ("run", "inputs.lock", "metrics")
    }
    for record in records.values():
        json.dumps(record, allow_nan=False)
    assert records["metrics"] == metrics
    assert metrics["steps_completed"] == 4
    assert metrics["epochs_completed"] == 2
    assert [row["step"] for row in metrics["selection_history"]] == [3, 4]
    assert [row["learning_rate"] for row in metrics["training_history"]] == [
        0.001,
        0.001,
        0.0001,
        0.0001,
    ]
    source = load_c3_supervised_source(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        fit_region=artifacts.fit_region,
        selection_region=artifacts.selection_region,
    )
    for role in ("best", "final"):
        loaded = load_ccnet5d_checkpoint(output / f"artifacts/{role}.pt")
        assert loaded.checkpoint_role == ("best_selection" if role == "best" else "final")
        assert loaded.global_step == (metrics["best_step"] if role == "best" else 4)
        assert loaded.amplitude_rms == source.amplitude_rms
        assert loaded.training_provenance == records["inputs.lock"]
        assert loaded.training_provenance["source_inputs_lock"] == source.inputs_lock
        assert loaded.training_provenance["patch_plan_sha256"] == file_sha256(
            output / "artifacts/patch_plan.json"
        )
        assert loaded.model.constructor_config() == {
            key: value for key, value in config["model"].items() if key != "name"
        }
    run = records["run"]
    assert run["status"] == "success" and not run["git_worktree_dirty"]
    assert run["device"] == "cpu"
    assert run["training_regime"] == "supervised_train_partition"
    assert run["training"]["random_seed"] == 7 and run["patches"]["random_seed"] == 42
    assert run["random_seed"] == 42
    assert (
        run["model"]["parameter_dtype"]
        == run["training"]["input_dtype"]
        == run["training"]["target_dtype"]
        == "float32"
    )
    assert yaml.safe_load((output / "config.resolved.yaml").read_text()) == config


def test_reproducible_training_seed_separate_from_fixed_source_and_plan(tmp_path):
    artifacts = prepare_ccnet5d_artifacts(tmp_path / "data")
    outputs = []
    for index, seed in enumerate((7, 7, 8)):
        config = ccnet5d_training_config(artifacts)
        config["training"]["random_seed"] = seed
        path = write_ccnet5d_config(tmp_path / f"config{index}.yaml", config)
        output = tmp_path / f"run{index}"
        metrics = _run(artifacts, path, output)
        outputs.append((output, metrics, load_ccnet5d_checkpoint(output / "artifacts/final.pt")))
    first, repeat, different = outputs
    assert first[1] == repeat[1]
    for name, state in first[2].model.state_dict().items():
        assert torch.equal(state, repeat[2].model.state_dict()[name])
    assert any(
        not torch.equal(state, different[2].model.state_dict()[name])
        for name, state in first[2].model.state_dict().items()
    )
    for other in (repeat, different):
        assert (
            first[2].training_provenance["source_inputs_lock"]
            == other[2].training_provenance["source_inputs_lock"]
        )
        assert (first[0] / "artifacts/patch_plan.json").read_bytes() == (
            other[0] / "artifacts/patch_plan.json"
        ).read_bytes()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_nontrain_amplitudes_cannot_change_labels_rms_weights_or_history(tmp_path, batch_size):
    runs = []
    for index, value in enumerate((None, -99999.0)):
        artifacts = prepare_ccnet5d_artifacts(tmp_path / f"data{index}", nontrain_value=value)
        config = ccnet5d_training_config(artifacts)
        config["training"]["batch_size"] = batch_size
        path = write_ccnet5d_config(tmp_path / f"config{index}.yaml", config)
        output = tmp_path / f"run{index}"
        runs.append((_run(artifacts, path, output), output))
    assert runs[0][0] == runs[1][0]
    for role in ("best", "final"):
        left, right = [load_ccnet5d_checkpoint(path / f"artifacts/{role}.pt") for _, path in runs]
        assert left.amplitude_rms == right.amplitude_rms
        assert left.training_provenance != right.training_provenance
        for name, state in left.model.state_dict().items():
            assert torch.equal(state, right.model.state_dict()[name])


@pytest.mark.parametrize("problem", ["overlap", "nontrain", "hash", "typo", "seed", "dataset"])
def test_preflight_rejects_without_output(tmp_path, problem):
    artifacts = prepare_ccnet5d_artifacts(tmp_path / "data")
    config = ccnet5d_training_config(artifacts)
    if problem == "overlap":
        config["supervision"]["selection_region"] = artifacts.fit_region
    elif problem == "nontrain":
        config["supervision"]["fit_region"]["source_line"] = [2, 3]
    elif problem == "hash":
        np.save(artifacts.interim / "amplitudes.npy", np.zeros((1, 5), dtype=np.float32))
    elif problem == "typo":
        config["training"]["max_epoch"] = 3
    elif problem == "seed":
        config["project"]["random_seed"] = 43
    elif problem == "dataset":
        config["data"]["dataset_id"] = "wrong"
    path = write_ccnet5d_config(tmp_path / "config.yaml", config)
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        _run(artifacts, path, output)
    assert not output.exists()


def test_zero_energy_selection_targets_fail_before_model_optimizer_or_output(tmp_path, monkeypatch):
    artifacts = prepare_ccnet5d_artifacts(tmp_path / "data", nonfit_value=0.0)
    config = write_ccnet5d_config(tmp_path / "config.yaml", ccnet5d_training_config(artifacts))
    output = tmp_path / "run"

    def unexpected_model(*args, **kwargs):
        pytest.fail("model must not be initialized before selection target preflight")

    def unexpected_optimizer(*args, **kwargs):
        pytest.fail("optimizer must not be initialized before selection target preflight")

    monkeypatch.setattr(train_ccnet5d_pipeline, "CCNet5D", unexpected_model)
    monkeypatch.setattr(torch.optim, "Adam", unexpected_optimizer)

    with pytest.raises(ValueError, match="selection target reference energy"):
        _run(artifacts, config, output)

    assert not output.exists()


def test_existing_output_is_unchanged(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        train_ccnet5d_run(
            config_path=tmp_path / "missing",
            interim_dir=tmp_path,
            processed_dir=tmp_path,
            output_dir=output,
        )
    assert marker.read_text() == "unchanged"
    assert list(output.iterdir()) == [marker]


def test_train_cli_end_to_end_json_and_progress(tmp_path, capsys):
    artifacts = prepare_ccnet5d_artifacts(tmp_path / "data")
    config = write_ccnet5d_config(tmp_path / "config.yaml", ccnet5d_training_config(artifacts))
    output = tmp_path / "run"
    result = main(
        [
            "train",
            "ccnet5d",
            "--config",
            str(config),
            "--interim",
            str(artifacts.interim),
            "--processed",
            str(artifacts.processed),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    summary = json.loads(captured.out)
    assert summary["steps_completed"] == 4
    assert captured.err
    assert (output / "artifacts/best.pt").is_file()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("benchmark_option", ["missing", True, False])
def test_batch_checkpoint_and_benchmark_after_seed_keep_default_kwargs(
    tmp_path, monkeypatch, batch_size, benchmark_option
):
    artifacts = prepare_ccnet5d_artifacts(tmp_path / "data")
    config = ccnet5d_training_config(artifacts)
    config["patches"]["fit_count"] = 3
    config["training"]["batch_size"] = batch_size
    if benchmark_option != "missing":
        config["training"]["cudnn_benchmark"] = benchmark_option
    path = write_ccnet5d_config(tmp_path / "training.yaml", config)
    seed = train_ccnet5d_pipeline.seed_global_model_initialization
    train = train_ccnet5d_pipeline.train_ccnet5d
    constructor = train_ccnet5d_pipeline.CCNet5D
    resources = run_records.runtime_resource_metadata
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    modes = []
    options = []

    def seed_with_cuda_behavior(*args, **kwargs):
        seed(*args, **kwargs)
        torch.backends.cudnn.benchmark = True

    def construct(*args, **kwargs):
        modes.append(torch.backends.cudnn.benchmark)
        return constructor(*args, **kwargs)

    def record_training(*args, **kwargs):
        options.append(kwargs.copy())
        return train(*args, **kwargs)

    def record_resources(device):
        modes.append(torch.backends.cudnn.benchmark)
        return resources(device)

    monkeypatch.setattr(
        train_ccnet5d_pipeline, "seed_global_model_initialization", seed_with_cuda_behavior
    )
    monkeypatch.setattr(train_ccnet5d_pipeline, "CCNet5D", construct)
    monkeypatch.setattr(train_ccnet5d_pipeline, "train_ccnet5d", record_training)
    monkeypatch.setattr(run_records, "runtime_resource_metadata", record_resources)
    output = tmp_path / "run"
    result = _run(artifacts, path, output)
    expected_steps = 2 * ((3 + batch_size - 1) // batch_size)
    assert result["steps_completed"] == expected_steps
    assert [row["step"] for row in result["selection_history"]] == [3, expected_steps]
    assert modes and all(value is (benchmark_option is not False) for value in modes)
    assert len(options) == 1
    assert options[0].get("batch_size", 1) == batch_size
    assert ("batch_size" in options[0]) is (batch_size > 1)
    assert ("cudnn_benchmark" in options[0]) is (benchmark_option is False)
    run = json.loads((output / "run.json").read_text())
    if benchmark_option == "missing":
        assert "cudnn_benchmark" not in run["training"]
    else:
        assert run["training"]["cudnn_benchmark"] is benchmark_option
    loaded = load_ccnet5d_checkpoint(output / "artifacts/final.pt")
    assert loaded.global_step == expected_steps
    assert loaded.checkpoint_role == "final"
    assert "batch_size" not in loaded.model.constructor_config()
    assert "cudnn_benchmark" not in loaded.model.constructor_config()
    assert loaded.training_provenance == json.loads((output / "inputs.lock.json").read_text())
    assert yaml.safe_load((output / "config.resolved.yaml").read_text()) == config
    if batch_size > 1:
        assert [row["sample_count"] for row in result["training_history"]] == [16, 8, 16, 8]
        assert all(
            row["loss_aggregation"] == "sample_weighted_mean" for row in result["training_history"]
        )
    else:
        assert all("sample_count" not in row for row in result["training_history"])


@pytest.mark.parametrize(
    "key,value",
    [("batch_size", value) for value in (0, -1, True, 1.5, None, "2")]
    + [("cudnn_benchmark", value) for value in (None, 0, 1, "false", [])],
)
def test_invalid_training_resource_options_reject_before_source_read(
    tmp_path, monkeypatch, key, value
):
    artifacts = prepare_ccnet5d_artifacts(tmp_path / "data")
    config = ccnet5d_training_config(artifacts)
    config["training"][key] = value
    path = write_ccnet5d_config(tmp_path / "config.yaml", config)
    monkeypatch.setattr(
        train_ccnet5d_pipeline,
        "load_c3_supervised_source",
        lambda **kwargs: pytest.fail("invalid configuration must fail before source reads"),
    )
    with pytest.raises(ValueError, match=key):
        _run(artifacts, path, tmp_path / "run")
    assert not (tmp_path / "run").exists()
