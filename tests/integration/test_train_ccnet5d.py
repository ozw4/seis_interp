from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from seis_interp import run_records
from seis_interp.data.c3_supervised_source import load_c3_supervised_source
from seis_interp.data.file_checksums import file_sha256
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


def test_nontrain_amplitudes_cannot_change_labels_rms_weights_or_history(tmp_path):
    runs = []
    for index, value in enumerate((None, -99999.0)):
        artifacts = prepare_ccnet5d_artifacts(tmp_path / f"data{index}", nontrain_value=value)
        path = write_ccnet5d_config(
            tmp_path / f"config{index}.yaml", ccnet5d_training_config(artifacts)
        )
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
