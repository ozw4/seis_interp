from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_study_013_amplitude_balancing as script
from seis_interp.configuration import (
    REPOSITORY_ROOT,
    get_required_config_value,
    load_resolved_config,
)


def test_study_013_config_locks_exactly_three_paired_conditions() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_013_amplitude_balancing" / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    expected = {
        "project.random_seed": 42,
        "sampling.random_trace_holdout_fraction": 0.20,
        "sampling.validation_fraction_of_holdout": 0.25,
        "normalization.coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
        "normalization.amplitude": "train_global_rms",
        "model.name": "siren",
        "model.input_features": 6,
        "model.hidden_width": 256,
        "model.hidden_layers": 4,
        "model.omega_0": 300.0,
        "model.hidden_omega": 1.0,
        "training.loss": "l2",
        "training.optimizer": "adam",
        "training.learning_rate": 1.0e-3,
        "training.batch_size": 5000,
        "training.total_updates": 50000,
        "training.report_interval": 500,
        "training.prediction_batch_size": 65536,
        "training.device": "cuda:0",
        "experiment.trace_count": 435,
        "experiment.batch_mode": "random_replacement",
        "experiment.replacement": True,
        "experiment.huber_delta": 1.0,
    }

    assert get_required_config_value(config, "study.status") in {"active", "completed"}
    assert {path: get_required_config_value(config, path) for path in expected} == expected
    assert config["experiment"]["conditions"] == [
        {
            "label": "global_rms_control",
            "amplitude_scaling": "global_rms",
            "loss": "l2",
        },
        {
            "label": "per_trace_rms",
            "amplitude_scaling": "per_trace_rms",
            "loss": "l2",
        },
        {
            "label": "huber_global_rms",
            "amplitude_scaling": "global_rms",
            "loss": "huber",
        },
    ]
    assert config["training"]["batch_size"] * config["training"]["total_updates"] == 250_000_000
    assert config["training"]["total_updates"] // config["training"]["report_interval"] == 100

    disallowed_experiment_keys = {
        "correlation_weight",
        "correlation_eps",
        "traces_per_update",
        "samples_per_trace",
        "patch_starts",
        "temporal_patch_overlap_fraction",
        "trace_counts",
        "carry_model_state",
        "carry_optimizer_state",
        "checkpoint",
    }
    assert disallowed_experiment_keys.isdisjoint(config["experiment"])
    assert set(config["experiment"]) == {
        "trace_count",
        "batch_mode",
        "replacement",
        "huber_delta",
        "conditions",
    }
    for condition in config["experiment"]["conditions"]:
        assert set(condition) == {"label", "amplitude_scaling", "loss"}


def test_study_013_script_forwards_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    received: dict[str, object] = {}

    def recording_run(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"decision": "per_trace_rms_strong_fit"}

    monkeypatch.setattr(script, "run_amplitude_balancing", recording_run)
    config = tmp_path / "config.yaml"
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    output_root = tmp_path / "runs"

    exit_code = script.main(
        [
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output-root",
            str(output_root),
            "--device",
            "cuda:1",
        ]
    )

    assert exit_code == 0
    assert received == {
        "config_path": config,
        "interim_dir": interim,
        "processed_dir": processed,
        "output_root": output_root,
        "device_override": "cuda:1",
    }


def test_study_013_script_reports_expected_pipeline_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def failing_run(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise ValueError("invalid amplitude condition")

    monkeypatch.setattr(script, "run_amplitude_balancing", failing_run)

    exit_code = script.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--interim",
            str(tmp_path / "interim"),
            "--processed",
            str(tmp_path / "processed"),
            "--output-root",
            str(tmp_path / "runs"),
        ]
    )

    assert exit_code == 1
    assert capsys.readouterr().err == (
        "Study 013 amplitude balancing failed: invalid amplitude condition\n"
    )
