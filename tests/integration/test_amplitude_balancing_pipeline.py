from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.trace_splits import TRAIN_SPLIT
from tests.fixtures.siren_experiment import build_experiment_fixture


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_finite_json(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_json(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_finite_json(child)


_CONDITION_ORDER = ["global_rms_control", "per_trace_rms", "huber_global_rms"]


def _build_amplitude_balancing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, interim, _ = build_experiment_fixture(tmp_path)
    processed = tmp_path / "processed_seed42"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.5,
        validation_fraction_of_holdout=0.5,
        random_seed=42,
        config_source="studies/synthetic/config.yaml",
    )
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["project"]["random_seed"] = 42
    config_data["model"].update(
        {
            "hidden_layers": 2,
            "omega_0": 300.0,
            "hidden_omega": 1.0,
        }
    )
    config_data["training"].update(
        {
            "batch_size": 4,
            "total_updates": 2,
            "report_interval": 1,
            "prediction_batch_size": 3,
        }
    )
    # Reversed input order demonstrates that execution order does not choose the result.
    config_data["experiment"] = {
        "trace_count": 2,
        "batch_mode": "random_replacement",
        "replacement": True,
        "huber_delta": 1.0,
        "conditions": [
            {
                "label": "huber_global_rms",
                "amplitude_scaling": "global_rms",
                "loss": "huber",
            },
            {
                "label": "per_trace_rms",
                "amplitude_scaling": "per_trace_rms",
                "loss": "l2",
            },
            {
                "label": "global_rms_control",
                "amplitude_scaling": "global_rms",
                "loss": "l2",
            },
        ],
    }
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return config, interim, processed


def test_amplitude_balancing_writes_three_isolated_immutable_cpu_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_amplitude_balancing_fixture(tmp_path)
    output_root = tmp_path / "runs" / pipeline.AMPLITUDE_BALANCING_STUDY_ID
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260827T040506Z")

    actual_sampler = pipeline.RandomPointSampler
    sampler_amplitudes: list[np.ndarray] = []
    sampler_rows: list[np.ndarray] = []
    sampler_seeds: list[int] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            sampler_amplitudes.append(np.asarray(args[2]).copy())
            sampler_rows.append(np.asarray(args[3]).copy())
            sampler_seeds.append(int(kwargs["random_seed"]))
            self._delegate = actual_sampler(*args, **kwargs)

        def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
            return self._delegate.sample(batch_size)

    monkeypatch.setattr(pipeline, "RandomPointSampler", RecordingSampler)

    summary = pipeline.run_amplitude_balancing(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    prefix = f"20260827T040506Z_{git_commit[:7]}"
    expected_run_ids = [f"{prefix}_{label}" for label in _CONDITION_ORDER]
    run_directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    summary_path = output_root / f"{prefix}_summary.json"
    assert sorted(path.name for path in run_directories) == sorted(expected_run_ids)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert set(summary) == {
        "study_id",
        "git_commit",
        "generated_at_utc",
        "decision",
        "control_validity",
        "point_evaluations_per_condition",
        "per_trace_scale_stats",
        "conditions",
    }
    assert summary["study_id"] == pipeline.AMPLITUDE_BALANCING_STUDY_ID
    assert summary["git_commit"] == git_commit
    assert summary["point_evaluations_per_condition"] == 8
    assert [condition["label"] for condition in summary["conditions"]] == _CONDITION_ORDER

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    training_rows = sorted(split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"])

    # All three samplers see the same training rows and seed; the per-trace
    # condition sees unit-RMS training targets and the others share the
    # unchanged global-RMS targets.
    assert sampler_seeds == [42, 42, 42]
    assert all(sorted(rows) == training_rows for rows in sampler_rows)
    amplitudes_by_label = dict(zip(_CONDITION_ORDER, sampler_amplitudes, strict=True))
    np.testing.assert_array_equal(
        amplitudes_by_label["global_rms_control"],
        amplitudes_by_label["huber_global_rms"],
    )
    per_trace_training = amplitudes_by_label["per_trace_rms"][training_rows]
    per_trace_rms = np.sqrt(np.mean(np.square(per_trace_training.astype(np.float64)), axis=1))
    np.testing.assert_allclose(per_trace_rms, np.ones(len(training_rows)), rtol=1e-5)
    control_training = amplitudes_by_label["global_rms_control"][training_rows]
    expected_scales = np.sqrt(np.mean(np.square(control_training.astype(np.float64)), axis=1))
    assert summary["per_trace_scale_stats"] == {
        "min": float(np.min(expected_scales)),
        "median": float(np.median(expected_scales)),
        "max": float(np.max(expected_scales)),
    }

    metrics_by_label: dict[str, dict[str, object]] = {}
    for run_directory in run_directories:
        assert sorted(path.name for path in run_directory.iterdir()) == [
            "config.resolved.yaml",
            "inputs.lock.json",
            "metrics.json",
            "run.json",
        ]
        saved_config = yaml.safe_load(
            (run_directory / "config.resolved.yaml").read_text(encoding="utf-8")
        )
        metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
        inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
        run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        label = str(metrics["condition"])
        metrics_by_label[label] = metrics
        expected_scaling = "per_trace_rms" if label == "per_trace_rms" else "global_rms"
        expected_loss = "huber" if label == "huber_global_rms" else "l2"

        assert saved_config["training"]["loss"] == expected_loss
        assert saved_config["training"]["device"] == "cpu"
        active_condition = saved_config["experiment"]["active_condition"]
        assert active_condition["label"] == label
        assert active_condition["amplitude_scaling"] == expected_scaling
        assert active_condition["loss"] == expected_loss

        assert metrics["amplitude_scaling"] == expected_scaling
        assert metrics["loss_name"] == expected_loss
        assert metrics["selected_array_rows"] == training_rows
        assert metrics["trace_count"] == 2
        assert metrics["updates_completed"] == 2
        assert [row["step"] for row in metrics["history"]] == [1, 2]

        training_contract = inputs_lock["training"]
        assert training_contract["condition"] == label
        assert training_contract["amplitude_scaling"] == expected_scaling
        assert training_contract["loss_name"] == expected_loss
        assert run_metadata["study_id"] == pipeline.AMPLITUDE_BALANCING_STUDY_ID
        assert run_metadata["amplitude_scaling"] == expected_scaling
        assert run_metadata["loss_name"] == expected_loss

        if label == "huber_global_rms":
            assert saved_config["training"]["huber_delta"] == 1.0
            assert active_condition["huber_delta"] == 1.0
            assert metrics["huber_delta"] == 1.0
            assert training_contract["huber_delta"] == 1.0
            assert run_metadata["huber_delta"] == 1.0
        else:
            assert "huber_delta" not in saved_config["training"]
            assert "huber_delta" not in active_condition
            assert "huber_delta" not in metrics
            assert "huber_delta" not in training_contract
            assert "huber_delta" not in run_metadata
        if label == "per_trace_rms":
            assert metrics["per_trace_scale_stats"] == summary["per_trace_scale_stats"]
        else:
            assert "per_trace_scale_stats" not in metrics

        for path in run_directory.iterdir():
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    summary_by_label = {str(condition["label"]): condition for condition in summary["conditions"]}
    for label, condition_summary in summary_by_label.items():
        metrics = metrics_by_label[label]
        assert condition_summary["classification"] == metrics["classification"]
        assert condition_summary["run_directory"] == f"{prefix}_{label}"
        best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
        assert condition_summary["best_report"] == best_row
        assert condition_summary["final_report"] == metrics["history"][-1]

    control_classification = str(metrics_by_label["global_rms_control"]["classification"])
    assert summary["control_validity"] is (control_classification == "near_zero")
    assert summary["decision"] == pipeline.amplitude_balancing_summary_decision(
        control_classification=control_classification,
        per_trace_classification=str(metrics_by_label["per_trace_rms"]["classification"]),
    )
    _assert_finite_json(summary)
    for path in output_root.rglob("*.json"):
        _assert_finite_json(json.loads(path.read_text(encoding="utf-8")))

    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_amplitude_balancing(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
