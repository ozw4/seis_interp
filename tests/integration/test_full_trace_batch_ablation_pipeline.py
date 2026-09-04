from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
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


_CONDITION_ORDER = [
    "small_batch_control",
    "full_trace_batch",
    "full_trace_batch_correlation",
    "full_trace_batch_per_trace_rms",
    "full_trace_batch_correlation_per_trace_rms",
]
_TRACE_BATCH_CONDITIONS = set(_CONDITION_ORDER[1:])
_CORRELATION_CONDITIONS = {
    "full_trace_batch_correlation",
    "full_trace_batch_correlation_per_trace_rms",
}
_PER_TRACE_CONDITIONS = {
    "full_trace_batch_per_trace_rms",
    "full_trace_batch_correlation_per_trace_rms",
}


def _build_full_trace_batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
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
            "omega_0": 30.0,
            "hidden_omega": 30.0,
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
        "correlation_weight": 0.3,
        "correlation_eps": 1.0e-4,
        "conditions": [
            {
                "label": label,
                "batch_mode": batch_mode,
                "correlation": correlation,
                "amplitude_scaling": amplitude_scaling,
            }
            for label, batch_mode, correlation, amplitude_scaling in reversed(
                pipeline._FULL_TRACE_BATCH_ABLATION_CONDITIONS
            )
        ],
    }
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return config, interim, processed


def test_full_trace_batch_ablation_writes_five_isolated_immutable_cpu_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_trace_batch_fixture(tmp_path)
    output_root = tmp_path / "runs" / pipeline.FULL_TRACE_BATCH_ABLATION_STUDY_ID
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260827T060708Z")

    summary = pipeline.run_full_trace_batch_ablation(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    prefix = f"20260827T060708Z_{git_commit[:7]}"
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
        "escape_reproduced",
        "correlation_weight",
        "correlation_eps",
        "per_trace_scale_stats",
        "conditions",
    }
    assert summary["study_id"] == pipeline.FULL_TRACE_BATCH_ABLATION_STUDY_ID
    assert summary["git_commit"] == git_commit
    assert summary["correlation_weight"] == 0.3
    assert summary["correlation_eps"] == 1.0e-4
    assert [condition["label"] for condition in summary["conditions"]] == _CONDITION_ORDER

    # Two training traces with five samples each: the complete-trace batch covers
    # all ten points per update while the control keeps the configured four.
    full_trace_batch_size = 2 * 5
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
        is_trace_batch = label in _TRACE_BATCH_CONDITIONS
        uses_correlation = label in _CORRELATION_CONDITIONS
        expected_scaling = "per_trace_rms" if label in _PER_TRACE_CONDITIONS else "global_rms"
        expected_batch_mode = "random_complete_traces" if is_trace_batch else "random_replacement"
        expected_batch_size = full_trace_batch_size if is_trace_batch else 4
        expected_weight = 0.3 if uses_correlation else 0.0

        assert saved_config["training"]["device"] == "cpu"
        active_condition = saved_config["experiment"]["active_condition"]
        assert active_condition == {
            "label": label,
            "batch_mode": expected_batch_mode,
            "correlation_weight": expected_weight,
            "amplitude_scaling": expected_scaling,
        }

        assert metrics["amplitude_scaling"] == expected_scaling
        assert metrics["batch_mode"] == expected_batch_mode
        assert metrics["batch_size"] == expected_batch_size
        assert metrics["trace_count"] == 2
        assert metrics["updates_completed"] == 2
        assert [row["step"] for row in metrics["history"]] == [1, 2]
        if is_trace_batch:
            assert metrics["traces_per_update"] == 2
        else:
            assert "traces_per_update" not in metrics
        if uses_correlation:
            assert metrics["correlation_weight"] == 0.3
            assert metrics["correlation_eps"] == 1.0e-4
            assert metrics["loss_semantics"] == "mse_plus_trace_correlation"
        else:
            assert "correlation_weight" not in metrics
            assert "loss_semantics" not in metrics
        if label in _PER_TRACE_CONDITIONS:
            assert metrics["per_trace_scale_stats"] == summary["per_trace_scale_stats"]
        else:
            assert "per_trace_scale_stats" not in metrics

        training_contract = inputs_lock["training"]
        assert training_contract["condition"] == label
        assert training_contract["batch_mode"] == expected_batch_mode
        assert training_contract["batch_size"] == expected_batch_size
        assert training_contract["replacement"] is (not is_trace_batch)
        assert training_contract["amplitude_scaling"] == expected_scaling
        assert training_contract["point_evaluations"] == expected_batch_size * 2
        if uses_correlation:
            assert training_contract["correlation_weight"] == 0.3
        else:
            assert "correlation_weight" not in training_contract

        assert run_metadata["study_id"] == pipeline.FULL_TRACE_BATCH_ABLATION_STUDY_ID
        assert run_metadata["batch_mode"] == expected_batch_mode
        assert run_metadata["amplitude_scaling"] == expected_scaling
        if uses_correlation:
            assert run_metadata["correlation_weight"] == 0.3
        else:
            assert "correlation_weight" not in run_metadata

        for path in run_directory.iterdir():
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    summary_by_label = {str(condition["label"]): condition for condition in summary["conditions"]}
    for label, condition_summary in summary_by_label.items():
        metrics = metrics_by_label[label]
        assert condition_summary["classification"] == metrics["classification"]
        assert condition_summary["run_directory"] == f"{prefix}_{label}"
        assert condition_summary["batch_size"] == metrics["batch_size"]
        best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
        assert condition_summary["best_report"] == best_row
        assert condition_summary["final_report"] == metrics["history"][-1]

    control_classification = str(metrics_by_label["small_batch_control"]["classification"])
    reproduction_classification = str(
        metrics_by_label["full_trace_batch_correlation_per_trace_rms"]["classification"]
    )
    assert summary["control_validity"] is (control_classification == "near_zero")
    assert summary["escape_reproduced"] is (reproduction_classification != "near_zero")
    assert summary["decision"] == pipeline.full_trace_batch_ablation_summary_decision(
        control_classification=control_classification,
        full_trace_batch_classification=str(metrics_by_label["full_trace_batch"]["classification"]),
        reproduction_classification=reproduction_classification,
    )
    _assert_finite_json(summary)
    for path in output_root.rglob("*.json"):
        _assert_finite_json(json.loads(path.read_text(encoding="utf-8")))

    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_full_trace_batch_ablation(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
