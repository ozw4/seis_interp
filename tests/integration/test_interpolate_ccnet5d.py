from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_interp import run_records
from seis_interp.data.c3_poc_inputs import C3_RANDOM80_POC_DATASET_ID
from seis_interp.data.c3_volume_adapter import load_observed_c3_volume
from seis_interp.pipelines import interpolate_ccnet5d as ccnet_pipeline
from seis_interp.pipelines.interpolate_ccnet5d import interpolate_ccnet5d_run
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.ccnet5d_poc_checkpoints import load_ccnet5d_poc_checkpoint
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume
from tests.fixtures.c3_volume_run_artifacts import (
    PreparedC3VolumeRunArtifacts,
    prepare_c3_volume_run_artifacts,
)


def _artifacts(tmp_path: Path, *, target_offset: float = 0.0) -> PreparedC3VolumeRunArtifacts:
    return prepare_c3_volume_run_artifacts(
        tmp_path,
        dataset_id=C3_RANDOM80_POC_DATASET_ID,
        missing_fraction=0.8,
        target_offset=target_offset,
        time_sample_count=4,
        receiver_y_count=3,
    )


def _dimensions(artifacts: PreparedC3VolumeRunArtifacts) -> C3BenchmarkDimensions:
    shape = tuple(artifacts.volume_metadata["shape"])
    return C3BenchmarkDimensions(
        time_range=(0, shape[0]),
        sail_line_numbers=(2, 3),
        shape=shape,  # type: ignore[arg-type]
    )


def _config(artifacts: PreparedC3VolumeRunArtifacts) -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "data": {"dataset_id": C3_RANDOM80_POC_DATASET_ID},
        "model": {
            "name": "ccnet5d",
            "hidden_channels": 1,
            "intermediate_channels": 1,
            "kernel_size": 1,
            "output_activation": "linear",
        },
        "patches": {
            "shape": list(artifacts.volume_metadata["shape"]),
            "inner_mask_fraction": 0.5,
            "mask_kind": "random_trace",
            "random_seed": 19,
        },
        "training": {
            "random_seed": 23,
            "optimizer": "adam",
            "loss": "masked_trace_relative_mse",
            "amplitude_scaling": "observed_volume_global_rms",
            "learning_rate": 1.0e-3,
            "max_steps": 2,
            "report_interval": 1,
            "device": "cpu",
        },
        "interpolation_mask": {
            "partition": "test",
            "kind": "random_trace",
            "missing_fraction": 0.8,
        },
        "benchmark_case": {"id": "synthetic_case"},
        "benchmark_volume": {
            "id": "synthetic_volume",
            "selection": deepcopy(artifacts.volume_metadata["selection"]),
        },
        "prediction": {"core_shape": list(artifacts.volume_metadata["shape"])},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }


def _run(
    tmp_path: Path,
    artifacts: PreparedC3VolumeRunArtifacts,
    *,
    progress_reporter=None,
) -> tuple[Path, dict[str, object]]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(artifacts), sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"
    metrics = interpolate_ccnet5d_run(
        config_path=config_path,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        output_dir=output,
        progress_reporter=progress_reporter,
        dimensions=_dimensions(artifacts),
    )
    return output, metrics


def test_observed_only_fit_predict_evaluate_writes_final_replayable_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_records,
        "current_git_metadata",
        lambda: {"git_commit": "c" * 40, "git_worktree_dirty": False},
    )
    artifacts = _artifacts(tmp_path / "data")
    progress: list[str] = []

    output, metrics = _run(tmp_path, artifacts, progress_reporter=progress.append)

    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == [
        "artifacts/final.pt",
        "artifacts/prediction.npy",
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    prediction = np.load(output / "artifacts/prediction.npy", allow_pickle=False)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )
    expected_scale = compute_observed_global_rms(
        observed.values,
        observed.observed_trace_mask,
    )
    checkpoint = load_ccnet5d_poc_checkpoint(output / "artifacts/final.pt")
    assert checkpoint.amplitude_scale == expected_scale
    assert checkpoint.optimizer_updates == 2
    replay = predict_ccnet5d_volume(
        checkpoint.model,
        observed,
        amplitude_rms=checkpoint.amplitude_scale,
        core_shape=tuple(observed.values.shape),
        device="cpu",
    )
    np.testing.assert_array_equal(replay.values, prediction)
    np.testing.assert_array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask],
    )
    assert np.all(np.isfinite(prediction))

    for name in ("metrics", "run", "inputs.lock"):
        record = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        json.dumps(record, allow_nan=False)
    stored_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert stored_metrics == metrics
    assert metrics["method"] == "ccnet5d"
    assert metrics["training_domain"] == "O_with_inner_pseudo_mask"
    assert metrics["loss"] == "masked_trace_relative_mse"
    assert metrics["checkpoint_role"] == "final"
    assert metrics["normalization"] == {
        "type": "global_rms",
        "source": "O_only",
        "scale": expected_scale,
    }
    for metric in ("snr_db", "rmse", "relative_l2", "mean_trace_relative_mse"):
        assert metric in metrics["evaluation_target"]
    assert (
        metrics["evaluation_target"]["covered_target_trace_count"]
        == metrics["evaluation_target"]["target_trace_count"]
    )
    assert run["training"]["optimizer_updates"] == 2
    assert run["training"]["validation"] is False
    assert run["training"]["best_checkpoint_selection"] is False
    assert run["coverage"]["minimum_target_coverage_count"] >= 1
    assert run["resources"]["training_seconds"] >= 0.0
    assert run["resources"]["prediction_seconds"] >= 0.0
    assert run["resources"]["end_to_end_seconds"] >= 0.0
    assert len(progress) >= 7


def test_target_truth_change_cannot_change_training_state_or_pre_evaluation_prediction(
    tmp_path: Path,
) -> None:
    runs = []
    for name, offset in (("base", 0.0), ("changed", 10000.0)):
        root = tmp_path / name
        artifacts = _artifacts(root / "data", target_offset=offset)
        output, metrics = _run(root, artifacts)
        checkpoint = load_ccnet5d_poc_checkpoint(output / "artifacts/final.pt")
        runs.append(
            (
                {key: value.clone() for key, value in checkpoint.model.state_dict().items()},
                np.load(output / "artifacts/prediction.npy", allow_pickle=False),
                metrics,
            )
        )

    first, changed = runs
    for key, tensor in first[0].items():
        torch.testing.assert_close(tensor, changed[0][key], rtol=0, atol=0)
    np.testing.assert_array_equal(first[1], changed[1])
    assert first[2]["evaluation_target"] != changed[2]["evaluation_target"]


def test_missing_target_coverage_fails_before_evaluation_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts(tmp_path / "data")
    original_predict = ccnet_pipeline.predict_ccnet5d_volume

    def omit_target(*args: object, **kwargs: object):
        predicted = original_predict(*args, **kwargs)
        observed = args[1]
        counts = predicted.coverage_counts.copy()
        target_position = tuple(np.argwhere(observed.evaluation_target_trace_mask)[-1])
        counts[target_position] = 0
        return replace(predicted, coverage_counts=counts)

    monkeypatch.setattr(ccnet_pipeline, "predict_ccnet5d_volume", omit_target)

    def unexpected_evaluation(*_args: object, **_kwargs: object) -> dict[str, object]:
        pytest.fail("incomplete target coverage must fail before target truth is read")

    monkeypatch.setattr(ccnet_pipeline, "evaluate_c3_volume_prediction", unexpected_evaluation)

    with pytest.raises(ValueError, match="cover every evaluation target trace"):
        _run(tmp_path, artifacts)

    assert not (tmp_path / "run").exists()
