from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.c3_poc_inputs import C3_RANDOM80_POC_BENCHMARK_ID
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.pipelines import interpolate_drr as drr_pipeline
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS
from seis_interp.processing.drr_windows import WindowedDrrResult
from seis_interp.training import amplitude_scaling


def _config() -> dict[str, object]:
    return {
        "drr": {
            "rank": 1,
            "damping_power": 3,
            "n_iterations": 2,
            "frequency_min_hz": 0.0,
            "frequency_max_hz": None,
            "spatial_window_shape": None,
            "spatial_overlap": None,
        },
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }


def _inputs() -> C3VolumeRunInputs:
    shape = (3, 1, 1, 1, 5)
    observed_mask = np.zeros(shape[1:], dtype=np.bool_)
    observed_mask[..., 0] = True
    target_mask = ~observed_mask
    values = np.zeros(shape, dtype=np.float32)
    values[:, observed_mask] = np.array([[2.0], [-4.0], [8.0]], dtype=np.float32)
    observed = ObservedC3Volume(
        values=values,
        time_s=np.arange(shape[0], dtype=np.float64) * 0.008,
        array_rows=np.arange(5, dtype=np.int64).reshape(shape[1:]),
        observed_trace_mask=observed_mask,
        evaluation_target_trace_mask=target_mask,
    )
    selection = {
        "time": [0, 3],
        "source_line": [25, 26],
        "shot_in_line": [0, 1],
        "relative_receiver_x": [0, 1],
        "relative_receiver_y": [0, 5],
    }
    return C3VolumeRunInputs(
        observed_volume=observed,
        index_table=pd.DataFrame({"array_row": np.arange(5, dtype=np.int64)}),
        case={
            "case_id": "poc_case",
            "dataset_id": "seg_c3_na",
            "partition": "test",
            "mask": {
                "kind": "random_trace",
                "missing_fraction": 0.8,
                "random_seed": 42,
            },
        },
        volume_metadata={
            "volume_id": "poc_volume",
            "selection": selection,
            "shape": list(shape),
            "trace_count": 5,
            "role_counts": {"observed": 1, "evaluation_target": 4},
        },
        inputs_lock={
            "benchmark_id": C3_RANDOM80_POC_BENCHMARK_ID,
            "case_id": "poc_case",
            "volume_id": "poc_volume",
        },
    )


def _arguments(tmp_path: Path, output_name: str = "run") -> dict[str, object]:
    return {
        "config_path": tmp_path / "config.yaml",
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "mask_dir": tmp_path / "mask",
        "case_dir": tmp_path / "case",
        "volume_dir": tmp_path / "volume",
        "output_dir": tmp_path / output_name,
    }


def _patch_run_inputs(
    monkeypatch: pytest.MonkeyPatch,
    inputs: C3VolumeRunInputs,
) -> None:
    monkeypatch.setattr(drr_pipeline, "load_resolved_config", lambda path: _config())

    def load_inputs(**kwargs: object) -> C3VolumeRunInputs:
        assert kwargs["dimensions"] is MAIN_C3_DIMENSIONS
        assert kwargs["config"] == _config()
        return inputs

    monkeypatch.setattr(drr_pipeline, "load_c3_random80_poc_inputs", load_inputs)
    monkeypatch.setattr(
        drr_pipeline.run_records,
        "current_git_metadata",
        lambda: {"git_commit": "a" * 40, "git_worktree_dirty": False},
    )


def _evaluation_summary() -> dict[str, object]:
    return {
        "evaluation_domain": "evaluation_target",
        "amplitude_domain": "physical",
        "evaluation_target": {
            "trace_count": 4,
            "sample_count": 12,
            "reference_energy": 24.0,
            "error_energy": 6.0,
            "snr_db": 6.020599913279624,
            "snr_status": "finite",
            "rmse": 0.5,
            "relative_l2": 0.5,
            "mean_trace_relative_mse": 0.25,
            "covered_target_trace_count": 4,
            "target_trace_count": 4,
        },
        "zero_fill": {"snr_db": 0.0, "snr_status": "finite", "rmse": 1.0},
        "observed_max_abs_error": 0.0,
    }


def test_drr_poc_run_passes_only_physical_observations_and_records_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    observed = inputs.observed_volume
    _patch_run_inputs(monkeypatch, inputs)

    def unexpected_global_rms(*args: object, **kwargs: object) -> float:
        raise AssertionError("DRR must not compute benchmark Global RMS")

    monkeypatch.setattr(amplitude_scaling, "compute_observed_global_rms", unexpected_global_rms)

    def interpolate(
        values: np.ndarray,
        observed_mask: np.ndarray,
        time_s: np.ndarray,
        **parameters: object,
    ) -> WindowedDrrResult:
        assert values is observed.values
        np.testing.assert_array_equal(values[:, observed_mask], [[2.0], [-4.0], [8.0]])
        np.testing.assert_array_equal(values[:, ~observed_mask], 0.0)
        assert time_s is observed.time_s
        assert parameters["rank"] == 1
        assert parameters["damping_power"] == 3
        assert parameters["n_iterations"] == 2
        prediction = values.copy()
        prediction[:, ~observed_mask] = 1.5
        return WindowedDrrResult(prediction, 1, 0, 0)

    monkeypatch.setattr(drr_pipeline, "interpolate_drr_volume", interpolate)

    def evaluate(
        values: np.ndarray,
        observed_volume: ObservedC3Volume,
        **kwargs: object,
    ) -> dict[str, object]:
        assert observed_volume is observed
        np.testing.assert_array_equal(
            values[:, observed.observed_trace_mask],
            observed.values[:, observed.observed_trace_mask],
        )
        coverage = kwargs["target_coverage_mask"]
        assert isinstance(coverage, np.ndarray)
        assert coverage.dtype == np.bool_
        assert np.all(coverage)
        return _evaluation_summary()

    monkeypatch.setattr(drr_pipeline, "evaluate_c3_volume_prediction", evaluate)

    metrics = drr_pipeline.interpolate_drr_run(**_arguments(tmp_path))

    target_metrics = metrics["evaluation_target"]
    assert isinstance(target_metrics, dict)
    assert set(("snr_db", "rmse", "relative_l2", "mean_trace_relative_mse")) <= set(target_metrics)
    prediction = np.load(tmp_path / "run" / drr_pipeline.PREDICTION_RELATIVE_PATH)
    assert np.all(np.isfinite(prediction[:, observed.evaluation_target_trace_mask]))
    np.testing.assert_array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask],
    )

    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["method"] == "drr"
    assert metadata["benchmark_id"] == C3_RANDOM80_POC_BENCHMARK_ID
    assert metadata["input_amplitude_domain"] == "physical"
    assert metadata["output_amplitude_domain"] == "physical"
    assert metadata["method_details"]["benchmark_global_rms_normalization"] is False
    assert metadata["method_details"]["native_rank_reduction"] is True
    assert metadata["method_details"]["native_iterative_updates"] is True
    assert metadata["method_details"]["drr"]["rank"] == 1
    assert metadata["method_details"]["drr"]["damping_power"] == 3
    assert metadata["method_details"]["drr"]["n_iterations"] == 2
    assert metadata["method_details"]["drr"]["amplitude_normalization"] == "none"
    assert metadata["coverage"] == {
        "analysis_trace_count": 5,
        "covered_analysis_trace_count": 5,
        "target_trace_count": 4,
        "covered_target_trace_count": 4,
        "complete": True,
    }
    assert metadata["timing"]["reconstruction_seconds"] >= 0.0
    assert metadata["timing"]["end_to_end_seconds"] >= 0.0
    assert metadata["status"] == "success"


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (
            WindowedDrrResult(np.zeros((2, 1, 1, 1, 5), dtype=np.float32), 1, 0, 0),
            "shape must match",
        ),
        (
            WindowedDrrResult(np.zeros((3, 1, 1, 1, 5), dtype=np.float32), 2, 1, 1),
            "complete analysis volume",
        ),
    ],
)
def test_drr_poc_run_rejects_wrong_shape_or_partial_coverage_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: WindowedDrrResult,
    match: str,
) -> None:
    inputs = _inputs()
    _patch_run_inputs(monkeypatch, inputs)
    monkeypatch.setattr(drr_pipeline, "interpolate_drr_volume", lambda *args, **kwargs: result)

    def unexpected_evaluation(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("invalid reconstruction must fail before evaluation")

    monkeypatch.setattr(drr_pipeline, "evaluate_c3_volume_prediction", unexpected_evaluation)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=match):
        drr_pipeline.interpolate_drr_run(**_arguments(tmp_path))

    assert not output.exists()
