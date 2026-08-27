from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.training.correlation_loss import trace_correlation_loss
from seis_interp.training.point_sampler import build_trace_points


def _training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = 625
    time = np.linspace(-1.0, 1.0, sample_count, dtype=np.float64)
    spatial = np.arange(50, dtype=np.float64).reshape(10, 5) / 50.0
    phase = np.linspace(0.0, 4.0 * np.pi, sample_count, dtype=np.float64)
    amplitudes = np.asarray(
        [np.sin(phase + row / 10.0) + row / 20.0 for row in range(10)],
        dtype=np.float32,
    )
    rows = np.arange(10, dtype=np.int64)
    return time, spatial, amplitudes, rows


def _model_config() -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "model": {
            "input_features": 6,
            "hidden_width": 8,
            "hidden_layers": 1,
            "omega_0": 10.0,
            "hidden_omega": 1.0,
        },
        "training": {"learning_rate": 1.0e-3},
    }


def test_trace_batch_correlation_uses_eight_complete_traces_and_combines_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows = _training_arrays()
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)
    actual_sampler = pipeline.RandomTraceBatchSampler
    sampled_trace_counts: list[int] = []
    correlation_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    correlation_epsilons: list[float] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._sampler = actual_sampler(*args, **kwargs)

        def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
            sampled_trace_counts.append(traces_per_update)
            return self._sampler.sample(traces_per_update)

    def recording_correlation_loss(
        prediction: Any,
        target: Any,
        *,
        eps: float,
    ) -> Any:
        correlation_shapes.append((tuple(prediction.shape), tuple(target.shape)))
        correlation_epsilons.append(eps)
        return trace_correlation_loss(prediction, target, eps=eps)

    monkeypatch.setattr(pipeline, "RandomTraceBatchSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "trace_correlation_loss", recording_correlation_loss)

    metrics = pipeline.run_training_fit_condition(
        config=_model_config(),
        label="tracebatch8_corr0p1_trace10",
        batch_mode="random_complete_traces",
        full_batch=False,
        replacement=False,
        total_updates=1,
        report_interval=1,
        batch_size=5000,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        normalized_amplitudes=amplitudes,
        selected_array_rows=rows,
        all_coordinate_tensor=None,
        all_target_tensor=None,
        training_coordinates=training_coordinates,
        training_targets=training_targets,
        sample_count=625,
        prediction_batch_size=4096,
        device="cpu",
        random_seed=42,
        traces_per_update=8,
        correlation_weight=0.1,
        correlation_eps=1.0e-4,
    )

    assert sampled_trace_counts == [8]
    assert correlation_shapes == [((8, 625), (8, 625))]
    assert correlation_epsilons == [1.0e-4]
    assert metrics["batch_mode"] == "random_complete_traces"
    assert metrics["traces_per_update"] == 8
    assert metrics["batch_size"] == 5000
    assert metrics["correlation_weight"] == 0.1
    assert metrics["correlation_eps"] == 1.0e-4
    assert metrics["loss_semantics"] == "mse_plus_trace_correlation"
    history_row = metrics["history"][0]
    assert history_row["mean_train_loss_since_last_report"] == pytest.approx(
        history_row["mean_train_mse_loss_since_last_report"]
        + 0.1 * history_row["mean_train_correlation_loss_since_last_report"]
    )


def test_pure_mse_condition_does_not_add_correlation_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.linspace(-1.0, 1.0, 3, dtype=np.float64)
    spatial = np.arange(10, dtype=np.float64).reshape(2, 5) / 10.0
    amplitudes = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    rows = np.arange(2, dtype=np.int64)
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)

    def unexpected_correlation_loss(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure-MSE training must not compute correlation loss")

    monkeypatch.setattr(pipeline, "trace_correlation_loss", unexpected_correlation_loss)

    metrics = pipeline.run_training_fit_condition(
        config=_model_config(),
        label="tracebatch1_trace2",
        batch_mode="random_complete_traces",
        full_batch=False,
        replacement=False,
        total_updates=1,
        report_interval=1,
        batch_size=3,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        normalized_amplitudes=amplitudes,
        selected_array_rows=rows,
        all_coordinate_tensor=None,
        all_target_tensor=None,
        training_coordinates=training_coordinates,
        training_targets=training_targets,
        sample_count=3,
        prediction_batch_size=3,
        device="cpu",
        random_seed=42,
        traces_per_update=1,
    )

    assert "correlation_weight" not in metrics
    assert "correlation_eps" not in metrics
    assert "loss_semantics" not in metrics
    assert set(metrics["history"][0]) == {
        "step",
        "mean_train_loss_since_last_report",
        "training_median_trace_snr_db",
        "training_global_snr_db",
        "training_median_trace_correlation",
        "training_prediction_target_rms_ratio",
    }
