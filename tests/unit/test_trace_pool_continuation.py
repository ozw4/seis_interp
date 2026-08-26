from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.training.point_sampler import build_trace_points


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    spatial = np.arange(20, dtype=np.float64).reshape(4, 5) / 20.0
    amplitudes = np.asarray(
        [[np.sin(sample + row / 10.0) for sample in range(5)] for row in range(4)],
        dtype=np.float32,
    )
    return time, spatial, amplitudes


def _model_config() -> dict[str, object]:
    return {
        "project": {"random_seed": 5},
        "model": {
            "input_features": 6,
            "hidden_width": 8,
            "hidden_layers": 1,
            "omega_0": 10.0,
        },
        "training": {"learning_rate": 1.0e-3},
    }


def _optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
    assert steps
    assert len(set(steps)) == 1
    return steps[0]


def test_training_condition_carries_supplied_model_and_exact_adam_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes = _arrays()
    config = _model_config()
    torch.manual_seed(5)
    model = pipeline._build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}

    def unexpected_model_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a supplied continuation model must not be rebuilt")

    def unexpected_optimizer_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a supplied continuation optimizer must not be rebuilt")

    monkeypatch.setattr(pipeline, "_build_model", unexpected_model_build)
    monkeypatch.setattr(pipeline.torch.optim, "Adam", unexpected_optimizer_build)

    stage_metrics: list[dict[str, object]] = []
    for stage_index, rows in enumerate(
        (np.asarray([0], dtype=np.int64), np.asarray([0, 1], dtype=np.int64))
    ):
        training_coordinates, training_targets = build_trace_points(
            time,
            spatial,
            amplitudes,
            rows,
        )
        stage_metrics.append(
            pipeline.run_training_fit_condition(
                config=config,
                label=f"stage{stage_index}",
                batch_mode="random_replacement",
                full_batch=False,
                replacement=True,
                total_updates=1,
                report_interval=1,
                batch_size=4,
                normalized_time=time,
                normalized_spatial_by_array_row=spatial,
                normalized_amplitudes=amplitudes,
                selected_array_rows=rows,
                all_coordinate_tensor=None,
                all_target_tensor=None,
                training_coordinates=training_coordinates,
                training_targets=training_targets,
                sample_count=5,
                prediction_batch_size=4,
                device="cpu",
                random_seed=5 + stage_index,
                model=model,
                optimizer=optimizer,
            )
        )

        assert _optimizer_step(optimizer) == stage_index + 1

    assert any(
        not torch.equal(initial_parameters[name], value) for name, value in model.named_parameters()
    )
    assert [metrics["updates_completed"] for metrics in stage_metrics] == [1, 1]
    assert [metrics["trace_count"] for metrics in stage_metrics] == [1, 2]
    assert all(
        {"correlation_weight", "correlation_eps", "loss_semantics"}.isdisjoint(metrics)
        for metrics in stage_metrics
    )
