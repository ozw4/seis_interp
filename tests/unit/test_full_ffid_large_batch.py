from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.training.point_sampler import build_trace_points


def _training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    spatial = np.arange(20, dtype=np.float64).reshape(4, 5) / 20.0
    amplitudes = np.asarray(
        [[1.0 + row + sample / 10.0 for sample in range(5)] for row in range(4)],
        dtype=np.float32,
    )
    rows = np.asarray([0, 2], dtype=np.int64)
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


def test_random_replacement_condition_uses_sampler_and_5000_point_tensor_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows = _training_arrays()
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)
    actual_sampler = pipeline.RandomPointSampler
    actual_to_model_tensors = pipeline.to_model_tensors
    sampler_rows: list[np.ndarray] = []
    sampled_batch_sizes: list[int] = []
    tensor_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            sampler_rows.append(np.asarray(args[3]).copy())
            self._sampler = actual_sampler(*args, **kwargs)

        def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
            sampled_batch_sizes.append(batch_size)
            return self._sampler.sample(batch_size)

    def recording_to_model_tensors(*args: Any, **kwargs: Any) -> Any:
        tensors = actual_to_model_tensors(*args, **kwargs)
        tensor_shapes.append((tuple(tensors[0].shape), tuple(tensors[1].shape)))
        return tensors

    monkeypatch.setattr(pipeline, "RandomPointSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "to_model_tensors", recording_to_model_tensors)

    metrics = pipeline.run_training_fit_condition(
        config=_model_config(),
        label="random5000_trace2",
        batch_mode="random_replacement",
        full_batch=False,
        replacement=True,
        total_updates=2,
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
        sample_count=5,
        prediction_batch_size=4,
        device="cpu",
        random_seed=42,
    )

    assert len(sampler_rows) == 1
    np.testing.assert_array_equal(sampler_rows[0], rows)
    assert sampled_batch_sizes == [5000, 5000]
    assert tensor_shapes == [((5000, 6), (5000, 1)), ((5000, 6), (5000, 1))]
    assert metrics["batch_mode"] == "random_replacement"
    assert metrics["replacement"] is True


@pytest.mark.parametrize(
    ("snr", "rms_ratio", "expected"),
    [
        (20.0, 0.0, "strong_fit"),
        (1.000001, 0.100001, "escaped_zero_predictor"),
        (1.0, 1.0, "near_zero"),
        (19.0, 0.1, "near_zero"),
    ],
)
def test_full_ffid_classification_boundaries(
    snr: float,
    rms_ratio: float,
    expected: str,
) -> None:
    metrics = {
        "best_training_median_trace_snr_db": snr,
        "best_training_prediction_target_rms_ratio": rms_ratio,
    }

    assert pipeline.classify_condition(metrics) == expected


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("strong_fit", "full_ffid_strong_fit"),
        ("escaped_zero_predictor", "full_ffid_escaped_zero_predictor"),
        ("near_zero", "full_ffid_near_zero"),
    ],
)
def test_full_ffid_summary_decision(
    classification: str,
    expected: str,
) -> None:
    assert pipeline.full_ffid_summary_decision(classification) == expected
