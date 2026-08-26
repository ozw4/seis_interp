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
        [[row * 100 + sample for sample in range(5)] for row in range(4)],
        dtype=np.float32,
    )
    rows = np.arange(4, dtype=np.int64)
    return time, spatial, amplitudes, rows


def _model_config() -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "model": {
            "input_features": 6,
            "hidden_width": 8,
            "hidden_layers": 1,
            "omega_0": 10.0,
        },
        "training": {"learning_rate": 1.0e-3},
    }


def test_trace_batch_condition_uses_complete_trace_sampler_each_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows = _training_arrays()
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)
    actual_sampler = pipeline.RandomTraceBatchSampler
    actual_to_model_tensors = pipeline.to_model_tensors
    sampler_rows: list[np.ndarray] = []
    sampled_trace_counts: list[int] = []
    sampled_batches: list[tuple[np.ndarray, np.ndarray]] = []
    tensor_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            sampler_rows.append(np.asarray(args[3]).copy())
            self._sampler = actual_sampler(*args, **kwargs)

        def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
            sampled_trace_counts.append(traces_per_update)
            coordinates, targets = self._sampler.sample(traces_per_update)
            sampled_batches.append((coordinates.copy(), targets.copy()))
            return coordinates, targets

    def recording_to_model_tensors(*args: Any, **kwargs: Any) -> Any:
        tensors = actual_to_model_tensors(*args, **kwargs)
        tensor_shapes.append((tuple(tensors[0].shape), tuple(tensors[1].shape)))
        return tensors

    monkeypatch.setattr(pipeline, "RandomTraceBatchSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "to_model_tensors", recording_to_model_tensors)

    metrics = pipeline.run_training_fit_condition(
        config=_model_config(),
        label="tracebatch2_trace4",
        batch_mode="random_complete_traces",
        full_batch=False,
        replacement=False,
        total_updates=2,
        report_interval=1,
        batch_size=10,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        normalized_amplitudes=amplitudes,
        selected_array_rows=rows,
        all_coordinate_tensor=None,
        all_target_tensor=None,
        training_coordinates=training_coordinates,
        training_targets=training_targets,
        sample_count=5,
        prediction_batch_size=7,
        device="cpu",
        random_seed=42,
        traces_per_update=2,
    )

    assert len(sampler_rows) == 1
    np.testing.assert_array_equal(sampler_rows[0], rows)
    assert sampled_trace_counts == [2, 2]
    assert tensor_shapes == [((10, 6), (10, 1)), ((10, 6), (10, 1))]
    for coordinates, targets in sampled_batches:
        assert coordinates.shape == (10, 6)
        assert targets.shape == (10,)
        target_traces = targets.reshape(2, 5)
        sampled_rows = (target_traces[:, 0] // 100).astype(np.int64)
        assert len(np.unique(sampled_rows)) == 2
        assert set(sampled_rows.tolist()) <= set(rows.tolist())
        for index, array_row in enumerate(sampled_rows):
            np.testing.assert_array_equal(coordinates[index * 5 : (index + 1) * 5, 0], time)
            np.testing.assert_array_equal(
                coordinates[index * 5 : (index + 1) * 5, 1:],
                np.repeat(spatial[array_row][None, :], 5, axis=0),
            )
            np.testing.assert_array_equal(target_traces[index], amplitudes[array_row])

    assert metrics["batch_mode"] == "random_complete_traces"
    assert metrics["full_batch"] is False
    assert metrics["replacement"] is False
    assert metrics["traces_per_update"] == 2
    assert metrics["batch_size"] == 10
    assert metrics["point_evaluations"] == 20


def test_eight_complete_traces_produce_the_fixed_5000_point_tensor_shapes() -> None:
    sample_count = 625
    time = np.linspace(-1.0, 1.0, sample_count, dtype=np.float64)
    spatial = np.arange(50, dtype=np.float64).reshape(10, 5) / 50.0
    amplitudes = np.arange(10 * sample_count, dtype=np.float32).reshape(10, sample_count)
    sampler = pipeline.RandomTraceBatchSampler(
        time,
        spatial,
        amplitudes,
        np.arange(10, dtype=np.int64),
        random_seed=42,
    )

    coordinates, targets = sampler.sample(8)
    coordinate_tensor, target_tensor = pipeline.to_model_tensors(
        coordinates,
        targets,
        device="cpu",
    )

    assert coordinates.shape == (5000, 6)
    assert targets.shape == (5000,)
    assert tuple(coordinate_tensor.shape) == (5000, 6)
    assert tuple(target_tensor.shape) == (5000, 1)
    trace_coordinates = coordinates.reshape(8, sample_count, 6)
    trace_targets = targets.reshape(8, sample_count)
    np.testing.assert_array_equal(trace_coordinates[:, :, 0], np.tile(time, (8, 1)))
    selected_rows: list[int] = []
    for trace_index in range(8):
        spatial_coordinate = trace_coordinates[trace_index, 0, 1:]
        matches = np.flatnonzero(np.all(spatial == spatial_coordinate, axis=1))
        assert len(matches) == 1
        array_row = int(matches[0])
        selected_rows.append(array_row)
        np.testing.assert_array_equal(
            trace_coordinates[trace_index, :, 1:],
            np.repeat(spatial[array_row][None, :], sample_count, axis=0),
        )
        np.testing.assert_array_equal(trace_targets[trace_index], amplitudes[array_row])
    assert len(set(selected_rows)) == 8


def test_trace_batch_condition_rejects_inconsistent_point_batch_size() -> None:
    time, spatial, amplitudes, rows = _training_arrays()
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)

    with pytest.raises(
        ValueError,
        match=r"batch_size must equal traces_per_update \* sample_count \(10\)",
    ):
        pipeline.run_training_fit_condition(
            config=_model_config(),
            label="tracebatch2_trace4",
            batch_mode="random_complete_traces",
            full_batch=False,
            replacement=False,
            total_updates=1,
            report_interval=1,
            batch_size=9,
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            normalized_amplitudes=amplitudes,
            selected_array_rows=rows,
            all_coordinate_tensor=None,
            all_target_tensor=None,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            sample_count=5,
            prediction_batch_size=7,
            device="cpu",
            random_seed=42,
            traces_per_update=2,
        )
