from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.training.point_sampler import build_trace_points


def _training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    spatial = np.arange(25, dtype=np.float64).reshape(5, 5) / 25.0
    amplitudes = np.asarray(
        [[row * 100 + sample for sample in range(8)] for row in range(5)],
        dtype=np.float32,
    )
    rows = np.asarray([0, 2, 3, 4], dtype=np.int64)
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


def test_temporal_patch_condition_uses_shared_contiguous_patches_and_pure_mse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes, rows = _training_arrays()
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)
    actual_sampler = pipeline.RandomTracePatchSampler
    actual_to_model_tensors = pipeline.to_model_tensors
    sampler_rows: list[np.ndarray] = []
    sampler_patch_contracts: list[tuple[int, list[int]]] = []
    sampled_trace_counts: list[int] = []
    sampled_batches: list[tuple[np.ndarray, np.ndarray]] = []
    tensor_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            sampler_rows.append(np.asarray(args[3]).copy())
            sampler_patch_contracts.append(
                (
                    int(kwargs["patch_size"]),
                    [int(value) for value in np.asarray(kwargs["patch_starts"])],
                )
            )
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

    def unexpected_correlation_loss(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure-MSE temporal-patch training must not use correlation loss")

    monkeypatch.setattr(pipeline, "RandomTracePatchSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "to_model_tensors", recording_to_model_tensors)
    monkeypatch.setattr(pipeline, "trace_correlation_loss", unexpected_correlation_loss)

    metrics = pipeline.run_training_fit_condition(
        config=_model_config(),
        label="patch4_trace3_trace4",
        batch_mode="random_shared_temporal_patch",
        full_batch=False,
        replacement=False,
        total_updates=2,
        report_interval=1,
        batch_size=12,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        normalized_amplitudes=amplitudes,
        selected_array_rows=rows,
        all_coordinate_tensor=None,
        all_target_tensor=None,
        training_coordinates=training_coordinates,
        training_targets=training_targets,
        sample_count=8,
        prediction_batch_size=7,
        device="cpu",
        random_seed=42,
        traces_per_update=3,
        samples_per_trace=4,
        patch_starts=(0, 2, 4),
        temporal_patch_overlap_fraction=0.5,
    )

    assert len(sampler_rows) == 1
    np.testing.assert_array_equal(sampler_rows[0], rows)
    assert sampler_patch_contracts == [(4, [0, 2, 4])]
    assert sampled_trace_counts == [3, 3]
    assert tensor_shapes == [((12, 6), (12, 1)), ((12, 6), (12, 1))]
    for coordinates, targets in sampled_batches:
        coordinate_patches = coordinates.reshape(3, 4, 6)
        target_patches = targets.reshape(3, 4)
        sampled_rows = (target_patches[:, 0].astype(np.int64) // 100).astype(np.int64)
        sampled_time_indices = (target_patches.astype(np.int64) % 100).astype(np.int64)
        shared_time_indices = sampled_time_indices[0]

        assert len(set(sampled_rows)) == 3
        assert set(sampled_rows) <= set(rows)
        assert int(shared_time_indices[0]) in {0, 2, 4}
        np.testing.assert_array_equal(
            shared_time_indices,
            np.arange(shared_time_indices[0], shared_time_indices[0] + 4),
        )
        for patch_index, array_row in enumerate(sampled_rows):
            np.testing.assert_array_equal(sampled_time_indices[patch_index], shared_time_indices)
            np.testing.assert_array_equal(
                coordinate_patches[patch_index, :, 0],
                time[shared_time_indices],
            )
            np.testing.assert_array_equal(
                coordinate_patches[patch_index, :, 1:],
                np.repeat(spatial[array_row][None, :], 4, axis=0),
            )
            np.testing.assert_array_equal(
                target_patches[patch_index],
                amplitudes[array_row, shared_time_indices],
            )

    assert metrics["batch_mode"] == "random_shared_temporal_patch"
    assert metrics["full_batch"] is False
    assert metrics["replacement"] is False
    assert metrics["traces_per_update"] == 3
    assert metrics["samples_per_trace"] == 4
    assert metrics["temporal_patch_overlap_fraction"] == 0.5
    assert metrics["patch_starts"] == [0, 2, 4]
    assert metrics["shared_temporal_patch"] is True
    assert metrics["batch_size"] == 12
    assert metrics["point_evaluations"] == 24
    assert [row["step"] for row in metrics["history"]] == [1, 2]
    assert all(
        set(row)
        == {
            "step",
            "mean_train_loss_since_last_report",
            "training_median_trace_snr_db",
            "training_global_snr_db",
            "training_median_trace_correlation",
            "training_prediction_target_rms_ratio",
        }
        for row in metrics["history"]
    )
    assert all(math.isfinite(float(value)) for row in metrics["history"] for value in row.values())
    assert {"correlation_weight", "correlation_eps", "loss_semantics"}.isdisjoint(metrics)


def test_real_temporal_patch_geometry_produces_4992_point_tensor_shapes() -> None:
    trace_count = 80
    sample_count = 625
    time = np.linspace(-1.0, 1.0, sample_count, dtype=np.float64)
    spatial = np.arange(trace_count * 5, dtype=np.float64).reshape(trace_count, 5)
    amplitudes = np.arange(trace_count * sample_count, dtype=np.float32).reshape(
        trace_count,
        sample_count,
    )
    patch_starts = pipeline.overlapping_patch_starts(sample_count, 64, 0.5)
    sampler = pipeline.RandomTracePatchSampler(
        time,
        spatial,
        amplitudes,
        np.arange(trace_count, dtype=np.int64),
        patch_size=64,
        patch_starts=np.asarray(patch_starts, dtype=np.int64),
        random_seed=42,
    )

    coordinates, targets = sampler.sample(78)
    coordinate_tensor, target_tensor = pipeline.to_model_tensors(
        coordinates,
        targets,
        device="cpu",
    )

    assert coordinates.shape == (4992, 6)
    assert targets.shape == (4992,)
    assert tuple(coordinate_tensor.shape) == (4992, 6)
    assert tuple(target_tensor.shape) == (4992, 1)
    coordinate_patches = coordinates.reshape(78, 64, 6)
    np.testing.assert_array_equal(
        coordinate_patches[:, :, 0],
        np.tile(coordinate_patches[0, :, 0], (78, 1)),
    )
    assert np.unique(coordinate_patches[:, 0, 1:], axis=0).shape[0] == 78


def test_temporal_patch_condition_rejects_inconsistent_batch_size() -> None:
    time, spatial, amplitudes, rows = _training_arrays()
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)

    with pytest.raises(
        ValueError,
        match=r"batch_size must equal traces_per_update \* samples_per_trace \(12\)",
    ):
        pipeline.run_training_fit_condition(
            config=_model_config(),
            label="patch4_trace3_trace4",
            batch_mode="random_shared_temporal_patch",
            full_batch=False,
            replacement=False,
            total_updates=1,
            report_interval=1,
            batch_size=11,
            normalized_time=time,
            normalized_spatial_by_array_row=spatial,
            normalized_amplitudes=amplitudes,
            selected_array_rows=rows,
            all_coordinate_tensor=None,
            all_target_tensor=None,
            training_coordinates=training_coordinates,
            training_targets=training_targets,
            sample_count=8,
            prediction_batch_size=7,
            device="cpu",
            random_seed=42,
            traces_per_update=3,
            samples_per_trace=4,
            patch_starts=(0, 2, 4),
            temporal_patch_overlap_fraction=0.5,
        )
