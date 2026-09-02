from __future__ import annotations

import numpy as np
import pytest

from seis_interp.training.point_sampler import (
    RandomPointSampler,
    RandomTraceBatchSampler,
    build_trace_points,
)


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
    spatial = np.arange(15, dtype=np.float64).reshape(3, 5)
    amplitudes = np.asarray(
        [[row * 1000 + time_index for time_index in range(4)] for row in range(3)],
        dtype=np.float32,
    )
    return time, spatial, amplitudes


def test_same_seed_produces_same_first_batch_without_changing_global_rng() -> None:
    time, spatial, amplitudes = _arrays()
    np.random.seed(17)
    expected_global = np.random.random(3)
    np.random.seed(17)

    first = RandomPointSampler(time, spatial, amplitudes, np.array([0, 2]), random_seed=4)
    second = RandomPointSampler(time, spatial, amplitudes, np.array([0, 2]), random_seed=4)
    first_coordinates, first_targets = first.sample(32)
    second_coordinates, second_targets = second.sample(32)

    np.testing.assert_array_equal(first_coordinates, second_coordinates)
    np.testing.assert_array_equal(first_targets, second_targets)
    np.testing.assert_array_equal(np.random.random(3), expected_global)
    assert first_coordinates.shape == (32, 6)
    assert first_coordinates.dtype == np.float64
    assert first_targets.dtype == np.float32
    assert first.amplitude_scaling == "train_global_rms"


def test_random_point_sampler_records_per_trace_target_scaling() -> None:
    time, spatial, amplitudes = _arrays()

    sampler = RandomPointSampler(
        time,
        spatial,
        amplitudes,
        np.array([0, 2]),
        random_seed=4,
        amplitude_scaling="per_trace_rms",
    )

    assert sampler.amplitude_scaling == "per_trace_rms"


def test_only_training_rows_are_sampled_and_coordinates_match_targets() -> None:
    time, spatial, amplitudes = _arrays()
    sampler = RandomPointSampler(time, spatial, amplitudes, np.array([0, 2]), random_seed=8)

    coordinates, targets = sampler.sample(200)

    assert not np.array_equal(coordinates[:100], coordinates[100:])
    rows = (targets.astype(np.int64) // 1000).astype(np.int64)
    time_indices = (targets.astype(np.int64) % 1000).astype(np.int64)
    assert set(rows) <= {0, 2}
    np.testing.assert_array_equal(coordinates[:, 0], time[time_indices])
    np.testing.assert_array_equal(coordinates[:, 1:], spatial[rows])


def test_random_trace_batches_are_seeded_without_changing_global_rng() -> None:
    time, spatial, amplitudes = _arrays()
    np.random.seed(17)
    expected_global = np.random.random(3)
    np.random.seed(17)

    first = RandomTraceBatchSampler(
        time,
        spatial,
        amplitudes,
        np.array([0, 1, 2]),
        random_seed=0,
    )
    second = RandomTraceBatchSampler(
        time,
        spatial,
        amplitudes,
        np.array([0, 1, 2]),
        random_seed=0,
    )
    first_batches = (first.sample(2), first.sample(2))
    second_batches = (second.sample(2), second.sample(2))

    for first_batch, second_batch in zip(first_batches, second_batches, strict=True):
        np.testing.assert_array_equal(first_batch[0], second_batch[0])
        np.testing.assert_array_equal(first_batch[1], second_batch[1])
    np.testing.assert_array_equal(np.random.random(3), expected_global)


def test_random_trace_batch_contains_each_sample_from_distinct_training_rows() -> None:
    time, spatial, amplitudes = _arrays()
    sampler = RandomTraceBatchSampler(
        time,
        spatial,
        amplitudes,
        np.array([0, 2]),
        random_seed=8,
    )

    coordinates, targets = sampler.sample(2)

    target_traces = targets.reshape(2, len(time))
    coordinate_traces = coordinates.reshape(2, len(time), -1)
    sampled_rows = (target_traces[:, 0].astype(np.int64) // 1000).astype(np.int64)
    assert len(set(sampled_rows)) == 2
    assert set(sampled_rows) == {0, 2}
    for trace_index, array_row in enumerate(sampled_rows):
        np.testing.assert_array_equal(target_traces[trace_index], amplitudes[array_row])
        np.testing.assert_array_equal(coordinate_traces[trace_index, :, 0], time)
        np.testing.assert_array_equal(
            coordinate_traces[trace_index, :, 1:],
            np.repeat(spatial[array_row][None, :], len(time), axis=0),
        )
    assert coordinates.shape == (8, 6)
    assert targets.shape == (8,)


def test_build_trace_points_uses_trace_major_time_minor_order_without_mutation() -> None:
    time, spatial, amplitudes = _arrays()
    copies = (time.copy(), spatial.copy(), amplitudes.copy())

    coordinates, targets = build_trace_points(time, spatial, amplitudes, np.array([2, 0]))

    np.testing.assert_array_equal(targets, np.concatenate((amplitudes[2], amplitudes[0])))
    np.testing.assert_array_equal(coordinates[:, 0], np.tile(time, 2))
    np.testing.assert_array_equal(coordinates[:, 1:], np.repeat(spatial[[2, 0]], 4, axis=0))
    assert coordinates.shape == (8, 6)
    for actual, expected in zip((time, spatial, amplitudes), copies, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_point_sampling_and_trace_expansion_follow_spatial_feature_width() -> None:
    time, spatial, amplitudes = _arrays()
    four_spatial_features = spatial[:, :4].copy()
    sampler = RandomPointSampler(
        time,
        four_spatial_features,
        amplitudes,
        np.array([0, 2]),
        random_seed=8,
    )

    sampled_coordinates, _ = sampler.sample(7)
    trace_coordinates, _ = build_trace_points(
        time,
        four_spatial_features,
        amplitudes,
        np.array([2, 0]),
    )

    assert sampled_coordinates.shape == (7, 5)
    assert trace_coordinates.shape == (8, 5)


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_rejects_invalid_batch_size(batch_size: int) -> None:
    time, spatial, amplitudes = _arrays()
    sampler = RandomPointSampler(time, spatial, amplitudes, np.array([0]), random_seed=1)

    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample(batch_size)


@pytest.mark.parametrize("traces_per_update", [0, -1, True])
def test_random_trace_batch_rejects_invalid_trace_count(traces_per_update: int) -> None:
    time, spatial, amplitudes = _arrays()
    sampler = RandomTraceBatchSampler(
        time,
        spatial,
        amplitudes,
        np.array([0, 2]),
        random_seed=1,
    )

    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample(traces_per_update)


def test_random_trace_batch_rejects_more_than_available_training_rows() -> None:
    time, spatial, amplitudes = _arrays()
    sampler = RandomTraceBatchSampler(
        time,
        spatial,
        amplitudes,
        np.array([0, 2]),
        random_seed=1,
    )

    with pytest.raises(ValueError, match="must not exceed"):
        sampler.sample(3)


def test_random_trace_batch_rejects_duplicate_training_rows() -> None:
    time, spatial, amplitudes = _arrays()

    with pytest.raises(ValueError, match="unique"):
        RandomTraceBatchSampler(
            time,
            spatial,
            amplitudes,
            np.array([0, 0, 2]),
            random_seed=1,
        )
