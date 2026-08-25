from __future__ import annotations

import numpy as np
import pytest

from seis_interp.training.point_sampler import RandomPointSampler, build_trace_points


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


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_rejects_invalid_batch_size(batch_size: int) -> None:
    time, spatial, amplitudes = _arrays()
    sampler = RandomPointSampler(time, spatial, amplitudes, np.array([0]), random_seed=1)

    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample(batch_size)
