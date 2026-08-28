from __future__ import annotations

import numpy as np
import pytest

from seis_interp.training.point_sampler import (
    RandomTracePatchSampler,
    overlapping_patch_starts,
)


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
    spatial = np.arange(25, dtype=np.float64).reshape(5, 5)
    amplitudes = np.asarray(
        [[row * 100 + sample for sample in range(10)] for row in range(5)],
        dtype=np.float32,
    )
    return time, spatial, amplitudes


def _sampler(*, random_seed: int = 8) -> RandomTracePatchSampler:
    time, spatial, amplitudes = _arrays()
    return RandomTracePatchSampler(
        time,
        spatial,
        amplitudes,
        np.asarray([0, 2, 3, 4], dtype=np.int64),
        patch_size=4,
        patch_starts=np.asarray([0, 3, 6], dtype=np.int64),
        random_seed=random_seed,
    )


def test_overlapping_patch_starts_matches_study_010_contract() -> None:
    assert overlapping_patch_starts(625, 64, 0.5) == (
        *range(0, 545, 32),
        561,
    )


@pytest.mark.parametrize(
    ("sample_count", "patch_size", "overlap_fraction", "expected"),
    [
        (10, 4, 0.5, (0, 2, 4, 6)),
        (11, 4, 0.5, (0, 2, 4, 6, 7)),
        (4, 4, 0.5, (0,)),
        (10, 4, 0.0, (0, 4, 6)),
    ],
)
def test_overlapping_patch_starts_adds_only_the_needed_end_aligned_patch(
    sample_count: int,
    patch_size: int,
    overlap_fraction: float,
    expected: tuple[int, ...],
) -> None:
    assert overlapping_patch_starts(sample_count, patch_size, overlap_fraction) == expected


@pytest.mark.parametrize(
    ("sample_count", "patch_size", "message"),
    [
        (0, 4, "sample_count must be a positive integer"),
        (10, 0, "patch_size must be a positive integer"),
        (10, 11, "patch_size must not exceed"),
        (True, 1, "sample_count must be a positive integer"),
    ],
)
def test_overlapping_patch_starts_rejects_invalid_sizes(
    sample_count: int,
    patch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        overlapping_patch_starts(sample_count, patch_size, 0.5)


@pytest.mark.parametrize(
    ("overlap_fraction", "message"),
    [
        (-0.5, "within"),
        (1.0, "within"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (True, "finite"),
        ("0.5", "finite"),
        (0.3, "positive integer"),
    ],
)
def test_overlapping_patch_starts_rejects_invalid_overlap(
    overlap_fraction: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        overlapping_patch_starts(10, 4, overlap_fraction)  # type: ignore[arg-type]


def test_trace_patch_batch_has_distinct_rows_and_one_shared_contiguous_window() -> None:
    time, spatial, amplitudes = _arrays()
    sampler = _sampler()

    coordinates, targets = sampler.sample(3)

    coordinate_patches = coordinates.reshape(3, 4, -1)
    target_patches = targets.reshape(3, 4)
    sampled_rows = (target_patches[:, 0].astype(np.int64) // 100).astype(np.int64)
    sampled_time_indices = (target_patches.astype(np.int64) % 100).astype(np.int64)
    shared_time_indices = sampled_time_indices[0]

    assert coordinates.shape == (12, 6)
    assert targets.shape == (12,)
    assert coordinates.dtype == np.float64
    assert targets.dtype == np.float32
    assert len(set(sampled_rows)) == 3
    assert set(sampled_rows) <= {0, 2, 3, 4}
    assert int(shared_time_indices[0]) in {0, 3, 6}
    np.testing.assert_array_equal(
        shared_time_indices,
        np.arange(shared_time_indices[0], shared_time_indices[0] + 4),
    )
    for patch_index, array_row in enumerate(sampled_rows):
        np.testing.assert_array_equal(sampled_time_indices[patch_index], shared_time_indices)
        np.testing.assert_array_equal(
            target_patches[patch_index],
            amplitudes[array_row, shared_time_indices],
        )
        np.testing.assert_array_equal(
            coordinate_patches[patch_index, :, 0],
            time[shared_time_indices],
        )
        np.testing.assert_array_equal(
            coordinate_patches[patch_index, :, 1:],
            np.repeat(spatial[array_row][None, :], 4, axis=0),
        )


def test_trace_patch_sampling_is_reproducible_without_changing_global_rng() -> None:
    np.random.seed(17)
    expected_global = np.random.random(3)
    np.random.seed(17)
    first = _sampler(random_seed=5)
    second = _sampler(random_seed=5)

    first_batches = (first.sample(2), first.sample(2), first.sample(2))
    second_batches = (second.sample(2), second.sample(2), second.sample(2))

    for first_batch, second_batch in zip(first_batches, second_batches, strict=True):
        np.testing.assert_array_equal(first_batch[0], second_batch[0])
        np.testing.assert_array_equal(first_batch[1], second_batch[1])
    np.testing.assert_array_equal(np.random.random(3), expected_global)


@pytest.mark.parametrize("traces_per_update", [0, -1, True])
def test_trace_patch_sampler_rejects_invalid_trace_count(traces_per_update: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _sampler().sample(traces_per_update)


def test_trace_patch_sampler_rejects_more_than_available_training_rows() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _sampler().sample(5)


@pytest.mark.parametrize("patch_size", [0, -1, True])
def test_trace_patch_sampler_rejects_invalid_patch_size(patch_size: int) -> None:
    time, spatial, amplitudes = _arrays()

    with pytest.raises(ValueError, match="positive integer"):
        RandomTracePatchSampler(
            time,
            spatial,
            amplitudes,
            np.asarray([0, 2], dtype=np.int64),
            patch_size=patch_size,
            patch_starts=np.asarray([0], dtype=np.int64),
            random_seed=1,
        )


def test_trace_patch_sampler_rejects_patch_larger_than_time_axis() -> None:
    time, spatial, amplitudes = _arrays()

    with pytest.raises(ValueError, match="must not exceed"):
        RandomTracePatchSampler(
            time,
            spatial,
            amplitudes,
            np.asarray([0, 2], dtype=np.int64),
            patch_size=11,
            patch_starts=np.asarray([0], dtype=np.int64),
            random_seed=1,
        )


@pytest.mark.parametrize(
    ("patch_starts", "message"),
    [
        (np.asarray([], dtype=np.int64), "non-empty one-dimensional"),
        (np.asarray([[0, 3]], dtype=np.int64), "non-empty one-dimensional"),
        (np.asarray([0.0, 3.0]), "integer dtype"),
        (np.asarray([False, True]), "integer dtype"),
        (np.asarray([0, 0], dtype=np.int64), "unique"),
        (np.asarray([-1, 3], dtype=np.int64), "within"),
        (np.asarray([0, 7], dtype=np.int64), "within"),
    ],
)
def test_trace_patch_sampler_rejects_invalid_patch_starts(
    patch_starts: np.ndarray,
    message: str,
) -> None:
    time, spatial, amplitudes = _arrays()

    with pytest.raises(ValueError, match=message):
        RandomTracePatchSampler(
            time,
            spatial,
            amplitudes,
            np.asarray([0, 2], dtype=np.int64),
            patch_size=4,
            patch_starts=patch_starts,
            random_seed=1,
        )


def test_trace_patch_sampler_rejects_duplicate_training_rows() -> None:
    time, spatial, amplitudes = _arrays()

    with pytest.raises(ValueError, match="unique"):
        RandomTracePatchSampler(
            time,
            spatial,
            amplitudes,
            np.asarray([0, 0, 2], dtype=np.int64),
            patch_size=4,
            patch_starts=np.asarray([0, 3], dtype=np.int64),
            random_seed=1,
        )


@pytest.mark.parametrize("random_seed", [-1, True, 1.5])
def test_trace_patch_sampler_rejects_invalid_seed(random_seed: object) -> None:
    time, spatial, amplitudes = _arrays()

    with pytest.raises(ValueError, match="integer|non-negative"):
        RandomTracePatchSampler(
            time,
            spatial,
            amplitudes,
            np.asarray([0, 2], dtype=np.int64),
            patch_size=4,
            patch_starts=np.asarray([0, 3], dtype=np.int64),
            random_seed=random_seed,  # type: ignore[arg-type]
        )
