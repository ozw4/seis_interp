import math

import numpy as np
import pytest
import torch

from seis_interp.training.nersi_profile_regularization import (
    ProfileCoordinateJitter,
    normalized_nuclear_norm,
    validate_nuclear_norm_weight,
)


def test_nuclear_norm_is_the_mean_singular_value_sum_over_the_profile_scale():
    rng = np.random.default_rng(3)
    profiles = torch.from_numpy(rng.normal(size=(3, 5, 7)))
    expected = np.mean([np.linalg.svd(p, compute_uv=False).sum() for p in profiles.numpy()])
    assert float(normalized_nuclear_norm(profiles)) == pytest.approx(expected / math.sqrt(35))


def test_unit_rms_profiles_score_between_rank_one_and_full_rank():
    rank_one = torch.ones(1, 4, 9)
    orthogonal = torch.zeros(1, 4, 9)
    orthogonal[0, torch.arange(4), torch.arange(4)] = 3.0  # four unit-RMS rows of length 9
    assert float(normalized_nuclear_norm(rank_one)) == pytest.approx(1.0)
    assert float(normalized_nuclear_norm(orthogonal)) == pytest.approx(2.0)


def test_nuclear_norm_gradient_is_finite_at_zero_and_repeated_singular_values():
    for profiles in (torch.zeros(2, 4, 6), torch.ones(2, 4, 6)):
        profiles = profiles.clone().requires_grad_(True)
        normalized_nuclear_norm(profiles).backward()
        assert torch.isfinite(profiles.grad).all()


@pytest.mark.parametrize("profiles", [torch.zeros(4, 5), torch.zeros(2, 3, 4, dtype=torch.int64)])
def test_nuclear_norm_needs_floating_profile_stacks(profiles):
    with pytest.raises(ValueError, match="floating"):
        normalized_nuclear_norm(profiles)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_nuclear_norm_weights_are_rejected(value):
    with pytest.raises(ValueError, match="nuclear_norm_weight"):
        validate_nuclear_norm_weight(value)


def test_jitter_stays_within_the_cell_fraction_of_each_axis():
    jitter = ProfileCoordinateJitter(0.5, (6, 32, 1), seed=7)
    coordinates = torch.full((4000, 3), 0.5)
    offsets = (jitter(coordinates) - coordinates).numpy().astype(np.float64)
    radius = np.array([0.5 / 5, 0.5 / 31, 0.5])
    assert np.all(np.abs(offsets) <= radius + 1e-7)
    assert np.all(np.abs(offsets).max(axis=0) > 0.9 * radius)


def test_jitter_is_reproducible_and_independent_of_the_seed_stream():
    first = ProfileCoordinateJitter(0.25, (6, 32, 8), seed=7)
    second = ProfileCoordinateJitter(0.25, (6, 32, 8), seed=7)
    coordinates = torch.zeros(5, 3)
    torch.testing.assert_close(first(coordinates), second(coordinates))
    sampler = np.random.default_rng(7)
    assert not np.allclose(first.rng.uniform(-1, 1, size=3), sampler.uniform(-1, 1, size=3)), (
        "the jitter stream must not replay the trace sampler's stream"
    )


@pytest.mark.parametrize("cells", [0, -0.1, 0.6, float("nan"), True])
def test_invalid_jitter_is_rejected(cells):
    with pytest.raises(ValueError, match="coordinate_jitter_cells"):
        ProfileCoordinateJitter(cells, (6, 32, 8), seed=7)
