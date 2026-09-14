from dataclasses import replace

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.training.c3_volume_nersi_data import build_c3_volume_nersi_data
from seis_interp.training.nersi_profile_mixup import observed_neighbor_profile_mixup


def _data():
    values = np.arange(8 * 1 * 1 * 2 * 8, dtype=np.float32).reshape(8, 1, 1, 2, 8)
    mask = np.zeros((1, 1, 2, 8), dtype=bool)
    mask[0, 0, 0, :5] = True
    mask[0, 0, 1, 3:] = True
    observed = ObservedC3Volume(
        values=values,
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
        time_s=np.arange(8, dtype=np.float64),
        array_rows=np.arange(16).reshape(mask.shape),
    )
    return build_c3_volume_nersi_data(observed, amplitude_scale=1.0)


def test_mixup_uses_same_fraction_for_coordinates_and_only_common_observed_traces():
    data = _data()
    poisoned = data.normalized_profiles.copy()
    for index in range(2):
        poisoned[index, 0, :, ~data.observed_trace_mask[index]] = np.nan
    coords, profiles, masks = observed_neighbor_profile_mixup(
        replace(data, normalized_profiles=poisoned),
        np.array([0]),
        max_fraction=0.5,
        rng=np.random.default_rng(501),
    )
    fraction = coords[0, 2]
    assert 0 <= fraction <= 0.5
    expected_mask = data.observed_trace_mask[0] & data.observed_trace_mask[1]
    np.testing.assert_array_equal(masks[0], expected_mask)
    expected = np.zeros_like(profiles[0])
    expected[0, :, expected_mask] = (1 - fraction) * data.normalized_profiles[
        0, 0, :, expected_mask
    ] + fraction * data.normalized_profiles[1, 0, :, expected_mask]
    np.testing.assert_allclose(profiles[0], expected, rtol=1e-6)
    assert np.isfinite(profiles).all()
    assert profiles.dtype == np.float32
    again = observed_neighbor_profile_mixup(
        data, np.array([0]), max_fraction=0.5, rng=np.random.default_rng(501)
    )
    for left, right in zip((coords, profiles, masks), again, strict=True):
        np.testing.assert_array_equal(left, right)


def test_no_shared_observation_produces_no_synthetic_label():
    data = _data()
    masks = data.observed_trace_mask.copy()
    masks[1] = ~masks[0]
    coords, profiles, masks = observed_neighbor_profile_mixup(
        replace(data, observed_trace_mask=masks),
        np.array([0, 1]),
        max_fraction=0.2,
        rng=np.random.default_rng(1),
    )
    assert coords.shape == (0, 3)
    assert profiles.shape == (0, 1, 8, 8)
    assert masks.shape == (0, 8)


@pytest.mark.parametrize("fraction", [0, -1, 0.51, np.nan, True])
def test_invalid_mixup_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="max_fraction"):
        observed_neighbor_profile_mixup(
            _data(), np.array([0]), max_fraction=fraction, rng=np.random.default_rng(1)
        )
