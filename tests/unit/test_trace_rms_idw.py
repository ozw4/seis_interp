import numpy as np
import pytest

from seis_interp.processing.trace_rms_idw import (
    interpolate_trace_rms_idw,
    match_prediction_trace_rms,
)


def test_idw_hand_computed_weights_and_target_poisoning():
    values = np.array([2, np.nan, np.inf, 8], dtype=np.float32).reshape(1, 1, 1, 1, 4)
    mask = np.array([True, False, False, True]).reshape(1, 1, 1, 4)
    result = interpolate_trace_rms_idw(values, mask, radius=3, power=2, axis_scales=(1,) * 4)
    np.testing.assert_allclose(result.ravel(), [2, (2 + 8 / 4) / 1.25, (2 / 4 + 8) / 1.25, 8])
    values[:, ~mask] = -1e30
    np.testing.assert_array_equal(
        result,
        interpolate_trace_rms_idw(values, mask, radius=3, power=2, axis_scales=(1,) * 4),
    )


def test_zero_energy_and_stable_rms():
    mask = np.ones((1, 1, 1, 2), dtype=bool)
    values = np.array([[0, 1e200], [0, -1e200]]).reshape(2, 1, 1, 1, 2)
    result = interpolate_trace_rms_idw(values, mask, radius=1, power=2, axis_scales=(1,) * 4)
    np.testing.assert_array_equal(result.ravel(), [0, 1e200])


def test_axis_scales_control_anisotropic_distance_not_amplitudes():
    values = np.array([0, 2, 8, 0], dtype=np.float32).reshape(1, 1, 1, 2, 2)
    mask = np.array([False, True, True, False]).reshape(1, 1, 2, 2)
    result = interpolate_trace_rms_idw(values, mask, radius=1, power=2, axis_scales=(1, 1, 1, 2))
    assert result[0, 0, 1, 1] == pytest.approx((2 + 8 / 4) / 1.25)
    np.testing.assert_array_equal(result[mask], [2, 8])


def test_idw_rejects_uncovered_and_nonfinite_observed():
    mask = np.array([True, False, False, False]).reshape(1, 1, 1, 4)
    values = np.zeros((2, *mask.shape), dtype=np.float32)
    with pytest.raises(ValueError, match="uncovered"):
        interpolate_trace_rms_idw(values, mask, radius=1, power=2, axis_scales=(1,) * 4)
    values[:, mask] = np.nan
    with pytest.raises(ValueError, match="observed amplitudes"):
        interpolate_trace_rms_idw(values, mask, radius=3, power=2, axis_scales=(1,) * 4)


@pytest.mark.parametrize(
    "options", [{"radius": 0}, {"power": float("nan")}, {"axis_scales": (0,) * 4}]
)
def test_invalid_options(options):
    kwargs = {"radius": 1, "power": 2, "axis_scales": (1,) * 4, **options}
    with pytest.raises(ValueError):
        interpolate_trace_rms_idw(
            np.ones((2, 1, 1, 1, 2)), np.ones((1, 1, 1, 2), dtype=bool), **kwargs
        )


def test_prediction_rms_matching_preserves_direction_observed_and_zero_waveforms():
    values = np.array([[2, 3, 0], [-2, -3, 0]], dtype=np.float32).reshape(2, 1, 1, 1, 3)
    mask = np.array([True, False, False]).reshape(1, 1, 1, 3)
    rms = np.array([100, 6, 8], dtype=np.float64).reshape(mask.shape)
    result = match_prediction_trace_rms(values, rms, mask)
    np.testing.assert_array_equal(result.reshape(2, 3), [[2, 6, 0], [-2, -6, 0]])
    assert result.dtype == values.dtype
    assert values[0, 0, 0, 0, 1] == 3
    with pytest.raises(ValueError):
        match_prediction_trace_rms(values, -rms, mask)
