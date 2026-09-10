from __future__ import annotations

import numpy as np
import pytest

from seis_interp.training.amplitude_scaling import (
    compute_observed_global_rms,
    normalize_by_global_rms,
    restore_physical_amplitude,
)


def test_observed_global_rms_matches_hand_calculation() -> None:
    values = np.array(
        [
            [[3.0, 99.0], [0.0, 99.0]],
            [[4.0, 99.0], [12.0, 99.0]],
        ]
    )
    observed_mask = np.array([[True, False], [True, False]])

    scale = compute_observed_global_rms(values, observed_mask)

    assert scale == pytest.approx(np.sqrt((3.0**2 + 4.0**2 + 12.0**2) / 4.0))


def test_evaluation_target_values_do_not_change_observed_global_rms() -> None:
    values = np.arange(1, 13, dtype=np.float64).reshape(3, 2, 2)
    observed_mask = np.array([[True, False], [True, False]])
    expected = compute_observed_global_rms(values, observed_mask)

    values[:, 0, 1] = np.finfo(np.float64).max

    assert compute_observed_global_rms(values, observed_mask) == expected


def test_qc_disabled_values_do_not_change_observed_global_rms() -> None:
    values = np.arange(1, 17, dtype=np.float64).reshape(4, 2, 2)
    observed_mask = np.array([[True, False], [False, True]])
    expected = compute_observed_global_rms(values, observed_mask)

    values[:, 1, 0] = -1.0e200

    assert compute_observed_global_rms(values, observed_mask) == expected


def test_float32_values_are_squared_and_accumulated_in_float64() -> None:
    values = np.array(
        [
            [[1.0e20, 0.0]],
            [[3.0, 0.0]],
            [[4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    observed_mask = np.array([[True, False]])
    observed = values[:, observed_mask].astype(np.float64)
    expected = np.sqrt(np.sum(np.square(observed), dtype=np.float64) / observed.size)

    scale = compute_observed_global_rms(values, observed_mask)

    assert np.isfinite(scale)
    assert scale == pytest.approx(expected, rel=1.0e-15)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_observed_global_rms_rejects_nonfinite_observed_values(nonfinite: float) -> None:
    values = np.ones((3, 2), dtype=np.float64)
    values[1, 0] = nonfinite

    with pytest.raises(ValueError, match="observed values must contain only finite"):
        compute_observed_global_rms(values, np.array([True, False]))


def test_observed_global_rms_rejects_no_observed_traces() -> None:
    with pytest.raises(ValueError, match="at least one trace"):
        compute_observed_global_rms(
            np.ones((3, 2), dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
        )


def test_observed_global_rms_rejects_zero_observed_energy() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        compute_observed_global_rms(
            np.zeros((3, 2), dtype=np.float32),
            np.array([True, False]),
        )


def test_normalize_then_restore_recovers_finite_physical_values() -> None:
    values = np.array([[-8.0, -0.5, 0.0], [0.25, 3.0, 17.0]], dtype=np.float64)

    normalized = normalize_by_global_rms(values, 2.5)
    restored = restore_physical_amplitude(normalized, 2.5)

    np.testing.assert_allclose(restored, values, rtol=1.0e-15, atol=0.0)


def test_global_rms_operations_do_not_modify_inputs() -> None:
    values = np.arange(1, 13, dtype=np.float32).reshape(3, 2, 2)
    observed_mask = np.array([[True, False], [False, True]])
    original_values = values.copy()
    original_mask = observed_mask.copy()

    scale = compute_observed_global_rms(values, observed_mask)
    normalized = normalize_by_global_rms(values, scale)
    restore_physical_amplitude(normalized, scale)

    np.testing.assert_array_equal(values, original_values)
    np.testing.assert_array_equal(observed_mask, original_mask)
