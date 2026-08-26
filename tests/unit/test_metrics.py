from __future__ import annotations

import math

import numpy as np
import pytest

from seis_interp.evaluation.metrics import (
    median_trace_correlation_coefficient,
    median_trace_signal_to_noise_ratio_db,
    signal_to_noise_ratio_db,
    trace_correlation_coefficient,
    trace_signal_to_noise_ratio_db,
)


def test_signal_to_noise_ratio_matches_known_energy_ratio() -> None:
    reference = np.array([1.0, -1.0])
    prediction = np.array([0.5, -0.5])

    assert signal_to_noise_ratio_db(reference, prediction) == pytest.approx(10.0 * math.log10(4.0))


def test_perfect_prediction_is_positive_infinity() -> None:
    assert signal_to_noise_ratio_db(np.array([1.0]), np.array([1.0])) == float("inf")


def test_rejects_shape_mismatch_and_zero_energy() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        signal_to_noise_ratio_db(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="energy"):
        signal_to_noise_ratio_db(np.zeros(2), np.ones(2))


def test_median_trace_ratio_is_not_dominated_by_one_high_energy_trace() -> None:
    reference = np.array(
        [
            [100.0, -100.0],
            [1.0, -1.0],
            [1.0, -1.0],
        ]
    )
    prediction = np.array(
        [
            [90.0, -90.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )

    per_trace = trace_signal_to_noise_ratio_db(reference, prediction)

    assert per_trace.dtype == np.float64
    assert per_trace == pytest.approx([20.0, 0.0, 0.0])
    assert median_trace_signal_to_noise_ratio_db(reference, prediction) == pytest.approx(0.0)
    assert signal_to_noise_ratio_db(reference, prediction) > 15.0


def test_perfectly_predicted_trace_is_positive_infinity() -> None:
    reference = np.array([[1.0, -1.0], [2.0, -2.0]])
    prediction = np.array([[1.0, -1.0], [0.0, 0.0]])

    assert trace_signal_to_noise_ratio_db(reference, prediction)[0] == float("inf")


def test_trace_ratio_rejects_one_dimensional_input_and_a_zero_energy_trace() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        trace_signal_to_noise_ratio_db(np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="trace energy"):
        trace_signal_to_noise_ratio_db(np.array([[1.0, -1.0], [0.0, 0.0]]), np.ones((2, 2)))


def test_trace_correlation_matches_identical_and_sign_reversed_traces() -> None:
    reference = np.array([[1.0, -1.0, 2.0], [-2.0, 0.0, 1.0]])

    assert trace_correlation_coefficient(reference, reference) == pytest.approx([1.0, 1.0])
    assert trace_correlation_coefficient(reference, -reference) == pytest.approx([-1.0, -1.0])


def test_trace_correlation_ignores_positive_scale_and_offset() -> None:
    reference = np.array([[1.0, -1.0, 2.0], [-2.0, 0.0, 1.0]])
    scale = np.array([[3.0], [5.0]])
    offset = np.array([[4.0], [-7.0]])
    prediction = scale * reference + offset

    assert trace_correlation_coefficient(reference, prediction) == pytest.approx([1.0, 1.0])


def test_trace_correlation_is_zero_for_constant_prediction() -> None:
    reference = np.array([[1.0, -1.0, 2.0], [-2.0, 0.0, 1.0]])

    assert trace_correlation_coefficient(reference, np.ones_like(reference)) == pytest.approx(
        [0.0, 0.0]
    )


def test_median_trace_correlation_matches_known_value() -> None:
    reference = np.array([[1.0, 0.0, -1.0]] * 3)
    prediction = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )

    correlation = trace_correlation_coefficient(reference, prediction)

    assert correlation == pytest.approx([1.0, 0.0, -1.0])
    assert median_trace_correlation_coefficient(reference, prediction) == pytest.approx(0.0)


def test_trace_correlation_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        trace_correlation_coefficient(np.ones(3), np.ones(3))
