from __future__ import annotations

import math

import numpy as np
import pytest

from seis_interp.evaluation.metrics import signal_to_noise_ratio_db


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
