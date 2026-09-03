from __future__ import annotations

import math

import pytest

from seis_interp.evaluation.oracle_trace_snr import (
    PRIMARY_METRIC,
    SUCCESS_COMPARISON,
    global_snr_db_from_energies,
    passes_success_threshold,
)


def test_global_snr_db_matches_known_energy_ratios() -> None:
    assert global_snr_db_from_energies(100.0, 1.0) == pytest.approx(20.0)
    assert global_snr_db_from_energies(10.0, 10.0) == pytest.approx(0.0)
    assert global_snr_db_from_energies(1.0, 100.0) == pytest.approx(-20.0)


@pytest.mark.parametrize("bad_signal", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_global_snr_db_rejects_non_positive_or_non_finite_signal(bad_signal: float) -> None:
    with pytest.raises(ValueError, match="signal energy must be positive and finite"):
        global_snr_db_from_energies(bad_signal, 1.0)


@pytest.mark.parametrize("bad_error", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_global_snr_db_rejects_non_positive_or_non_finite_error(bad_error: float) -> None:
    with pytest.raises(ValueError, match="error energy must be positive and finite"):
        global_snr_db_from_energies(1.0, bad_error)


def test_success_requires_strictly_greater_metric() -> None:
    assert passes_success_threshold(20.000001, 20.0) is True
    assert passes_success_threshold(20.0, 20.0) is False
    assert passes_success_threshold(19.999999, 20.0) is False


def test_primary_metric_and_comparison_constants() -> None:
    assert PRIMARY_METRIC == "oracle_per_trace_unit_rms_global_snr_db"
    assert SUCCESS_COMPARISON == "strictly_greater_than"
