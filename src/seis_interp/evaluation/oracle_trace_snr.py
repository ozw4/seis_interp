"""Energy-based global S/N and the strict success threshold for formal runs."""

from __future__ import annotations

import math

PRIMARY_METRIC = "oracle_per_trace_unit_rms_global_snr_db"
SUCCESS_COMPARISON = "strictly_greater_than"


def global_snr_db_from_energies(
    signal_energy: float,
    error_energy: float,
) -> float:
    """Return the global S/N in decibels from positive finite energies."""
    if not math.isfinite(signal_energy) or signal_energy <= 0.0:
        raise ValueError("validation signal energy must be positive and finite")
    if not math.isfinite(error_energy) or error_energy <= 0.0:
        raise ValueError("validation error energy must be positive and finite")
    return 10.0 * math.log10(signal_energy / error_energy)


def passes_success_threshold(metric_db: float, threshold_db: float) -> bool:
    """Return whether the metric strictly exceeds the success threshold."""
    return bool(metric_db > threshold_db)
