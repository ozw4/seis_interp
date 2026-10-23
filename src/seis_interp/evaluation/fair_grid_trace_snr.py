"""Per-trace SNR (dB) of a fair-grid final evaluation and its paired contrasts.

The per-trace SNR is ``10 log10(target energy / error energy)`` without an
epsilon (``trace_snr_db``, the C3 v3 metric); higher is better. Seed means and
contrasts accept finite values only: a perfect or zero-energy trace is counted
by ``summarize_trace_snr`` and rejected here rather than clipped.
"""

import numpy as np
import pandas as pd

from seis_interp.evaluation.forge_fair_grid_statistics import paired_contrasts
from seis_interp.evaluation.trace_snr import trace_snr_db

TRACE_CHUNK = 4096


def trace_snr_by_trace(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-trace SNR dB of matching ``[trace, sample]`` arrays, computed in trace chunks."""
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("prediction and target must be matching [trace, sample] arrays")
    return np.concatenate(
        [
            trace_snr_db(
                target[start : start + TRACE_CHUNK], prediction[start : start + TRACE_CHUNK]
            )
            for start in range(0, len(target), TRACE_CHUNK)
        ]
    )


def seed_averaged_trace_snr(per_seed_snr) -> np.ndarray:
    """Mean per-trace SNR dB over seeds; every value must be finite."""
    stacked = np.asarray(per_seed_snr, dtype=np.float64)
    if stacked.ndim != 2 or not stacked.shape[0]:
        raise ValueError("per-seed SNR must be a [seed, trace] array")
    nonfinite = int(np.count_nonzero(~np.isfinite(stacked)))
    if nonfinite:
        raise ValueError(f"{nonfinite} non-finite per-trace SNR values cannot enter a contrast")
    return stacked.mean(axis=0)


def paired_trace_snr_contrasts(snr, cluster_ids, weights, contrasts) -> pd.DataFrame:
    """``paired_contrasts`` on per-trace SNR dB: ``left - right`` dB, positive favours left.

    The per-method means are ``left/right_mean_trace_snr_db`` and
    ``fraction_left_higher_snr`` replaces the error table's lower-error fraction.
    """
    table = paired_contrasts(snr, cluster_ids, weights, contrasts).rename(
        columns={
            "left_mean_error": "left_mean_trace_snr_db",
            "right_mean_error": "right_mean_trace_snr_db",
        }
    )
    higher = [
        float(np.mean(np.asarray(snr[left]) > np.asarray(snr[right])))
        for _, left, right, _ in contrasts
    ]
    table.insert(
        table.columns.get_loc("fraction_left_lower_error"), "fraction_left_higher_snr", higher
    )
    return table.drop(columns="fraction_left_lower_error")
