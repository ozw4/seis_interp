"""Split each trace at its own arrival time and score the two halves separately.

``masked_trace_relative_mse`` and ``mean_trace_snr_db`` add two different failures
into one number: not reproducing the signal, and putting energy where the trace has
none. A method that mixes neighbours without a time shift must leak energy ahead of
the arrival of its nearest neighbour, and that leakage is invisible in a whole-trace
score. Splitting at a geometry-only window separates the two.

The window is declared, never picked from a waveform: a first-break pick would read
the target and break the scoring discipline. ``t < intercept_s + offset / velocity``
is a straight line under the measured arrival curve, so every sample it selects
precedes the arrival. The constants come from observed traces only -- the model's
own input -- with a margin, and they are recorded with every result.

The headline number is the amplitude a method invented before the arrival: the RMS
of its own prediction inside the window, averaged over traces with no weighting. It
is in the data's units, it stays defined when the reference window is silent, and it
is read beside the reference's own pre-arrival amplitude on the same traces.

A zero prediction invents nothing, so this number alone cannot rank methods. Read it
beside the post-arrival SNR, never on its own.
"""

from numbers import Integral, Real

import numpy as np

# Measured from observed traces alone (median envelope per offset octile, rise above
# three times its own early level), then rounded down in intercept and up in velocity
# so the declared window ends before the measured arrival at every offset.
#   FORGE  arrivals  t = 0.314 s + offset / 3865 m/s; declared 0.22 s + offset / 5500
#   C3     arrivals  t = 0.375 s + offset / 5227 m/s; declared 0.25 s + offset / 7000
ARRIVAL_WINDOWS = {
    "forge_2017": {"intercept_s": 0.22, "velocity_m_per_s": 5500.0},
    "seg_c3_na": {"intercept_s": 0.25, "velocity_m_per_s": 7000.0},
}
MINIMUM_WINDOW_SAMPLES = 16


def arrival_window_ends(
    offsets,
    *,
    intercept_s,
    velocity_m_per_s,
    sample_interval_s,
    sample_count,
    minimum_samples=MINIMUM_WINDOW_SAMPLES,
):
    """One past the last pre-arrival sample of each trace, from geometry alone.

    Raises rather than silently shortening: a declared window that cannot give every
    trace ``minimum_samples`` is the wrong window for this geometry, and a window that
    swallows a trace leaves nothing to score after it.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError("offsets must be a finite one-dimensional array")
    if np.any(offsets < 0):
        raise ValueError("offsets must be nonnegative distances in metres")
    intercept_s = _nonnegative_float(intercept_s, "intercept_s")
    velocity_m_per_s = _positive_float(velocity_m_per_s, "velocity_m_per_s")
    sample_interval_s = _positive_float(sample_interval_s, "sample_interval_s")
    sample_count = _positive_integer(sample_count, "sample_count")
    minimum_samples = _positive_integer(minimum_samples, "minimum_samples")
    ends = np.floor((intercept_s + offsets / velocity_m_per_s) / sample_interval_s).astype(np.int64)
    if ends.min() < minimum_samples:
        raise ValueError(
            f"the declared window gives {ends.min()} samples to the shortest trace, "
            f"below the required {minimum_samples}"
        )
    if ends.max() >= sample_count:
        raise ValueError("the declared window leaves no post-arrival samples to score")
    return ends


def windowed_waveform_columns(prediction, target, window_ends):
    """Pre-arrival amplitude and post-arrival fidelity for matching trace arrays.

    ``pre_prediction_rms`` is the invented amplitude and ``pre_target_rms`` the real
    one, both in the data's units and both defined even where the reference window is
    silent. The ratios are kept for reference: ``pre_energy_ratio`` divides by the
    whole trace's reference energy and ``pre_excess_ratio`` by the reference's own
    pre-arrival energy, which is undefined exactly where that energy is zero.
    """
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("prediction and target must be matching finite trace arrays")
    ends = np.asarray(window_ends, dtype=np.int64)
    if ends.shape != (len(y),) or ends.min() < 1 or ends.max() >= y.shape[1]:
        raise ValueError("window_ends must give every trace a pre- and post-arrival part")
    before = np.arange(y.shape[1])[None, :] < ends[:, None]
    pre_prediction = np.square(np.where(before, p, 0.0)).sum(axis=1)
    pre_target = np.square(np.where(before, y, 0.0)).sum(axis=1)
    post_target = np.square(np.where(before, 0.0, y)).sum(axis=1)
    post_error = np.square(np.where(before, 0.0, p - y)).sum(axis=1)
    target_energy = np.square(y).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        post_snr = 10 * (np.log10(post_target) - np.log10(post_error))
    trace_rms = np.sqrt(target_energy / y.shape[1])
    return {
        "window_samples": ends,
        "pre_prediction_energy": pre_prediction,
        "pre_target_energy": pre_target,
        "post_target_energy": post_target,
        "post_error_energy": post_error,
        # The amplitude the method invented before the arrival, and the amplitude that is
        # really there, in the same physical units. Averaging these over traces needs no
        # weighting rule, and both stay defined when the reference window is silent.
        "pre_prediction_rms": np.sqrt(pre_prediction / ends),
        "pre_target_rms": np.sqrt(pre_target / ends),
        "pre_prediction_rms_over_trace_rms": _ratio(np.sqrt(pre_prediction / ends), trace_rms),
        "pre_target_rms_over_trace_rms": _ratio(np.sqrt(pre_target / ends), trace_rms),
        "pre_energy_ratio": _ratio(pre_prediction, target_energy),
        "pre_target_share": _ratio(pre_target, target_energy),
        "pre_excess_ratio": _ratio(pre_prediction, pre_target),
        # The study's per-trace rule: fall back to the sample count when the part is silent,
        # so a windowed relative MSE can be averaged the same way as the whole-trace one.
        "post_relative_mse": post_error / np.where(post_target > 0, post_target, y.shape[1] - ends),
        "post_snr_db": post_snr,
    }


def _ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan),
        where=denominator > 0,
    )


def _positive_float(value, name):
    value = _finite_float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(value, name):
    value = _finite_float(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
