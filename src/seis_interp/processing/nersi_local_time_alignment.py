"""Fit constant local receiver-y shears from complete observed trace pairs only."""

import numpy as np


def fit_observed_receiver_y_shear(values, observed_mask, *, candidates, distances):
    """Maximize mean unit-RMS circular correlation across observed receiver-y pairs.

    Positive shear delays larger receiver-y indices, matching the NeRSI time
    alignment convention. The real Nyquist coefficient is left unchanged.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    if (
        candidates.ndim != 1
        or not len(candidates)
        or not np.isfinite(candidates).all()
        or np.any(np.diff(candidates) <= 0)
    ):
        raise ValueError("shear candidates must be finite and strictly increasing")
    if (
        values.ndim != 5
        or observed_mask.dtype != np.bool_
        or observed_mask.shape != values.shape[1:]
        or not observed_mask.any()
    ):
        raise ValueError("shear fitting requires a volume and nonempty whole-trace O mask")
    if (
        not distances
        or any(
            isinstance(d, bool) or not isinstance(d, int) or not 1 <= d < values.shape[-1]
            for d in distances
        )
        or len(set(distances)) != len(distances)
    ):
        raise ValueError("receiver-y distances must be unique positive in-volume integers")
    selected = values[:, observed_mask].astype(np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("observed traces must be finite")
    peak = np.max(np.abs(selected), axis=0)
    selected /= np.where(peak > 0, peak, 1)
    rms = np.sqrt(np.mean(selected**2, axis=0))
    selected /= np.where(rms > 0, rms, 1)
    normalized = np.zeros(values.shape, dtype=np.float64)
    normalized[:, observed_mask] = selected
    time_count, receiver_y = values.shape[0], values.shape[-1]
    profiles = normalized.transpose(1, 2, 3, 4, 0).reshape(-1, receiver_y, time_count)
    mask = observed_mask.reshape(-1, receiver_y)
    spectra = np.fft.rfft(profiles, axis=-1)
    frequencies = np.fft.rfftfreq(time_count)
    weights = np.full(len(frequencies), 2.0)
    weights[0] = 1
    if time_count % 2 == 0:
        weights[-1] = 1
    scores = np.zeros(len(candidates))
    pair_count = 0
    for distance in distances:
        pairs = mask[:, :-distance] & mask[:, distance:]
        cross = (spectra[:, :-distance].conj() * spectra[:, distance:])[pairs].sum(axis=0)
        phase = np.exp(-2j * np.pi * candidates[:, None] * distance * frequencies[None, :])
        if time_count % 2 == 0:
            phase[:, -1] = 1
        scores += (phase * cross[None, :] * weights).sum(axis=-1).real / time_count**2
        pair_count += int(pairs.sum())
    if not pair_count:
        raise ValueError("no observed trace pairs at the configured receiver-y distances")
    scores /= pair_count
    best = int(np.argmax(scores))
    return {
        "receiver_y_shift_samples_per_cell": float(candidates[best]),
        "observed_pair_count": pair_count,
        "candidates": candidates.tolist(),
        "mean_observed_correlation": scores.tolist(),
        "fit_domain": "observed_only",
    }
