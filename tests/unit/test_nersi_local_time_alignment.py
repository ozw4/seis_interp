import numpy as np
import pytest

from seis_interp.processing.nersi_local_time_alignment import fit_observed_receiver_y_shear


def test_known_fractional_shear_and_target_storage_invariance():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(64, 2, 2, 1))
    ft = np.fft.rfft(base, axis=0)
    values = np.empty((64, 2, 2, 1, 8))
    for y in range(8):
        phase = np.exp(2j * np.pi * np.fft.rfftfreq(64) * 1.5 * y)
        phase[-1] = 1
        values[..., y] = np.fft.irfft(ft * phase[:, None, None, None], n=64, axis=0)
    mask = np.ones(values.shape[1:], dtype=bool)
    mask[..., 3] = False
    options = dict(candidates=[1.0, 1.25, 1.5, 1.75, 2.0], distances=[1, 2, 3])
    expected = fit_observed_receiver_y_shear(values, mask, **options)
    assert expected["receiver_y_shift_samples_per_cell"] == 1.5
    assert max(expected["mean_observed_correlation"]) == pytest.approx(1.0)
    values[:, ~mask] = np.nan
    assert fit_observed_receiver_y_shear(values, mask, **options) == expected


@pytest.mark.parametrize(
    "candidates,distances",
    [([], [1]), ([1, 1], [1]), ([np.nan], [1]), ([1], [0]), ([1], [4]), ([1], [1, 1])],
)
def test_invalid_search(candidates, distances):
    with pytest.raises(ValueError):
        fit_observed_receiver_y_shear(
            np.ones((8, 1, 1, 1, 4)),
            np.ones((1, 1, 1, 4), dtype=bool),
            candidates=candidates,
            distances=distances,
        )


def test_no_pairs_rejected():
    mask = np.zeros((1, 1, 1, 4), dtype=bool)
    mask[..., 0] = True
    with pytest.raises(ValueError, match="no observed trace pairs"):
        fit_observed_receiver_y_shear(
            np.ones((8, 1, 1, 1, 4)), mask, candidates=[1.0, 2.0], distances=[1]
        )
