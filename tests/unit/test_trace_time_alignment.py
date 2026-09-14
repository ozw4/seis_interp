import numpy as np
import pytest

from seis_interp.processing.trace_time_alignment import (
    align_receiver_y_time,
    alignment_time_padding,
    validate_time_alignment,
)


@pytest.mark.parametrize("step", [-3, 3])
def test_alignment_is_exact_trace_permutation_with_exact_inverse(step):
    values = np.random.default_rng(1).normal(size=(16, 2, 3, 2, 8)).astype(np.float32)
    original = values.copy()
    options = {"receiver_y_shift_samples_per_cell": step, "boundary": "circular"}
    aligned = align_receiver_y_time(values, options)
    for y in range(8):
        np.testing.assert_array_equal(aligned[..., y], np.roll(values[..., y], step * (y - 4), 0))
    np.testing.assert_array_equal(align_receiver_y_time(aligned, options, inverse=True), values)
    np.testing.assert_array_equal(np.sort(aligned, axis=0), np.sort(values, axis=0))
    np.testing.assert_array_equal(values, original)
    assert aligned.dtype == values.dtype
    assert aligned.flags.c_contiguous


@pytest.mark.parametrize("step", [0, True, 0.5, float("nan"), "3"])
def test_alignment_rejects_invalid_shifts(step):
    with pytest.raises(ValueError, match="nonzero integer"):
        validate_time_alignment({"receiver_y_shift_samples_per_cell": step, "boundary": "circular"})


def test_alignment_rejects_ambiguous_boundaries_and_nonfinite_values():
    options = {"receiver_y_shift_samples_per_cell": 3, "boundary": "circular"}
    with pytest.raises(ValueError, match="requires"):
        validate_time_alignment({"receiver_y_shift_samples_per_cell": 3})
    with pytest.raises(ValueError, match="circular"):
        validate_time_alignment({**options, "boundary": "zero"})
    with pytest.raises(ValueError, match="finite"):
        align_receiver_y_time(np.full((8, 1, 1, 1, 8), np.nan), options)


@pytest.mark.parametrize("step", [-3, 3])
def test_zero_padded_alignment_preserves_all_samples_and_exact_inverse(step):
    values = np.random.default_rng(1).normal(size=(16, 2, 3, 2, 8)).astype(np.float32)
    options = {"receiver_y_shift_samples_per_cell": step, "boundary": "zero_pad"}
    assert alignment_time_padding(8, options) == 24
    aligned = align_receiver_y_time(values, options)
    assert aligned.shape == (40, 2, 3, 2, 8)
    shifts = [step * (y - 4) for y in range(8)]
    for y, shift in enumerate(shifts):
        offset = shift - min(shifts)
        np.testing.assert_array_equal(aligned[offset : offset + 16, ..., y], values[..., y])
        assert not aligned[:offset, ..., y].any()
        assert not aligned[offset + 16 :, ..., y].any()
    np.testing.assert_array_equal(align_receiver_y_time(aligned, options, inverse=True), values)
    np.testing.assert_allclose(
        np.square(aligned, dtype=np.float64).sum(0), np.square(values, dtype=np.float64).sum(0)
    )
    with pytest.raises(ValueError, match="physical time count"):
        align_receiver_y_time(values, options, inverse=True)


@pytest.mark.parametrize("step", [-3.0625, 3.0625])
def test_fourier_alignment_shifts_resolved_frequencies_and_preserves_nyquist(step):
    time = np.arange(64, dtype=np.float64)
    wave = np.sin(2 * np.pi * 7 * time / 64)
    nyquist = 0.1 * (-1.0) ** time
    values = np.broadcast_to((2 + wave + nyquist)[:, None, None, None, None], (64, 1, 1, 1, 8))
    options = {"receiver_y_shift_samples_per_cell": step, "boundary": "fourier_periodic"}
    aligned = align_receiver_y_time(values, options)
    for y in range(8):
        shifted = np.sin(2 * np.pi * 7 * (time - step * (y - 4)) / 64)
        np.testing.assert_allclose(aligned[:, 0, 0, 0, y], 2 + shifted + nyquist, atol=1e-13)


@pytest.mark.parametrize("dtype,tolerance", [(np.float32, 3e-7), (np.float64, 2e-14)])
@pytest.mark.parametrize("time_count", [63, 64])
def test_fourier_alignment_preserves_trace_energy_and_has_an_inverse(dtype, tolerance, time_count):
    values = np.random.default_rng(2).normal(size=(time_count, 2, 3, 2, 8)).astype(dtype)
    options = {"receiver_y_shift_samples_per_cell": 3.0625, "boundary": "fourier_periodic"}
    assert alignment_time_padding(8, options) == 0
    aligned = align_receiver_y_time(values, options)
    assert aligned.dtype == dtype
    np.testing.assert_allclose(
        np.square(aligned, dtype=np.float64).sum(0),
        np.square(values, dtype=np.float64).sum(0),
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        align_receiver_y_time(aligned, options, inverse=True),
        values,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("step", [True, 0, float("nan"), float("inf"), float("-inf"), "3.0625"])
def test_fourier_alignment_rejects_invalid_shifts(step):
    with pytest.raises(ValueError, match="finite and nonzero"):
        validate_time_alignment(
            {"receiver_y_shift_samples_per_cell": step, "boundary": "fourier_periodic"}
        )
