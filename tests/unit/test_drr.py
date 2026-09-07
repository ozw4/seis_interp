from __future__ import annotations

import numpy as np
import pytest

from seis_interp.processing import drr
from seis_interp.processing.damped_rank_reduction import damped_rank_reduce
from seis_interp.processing.drr import (
    interpolate_drr_block,
    interpolate_drr_frequency_slice,
    select_drr_frequencies,
)
from seis_interp.processing.level_four_hankel import (
    average_level_four_hankel,
    hankelize_level_four,
)


def _frequency_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    shape = (3, 3, 3, 3)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return values, rng.random(shape) > 0.4


def test_one_frequency_iteration_matches_explicit_operator_composition() -> None:
    values, mask = _frequency_fixture()
    initial = np.zeros_like(values)
    initial[mask] = values[mask]
    expected = average_level_four_hankel(
        damped_rank_reduce(hankelize_level_four(initial), rank=2, damping_power=3), values.shape
    )
    expected[mask] = values[mask]

    actual = interpolate_drr_frequency_slice(values, mask, rank=2, damping_power=3, n_iterations=1)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(actual[~mask].imag)) > 0.01


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_frequency_reconstruction_preserves_observations_and_inputs(dtype) -> None:
    values, mask = _frequency_fixture()
    values = values.astype(dtype)
    original, original_mask = values.copy(), mask.copy()
    result = interpolate_drr_frequency_slice(values, mask, rank=2, damping_power=3, n_iterations=4)
    assert result.dtype == np.complex128
    np.testing.assert_array_equal(result[mask], values[mask])
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(mask, original_mask)
    assert not np.shares_memory(result, values)


def test_frequency_missing_values_are_ignored_and_results_are_reproducible() -> None:
    values, mask = _frequency_fixture()
    results = []
    for replacement in (0, 1e100 + 1e100j, np.nan, 0):
        variant = values.copy()
        variant[~mask] = replacement
        results.append(
            interpolate_drr_frequency_slice(variant, mask, rank=2, damping_power=3, n_iterations=3)
        )
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mode", ["missing", "observed", "zero_observations"])
def test_frequency_special_cases(mode: str) -> None:
    values, mask = _frequency_fixture()
    if mode == "missing":
        mask[:] = False
        values[:] = np.nan
    elif mode == "observed":
        mask[:] = True
    else:
        values[mask] = 0
    result = interpolate_drr_frequency_slice(values, mask, rank=2, damping_power=3, n_iterations=2)
    expected = values if mode == "observed" else np.zeros_like(values)
    np.testing.assert_array_equal(result, expected)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("name", ["rank", "damping_power", "n_iterations"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, np.bool_(True)])
def test_frequency_settings_are_validated_even_for_all_missing(name, invalid) -> None:
    values, mask = _frequency_fixture()
    mask[:] = False
    kwargs = dict(rank=2, damping_power=3, n_iterations=2)
    kwargs[name] = invalid
    with pytest.raises(ValueError, match=name):
        interpolate_drr_frequency_slice(values, mask, **kwargs)


def test_frequency_rank_is_checked_against_actual_hankel_shape() -> None:
    values = np.ones((2, 3, 1, 3), dtype=np.complex128)
    with pytest.raises(ValueError, match=r"rank.*Hankel matrix shape \(8, 4\)"):
        interpolate_drr_frequency_slice(
            values, np.ones(values.shape, dtype=bool), rank=4, damping_power=2, n_iterations=1
        )


@pytest.mark.parametrize(
    "invalid", ["dimensions", "dtype", "mask_dtype", "mask_shape", "nan", "empty_axis"]
)
def test_frequency_array_contracts(invalid: str) -> None:
    values, mask = _frequency_fixture()
    if invalid == "dimensions":
        values = values[0]
    elif invalid == "dtype":
        values = values.real
    elif invalid == "mask_dtype":
        mask = mask.astype(int)
    elif invalid == "mask_shape":
        mask = mask[0]
    elif invalid == "nan":
        values[mask] = np.nan
    else:
        values, mask = values[:0], mask[:0]
    with pytest.raises(ValueError):
        interpolate_drr_frequency_slice(values, mask, rank=2, damping_power=3, n_iterations=1)


def test_low_rank_complex_waves_improve_missing_snr_over_zero_fill() -> None:
    shape = (4, 4, 4, 4)
    coordinates = np.indices(shape)
    phase_a = sum(f * coordinates[j] / shape[j] for j, f in enumerate((0.3, 0.6, -0.4, 0.2)))
    phase_b = sum(f * coordinates[j] / shape[j] for j, f in enumerate((-0.6, 0.2, 0.5, -0.3)))
    truth = np.exp(2j * np.pi * phase_a) + 0.6 * np.exp(2j * np.pi * phase_b + 0.7j)
    mask = np.random.default_rng(42).random(shape) > 0.4

    result = interpolate_drr_frequency_slice(truth, mask, rank=2, damping_power=3, n_iterations=8)

    reference_energy = np.sum(np.abs(truth[~mask]) ** 2)
    error_energy = np.sum(np.abs(result[~mask] - truth[~mask]) ** 2)
    assert 10 * np.log10(reference_energy / error_energy) > 3.0


def _block_fixture(time_count=5, dtype=np.float64):
    rng = np.random.default_rng(21)
    values = rng.normal(size=(time_count, 3, 3, 1, 1)).astype(dtype)
    mask = rng.random(values.shape[1:]) > 0.4
    return values, mask, 0.032 + np.arange(time_count) * 0.008


def _block_kwargs():
    return dict(
        rank=1, damping_power=3, n_iterations=2, frequency_min_hz=0.0, frequency_max_hz=None
    )


@pytest.mark.parametrize("time_count,fft_length", [(3, 4), (4, 4), (5, 8), (6, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_block_matches_explicit_padded_fft_reference(time_count, fft_length, dtype) -> None:
    values, mask, times = _block_fixture(time_count, dtype)
    original, original_mask, original_times = values.copy(), mask.copy(), times.copy()
    zero_filled = np.zeros(values.shape, dtype=np.float64)
    zero_filled[:, mask] = values[:, mask]
    transformed = np.fft.rfft(zero_filled, n=fft_length, axis=0, norm="ortho")
    for index in range(len(transformed)):
        reconstructed = interpolate_drr_frequency_slice(
            transformed[index], mask, rank=1, damping_power=3, n_iterations=2
        )
        if index in (0, fft_length // 2):
            reconstructed = reconstructed.real
        transformed[index] = reconstructed
    expected = np.fft.irfft(transformed, n=fft_length, axis=0, norm="ortho")
    expected = expected[:time_count].astype(dtype)
    expected[:, mask] = values[:, mask]

    actual = interpolate_drr_block(values, mask, times, **_block_kwargs())

    assert actual.shape == values.shape
    assert actual.dtype == dtype
    assert np.isrealobj(actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual[:, mask], values[:, mask])
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(times, original_times)


def test_block_clears_dc_and_nyquist_imaginary_parts_before_inverse(monkeypatch) -> None:
    values, mask, times = _block_fixture()
    real_irfft = np.fft.irfft
    captured = []

    def fake_slice(observed, observed_trace_mask, **kwargs):
        return np.full(observed.shape, 2 + 5j, dtype=np.complex128)

    def capture_inverse(spectrum, **kwargs):
        captured.append(spectrum.copy())
        assert kwargs == dict(n=8, axis=0, norm="ortho")
        return real_irfft(spectrum, **kwargs)

    monkeypatch.setattr(drr, "interpolate_drr_frequency_slice", fake_slice)
    monkeypatch.setattr(np.fft, "irfft", capture_inverse)
    interpolate_drr_block(values, mask, times, **_block_kwargs())

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0][[0, -1]].imag, 0)
    np.testing.assert_array_equal(captured[0][1:-1].imag, 5)


def test_block_band_limits_are_inclusive_and_missing_outside_bins_stay_zero() -> None:
    values, mask, times = _block_fixture(8)
    times = np.arange(8) * 0.125
    kwargs = _block_kwargs() | dict(frequency_min_hz=1.0, frequency_max_hz=2.0)
    result = interpolate_drr_block(values, mask, times, **kwargs)
    spectrum = np.fft.rfft(result[:, ~mask], axis=0, norm="ortho")
    np.testing.assert_allclose(spectrum[[0, 3, 4]], 0, atol=1e-13)
    assert np.linalg.norm(spectrum[1]) > 0
    assert np.linalg.norm(spectrum[2]) > 0
    np.testing.assert_array_equal(result[:, mask], values[:, mask])


def test_block_ignores_missing_values_and_time_origin() -> None:
    values, mask, times = _block_fixture()
    results = []
    for replacement in (0, 1e100, np.nan):
        variant = values.copy()
        variant[:, ~mask] = replacement
        results.append(interpolate_drr_block(variant, mask, times, **_block_kwargs()))
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=1e-13, atol=1e-13)
    shifted = interpolate_drr_block(values, mask, times + 10, **_block_kwargs())
    np.testing.assert_allclose(shifted, results[0], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mode", ["missing", "observed", "zero_observations"])
def test_block_special_cases(mode) -> None:
    values, mask, times = _block_fixture()
    if mode == "missing":
        mask[:] = False
        values[:] = np.nan
    elif mode == "observed":
        mask[:] = True
    else:
        values[:, mask] = 0
    result = interpolate_drr_block(values, mask, times, **_block_kwargs())
    expected = values if mode == "observed" else np.zeros_like(values)
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0]),
        np.array([0.0, 0.1, 0.21]),
        np.array([0.0, 0.1, 0.1]),
        np.array([0.1, 0.0]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0j, 1j]),
        np.ones((2, 2)),
    ],
)
def test_frequency_selection_rejects_invalid_time_axis(times) -> None:
    with pytest.raises(ValueError, match="time_s"):
        select_drr_frequencies(times, frequency_min_hz=0, frequency_max_hz=None)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (-1, None),
        (np.nan, None),
        (True, None),
        (0, np.inf),
        (0, True),
        (1, 1),
        (1, 0),
        (0, 5),
        (0.1, 0.2),
        (5, None),
    ],
)
def test_frequency_selection_rejects_invalid_or_empty_ranges(lower, upper) -> None:
    with pytest.raises(ValueError, match="frequency"):
        select_drr_frequencies(np.arange(8) * 0.125, frequency_min_hz=lower, frequency_max_hz=upper)


def test_float32_uniform_time_axis_and_frequency_metadata() -> None:
    times = np.arange(64, 128, dtype=np.float32) * np.float32(0.008)
    selected = select_drr_frequencies(times, frequency_min_hz=5.0, frequency_max_hz=None)
    assert selected.sample_interval_s == pytest.approx(0.008)
    assert selected.fft_length == 64
    assert selected.bin_indices[0] == 3
    assert selected.bin_indices[-1] == 32
    assert len(selected.bin_indices) == 30
    assert selected.frequencies_hz[-1] == pytest.approx(62.5)


@pytest.mark.parametrize(
    "invalid",
    [
        "rank",
        "time_shape",
        "time_uniform",
        "time_count",
        "dtype",
        "mask",
        "observed_nan",
        "frequency",
    ],
)
def test_block_validation_applies_before_special_cases(invalid) -> None:
    values, mask, times = _block_fixture()
    mask[:] = True
    kwargs = _block_kwargs()
    if invalid == "rank":
        kwargs["rank"] = 4
    elif invalid == "time_shape":
        times = times[:-1]
    elif invalid == "time_uniform":
        times[2] += 0.001
    elif invalid == "time_count":
        values, times = values[:1], times[:1]
    elif invalid == "dtype":
        values = values.astype(int)
    elif invalid == "mask":
        mask = np.broadcast_to(mask, values.shape)
    elif invalid == "observed_nan":
        values[0, 0, 0, 0, 0] = np.nan
    else:
        kwargs["frequency_max_hz"] = 100
    with pytest.raises(ValueError):
        interpolate_drr_block(values, mask, times, **kwargs)


def test_block_plane_waves_improve_missing_trace_snr() -> None:
    shape = (4, 4, 3, 3)
    grid = np.indices(shape)
    phase = sum(f * grid[j] / shape[j] for j, f in enumerate((0.2, 0.5, -0.3, 0.4)))
    times = np.arange(8) * 0.125
    truth = np.cos(2 * np.pi * (times[:, None, None, None, None] + phase))
    mask = np.random.default_rng(7).random(shape) > 0.35
    result = interpolate_drr_block(
        truth,
        mask,
        times,
        rank=1,
        damping_power=3,
        n_iterations=6,
        frequency_min_hz=0,
        frequency_max_hz=None,
    )
    reference_energy = np.sum(truth[:, ~mask] ** 2)
    error_energy = np.sum((truth[:, ~mask] - result[:, ~mask]) ** 2)
    assert 10 * np.log10(reference_energy / error_energy) > 3
