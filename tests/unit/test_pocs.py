from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seis_interp.processing.pocs import (
    interpolate_pocs_block,
    interpolate_pocs_frequency_slice,
)


def _reference_frequency_slice(
    observed: np.ndarray,
    mask: np.ndarray,
    *,
    n_iterations: int,
    threshold_start: float,
    threshold_end: float,
    update_threshold_maximum: bool = False,
) -> np.ndarray:
    zero_filled = np.zeros(observed.shape, dtype=np.complex128)
    zero_filled[mask] = observed[mask]
    if not np.any(mask):
        return zero_filled

    threshold_maximum = np.max(np.abs(np.fft.fftn(zero_filled, axes=(0, 1, 2, 3), norm="ortho")))
    if threshold_maximum == 0.0:
        return zero_filled

    reconstructed = zero_filled.copy()
    for ratio in np.geomspace(threshold_start, threshold_end, n_iterations):
        spectrum = np.fft.fftn(reconstructed, axes=(0, 1, 2, 3), norm="ortho")
        spectrum[np.abs(spectrum) < ratio * threshold_maximum] = 0.0
        reconstructed = np.fft.ifftn(spectrum, axes=(0, 1, 2, 3), norm="ortho")
        reconstructed[mask] = zero_filled[mask]
        if update_threshold_maximum:
            threshold_maximum = np.max(
                np.abs(np.fft.fftn(reconstructed, axes=(0, 1, 2, 3), norm="ortho"))
            )
    return reconstructed


def _reference_block_from_frequency_slices(
    observed: np.ndarray,
    mask: np.ndarray,
    *,
    n_iterations: int,
    threshold_start: float,
    threshold_end: float,
) -> np.ndarray:
    zero_filled = np.zeros(observed.shape, dtype=np.float64)
    zero_filled[:, mask] = observed[:, mask]
    transformed = np.fft.rfft(zero_filled, axis=0, norm="ortho")
    for frequency_index in range(len(transformed)):
        reconstructed = interpolate_pocs_frequency_slice(
            transformed[frequency_index],
            mask,
            n_iterations=n_iterations,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
        )
        if frequency_index == 0 or (
            observed.shape[0] % 2 == 0 and frequency_index == len(transformed) - 1
        ):
            reconstructed = reconstructed.real
        transformed[frequency_index] = reconstructed
    result = np.fft.irfft(
        transformed,
        n=observed.shape[0],
        axis=0,
        norm="ortho",
    ).astype(observed.dtype)
    result[:, mask] = observed[:, mask]
    return result


def _pocs_kwargs() -> dict[str, Any]:
    return {
        "n_iterations": 5,
        "threshold_start": 1.0,
        "threshold_end": 0.1,
    }


def test_frequency_slice_all_observed_is_an_unmodified_complex128_copy() -> None:
    rng = np.random.default_rng(4)
    observed = (rng.normal(size=(2, 3, 2, 4)) + 1j * rng.normal(size=(2, 3, 2, 4))).astype(
        np.complex64
    )
    original = observed.copy()
    mask = np.ones(observed.shape, dtype=np.bool_)

    result = interpolate_pocs_frequency_slice(observed, mask, **_pocs_kwargs())

    assert result.dtype == np.complex128
    assert result is not observed
    np.testing.assert_array_equal(result, observed.astype(np.complex128))
    np.testing.assert_array_equal(observed, original)


@pytest.mark.parametrize("observed_value", [0.0, 3.0 + 4.0j])
def test_frequency_slice_all_missing_returns_finite_zero(observed_value: complex) -> None:
    observed = np.full((2, 2, 2, 2), observed_value, dtype=np.complex128)
    mask = np.zeros(observed.shape, dtype=np.bool_)

    result = interpolate_pocs_frequency_slice(observed, mask, **_pocs_kwargs())

    assert np.all(np.isfinite(result))
    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_frequency_slice_zero_observations_return_zero() -> None:
    observed = np.full((2, 2, 2, 2), 100.0 + 20.0j, dtype=np.complex128)
    mask = np.zeros(observed.shape, dtype=np.bool_)
    mask[0, 0, 0, 0] = True
    observed[mask] = 0.0

    result = interpolate_pocs_frequency_slice(observed, mask, **_pocs_kwargs())

    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_frequency_slice_preserves_observed_values_exactly_after_interpolation() -> None:
    rng = np.random.default_rng(18)
    observed = (rng.normal(size=(2, 3, 2, 3)) + 1j * rng.normal(size=(2, 3, 2, 3))).astype(
        np.complex64
    )
    mask = rng.random(observed.shape) > 0.45

    result = interpolate_pocs_frequency_slice(observed, mask, **_pocs_kwargs())

    np.testing.assert_array_equal(result[mask], observed[mask].astype(np.complex128))
    assert np.any(np.abs(result[~mask]) > 0.0)


def test_frequency_slice_ignores_every_unobserved_value_including_nan() -> None:
    rng = np.random.default_rng(21)
    source = (rng.normal(size=(2, 3, 2, 4)) + 1j * rng.normal(size=(2, 3, 2, 4))).astype(
        np.complex128
    )
    mask = rng.random(source.shape) > 0.4
    variants = []
    for replacement in (0.0, 1.0e30 - 4.0e29j, np.nan + 1j * np.nan):
        observed = source.copy()
        observed[~mask] = replacement
        variants.append(interpolate_pocs_frequency_slice(observed, mask, **_pocs_kwargs()))

    np.testing.assert_array_equal(variants[0], variants[1])
    np.testing.assert_array_equal(variants[0], variants[2])


def test_frequency_slice_keeps_complex_phase_and_matches_independent_reference() -> None:
    rng = np.random.default_rng(1)
    observed = (rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(size=(2, 3, 4, 5))).astype(
        np.complex128
    )
    mask = rng.random(observed.shape) > 0.45
    expected = _reference_frequency_slice(
        observed,
        mask,
        n_iterations=4,
        threshold_start=1.0,
        threshold_end=0.2,
    )

    result = interpolate_pocs_frequency_slice(
        observed,
        mask,
        n_iterations=4,
        threshold_start=1.0,
        threshold_end=0.2,
    )

    np.testing.assert_allclose(result, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.max(np.abs(result[~mask].imag)) > 0.1


def test_frequency_slice_keeps_coefficients_equal_to_the_hard_threshold() -> None:
    rng = np.random.default_rng(99)
    observed = (rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2))).astype(
        np.complex128
    )
    mask = rng.random(observed.shape) < 0.35
    expected = _reference_frequency_slice(
        observed,
        mask,
        n_iterations=2,
        threshold_start=1.0,
        threshold_end=1.0,
    )

    result = interpolate_pocs_frequency_slice(
        observed,
        mask,
        n_iterations=2,
        threshold_start=1.0,
        threshold_end=1.0,
    )

    np.testing.assert_allclose(result, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.max(np.abs(result[~mask])) > 0.5


def test_frequency_slice_uses_the_fixed_initial_threshold_maximum() -> None:
    rng = np.random.default_rng(123)
    observed = (rng.normal(size=(2, 3, 2, 2)) + 1j * rng.normal(size=(2, 3, 2, 2))).astype(
        np.complex128
    )
    mask = rng.random(observed.shape) > 0.45
    parameters = {
        "n_iterations": 5,
        "threshold_start": 0.9,
        "threshold_end": 0.15,
    }
    fixed_reference = _reference_frequency_slice(observed, mask, **parameters)
    updated_reference = _reference_frequency_slice(
        observed,
        mask,
        update_threshold_maximum=True,
        **parameters,
    )

    result = interpolate_pocs_frequency_slice(observed, mask, **parameters)

    np.testing.assert_allclose(result, fixed_reference, rtol=1.0e-13, atol=1.0e-13)
    assert np.max(np.abs(result - updated_reference)) > 0.1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_iterations", 1),
        ("n_iterations", True),
        ("n_iterations", 2.0),
        ("threshold_start", 0.0),
        ("threshold_start", 1.1),
        ("threshold_start", np.inf),
        ("threshold_start", True),
        ("threshold_end", 0.0),
        ("threshold_end", np.nan),
        ("threshold_end", 0.75),
    ],
)
def test_frequency_slice_rejects_invalid_parameters(field: str, value: object) -> None:
    observed = np.ones((2, 2, 2, 2), dtype=np.complex128)
    mask = np.ones(observed.shape, dtype=np.bool_)
    parameters: dict[str, object] = {
        "n_iterations": 2,
        "threshold_start": 0.5,
        "threshold_end": 0.25,
    }
    parameters[field] = value

    with pytest.raises(
        ValueError, match=field if field != "threshold_end" or value != 0.75 else "threshold"
    ):
        interpolate_pocs_frequency_slice(observed, mask, **parameters)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("observed", "mask", "message"),
    [
        (np.ones((2, 2, 2), dtype=np.complex128), np.ones((2, 2, 2), dtype=bool), "four"),
        (np.ones((2, 2, 2, 2), dtype=np.float64), np.ones((2, 2, 2, 2), dtype=bool), "complex"),
        (np.ones((2, 2, 2, 2), dtype=np.complex128), np.ones((2, 2, 2), dtype=bool), "shape"),
        (
            np.ones((2, 2, 2, 2), dtype=np.complex128),
            np.ones((2, 2, 2, 2), dtype=np.int8),
            "boolean",
        ),
    ],
)
def test_frequency_slice_rejects_invalid_array_contracts(
    observed: np.ndarray,
    mask: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        interpolate_pocs_frequency_slice(observed, mask, **_pocs_kwargs())


def test_frequency_slice_rejects_nonfinite_observed_values_only() -> None:
    observed = np.ones((2, 2, 2, 2), dtype=np.complex128)
    mask = np.zeros(observed.shape, dtype=np.bool_)
    mask[0, 0, 0, 0] = True
    observed[mask] = np.nan

    with pytest.raises(ValueError, match="finite"):
        interpolate_pocs_frequency_slice(observed, mask, **_pocs_kwargs())


@pytest.mark.parametrize(
    ("time_count", "dtype", "spatial_shape"),
    [
        (7, np.float32, (1, 3, 1, 2)),
        (8, np.float64, (2, 3, 2, 3)),
    ],
)
def test_block_matches_direct_rfft_and_frequency_slice_reference(
    time_count: int,
    dtype: type[np.floating[Any]],
    spatial_shape: tuple[int, int, int, int],
) -> None:
    rng = np.random.default_rng(time_count)
    observed = rng.normal(size=(time_count, *spatial_shape)).astype(dtype)
    mask = rng.random(spatial_shape) > 0.4
    original = observed.copy()
    expected = _reference_block_from_frequency_slices(
        observed,
        mask,
        n_iterations=4,
        threshold_start=1.0,
        threshold_end=0.1,
    )

    result = interpolate_pocs_block(
        observed,
        mask,
        n_iterations=4,
        threshold_start=1.0,
        threshold_end=0.1,
    )

    assert result.shape == observed.shape
    assert result.dtype == observed.dtype
    assert np.isrealobj(result)
    np.testing.assert_allclose(result, expected, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_array_equal(result[:, mask], observed[:, mask])
    np.testing.assert_array_equal(observed, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_block_all_observed_is_identity(dtype: type[np.floating[Any]]) -> None:
    observed = np.arange(6 * 2 * 2 * 2 * 2, dtype=dtype).reshape(6, 2, 2, 2, 2)
    mask = np.ones(observed.shape[1:], dtype=np.bool_)

    result = interpolate_pocs_block(observed, mask, **_pocs_kwargs())

    assert result is not observed
    assert result.dtype == observed.dtype
    np.testing.assert_array_equal(result, observed)


def test_block_all_missing_returns_finite_zero() -> None:
    observed = np.full((5, 2, 2, 2, 2), np.nan, dtype=np.float32)
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)

    result = interpolate_pocs_block(observed, mask, **_pocs_kwargs())

    assert np.all(np.isfinite(result))
    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_block_ignores_missing_values_and_preserves_an_observed_zero_trace() -> None:
    rng = np.random.default_rng(20)
    source = rng.normal(size=(7, 2, 3, 2, 3)).astype(np.float32)
    mask = rng.random(source.shape[1:]) > 0.5
    zero_trace_index = tuple(np.argwhere(mask)[0])
    source[(slice(None), *zero_trace_index)] = 0.0
    variants = []
    originals = []
    for replacement in (0.0, 1.0e30, np.nan):
        observed = source.copy()
        observed[:, ~mask] = replacement
        originals.append(observed.copy())
        variants.append(interpolate_pocs_block(observed, mask, **_pocs_kwargs()))
        assert np.array_equal(observed, originals[-1], equal_nan=True)

    np.testing.assert_array_equal(variants[0], variants[1])
    np.testing.assert_array_equal(variants[0], variants[2])
    np.testing.assert_array_equal(
        variants[0][(slice(None), *zero_trace_index)],
        np.zeros(source.shape[0], dtype=np.float32),
    )


@pytest.mark.parametrize("mask_kind", ["random_trace", "random_whole_ffid"])
def test_block_fixed_5d_fourier_fixture_improves_missing_trace_snr(mask_kind: str) -> None:
    shape = (32, 6, 8, 4, 6)
    indices = np.indices(shape)
    first_frequencies = (3, 1, 1, 1, 1)
    second_frequencies = (5, 2, -1, 1, -1)
    first_phase = sum(
        frequency * indices[axis] / shape[axis] for axis, frequency in enumerate(first_frequencies)
    )
    second_phase = sum(
        frequency * indices[axis] / shape[axis] for axis, frequency in enumerate(second_frequencies)
    )
    expected = np.cos(2.0 * np.pi * first_phase) + 0.6 * np.cos(2.0 * np.pi * second_phase + 0.3)
    rng = np.random.default_rng(42)
    if mask_kind == "random_trace":
        mask = rng.random(shape[1:]) >= 0.5
    else:
        whole_ffid_mask = rng.random(shape[1:3]) >= 0.5
        mask = np.broadcast_to(whole_ffid_mask[..., None, None], shape[1:]).copy()
    observed = np.zeros(shape, dtype=np.float64)
    observed[:, mask] = expected[:, mask]

    result = interpolate_pocs_block(
        observed,
        mask,
        n_iterations=60,
        threshold_start=1.0,
        threshold_end=0.01,
    )

    missing = ~mask
    zero_fill_error_energy = float(np.sum(expected[:, missing] ** 2))
    reconstructed_error_energy = float(np.sum((expected[:, missing] - result[:, missing]) ** 2))
    improvement_db = 10.0 * np.log10(zero_fill_error_energy / reconstructed_error_energy)
    assert improvement_db > 10.0


@pytest.mark.parametrize(
    ("observed", "mask", "message"),
    [
        (np.ones((5, 2, 2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=bool), "five"),
        (np.ones((5, 2, 2, 2, 2), dtype=np.int16), np.ones((2, 2, 2, 2), dtype=bool), "float"),
        (np.ones((5, 2, 2, 2, 2), dtype=np.float32), np.ones((5, 2, 2, 2, 2), dtype=bool), "shape"),
        (
            np.ones((5, 2, 2, 2, 2), dtype=np.float32),
            np.ones((2, 2, 2, 2), dtype=np.int8),
            "boolean",
        ),
    ],
)
def test_block_rejects_invalid_array_contracts(
    observed: np.ndarray,
    mask: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        interpolate_pocs_block(observed, mask, **_pocs_kwargs())


def test_block_rejects_nonfinite_observed_samples() -> None:
    observed = np.ones((5, 2, 2, 2, 2), dtype=np.float64)
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)
    mask[0, 0, 0, 0] = True
    observed[2, 0, 0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        interpolate_pocs_block(observed, mask, **_pocs_kwargs())
