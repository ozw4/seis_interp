from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from seis_interp.processing import pocs_windows as pocs_windows_module
from seis_interp.processing.pocs import interpolate_pocs_block
from seis_interp.processing.pocs_windows import (
    _axis_window_starts,
    interpolate_pocs_volume,
)


def _pocs_parameters() -> dict[str, object]:
    return {
        "n_iterations": 3,
        "threshold_start": 1.0,
        "threshold_end": 0.1,
    }


def test_none_and_covering_window_match_the_single_block_core() -> None:
    rng = np.random.default_rng(5)
    observed = rng.normal(size=(5, 2, 3, 2, 2)).astype(np.float32)
    mask = rng.random(observed.shape[1:]) > 0.45
    expected = interpolate_pocs_block(observed, mask, **_pocs_parameters())  # type: ignore[arg-type]

    without_windows = interpolate_pocs_volume(
        observed,
        mask,
        window_shape=None,
        overlap=None,
        **_pocs_parameters(),  # type: ignore[arg-type]
    )
    covering_window = interpolate_pocs_volume(
        observed,
        mask,
        window_shape=(8, 4, 5, 3, 6),
        overlap=(7, 3, 4, 2, 5),
        **_pocs_parameters(),  # type: ignore[arg-type]
    )

    for result in (without_windows, covering_window):
        np.testing.assert_array_equal(result.values, expected)
        assert result.block_count == 1
        assert result.empty_block_count == 0
        assert result.uncovered_sample_count == 0


@pytest.mark.parametrize(
    ("length", "window_length", "overlap", "expected"),
    [
        (3, 5, 4, (0,)),
        (1, 1, 0, (0,)),
        (10, 4, 1, (0, 3, 6)),
        (11, 4, 1, (0, 3, 6, 7)),
        (8, 3, 2, (0, 1, 2, 3, 4, 5)),
    ],
)
def test_axis_window_starts_cover_the_end_without_duplicates(
    length: int,
    window_length: int,
    overlap: int,
    expected: tuple[int, ...],
) -> None:
    assert _axis_window_starts(length, window_length, overlap) == expected


def test_multiwindow_slices_all_five_axes_in_lexicographic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (3, 4, 3, 3, 4)
    observed = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    mask = np.ones(shape[1:], dtype=np.bool_)
    window = (2, 3, 2, 2, 3)
    overlap = (1, 1, 1, 1, 1)
    starts_by_axis = tuple(
        _axis_window_starts(length, window_length, axis_overlap)
        for length, window_length, axis_overlap in zip(
            shape,
            window,
            overlap,
            strict=True,
        )
    )
    calls: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []

    def identity_block(
        block: np.ndarray,
        block_mask: np.ndarray,
        **parameters: object,
    ) -> np.ndarray:
        assert parameters == _pocs_parameters()
        calls.append((float(block[0, 0, 0, 0, 0]), block.shape, block_mask.shape))
        return block.copy()

    monkeypatch.setattr(pocs_windows_module, "interpolate_pocs_block", identity_block)

    result = interpolate_pocs_volume(
        observed,
        mask,
        window_shape=window,
        overlap=overlap,
        **_pocs_parameters(),  # type: ignore[arg-type]
    )

    expected_calls = []
    for starts in product(*starts_by_axis):
        slices = tuple(
            slice(start, min(start + window_length, length))
            for start, window_length, length in zip(starts, window, shape, strict=True)
        )
        block = observed[slices]
        expected_calls.append((float(block[0, 0, 0, 0, 0]), block.shape, block.shape[1:]))
    assert calls == expected_calls
    assert result.block_count == len(expected_calls)
    assert result.empty_block_count == 0
    assert result.uncovered_sample_count == 0
    np.testing.assert_allclose(result.values, observed, rtol=1.0e-15, atol=1.0e-15)


@pytest.mark.parametrize("constant", [0.0, -3.5, 8.0])
def test_positive_hann_blending_preserves_constant_blocks_at_volume_edges(
    monkeypatch: pytest.MonkeyPatch,
    constant: float,
) -> None:
    observed = np.full((4, 3, 2, 2, 4), constant, dtype=np.float64)
    mask = np.ones(observed.shape[1:], dtype=np.bool_)

    def identity_block(
        block: np.ndarray,
        block_mask: np.ndarray,
        **parameters: object,
    ) -> np.ndarray:
        assert block_mask.shape == block.shape[1:]
        assert parameters == _pocs_parameters()
        return block.copy()

    monkeypatch.setattr(pocs_windows_module, "interpolate_pocs_block", identity_block)

    result = interpolate_pocs_volume(
        observed,
        mask,
        window_shape=(3, 2, 2, 2, 3),
        overlap=(2, 1, 1, 1, 2),
        **_pocs_parameters(),  # type: ignore[arg-type]
    )

    np.testing.assert_allclose(result.values, observed, rtol=0.0, atol=1.0e-14)
    assert result.uncovered_sample_count == 0


def test_oversized_and_length_one_axes_use_only_the_available_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 1, 2, 1, 3)
    observed = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape[1:], dtype=np.bool_)
    block_shapes: list[tuple[int, ...]] = []

    def identity_block(
        block: np.ndarray,
        block_mask: np.ndarray,
        **parameters: object,
    ) -> np.ndarray:
        assert block_mask.shape == block.shape[1:]
        block_shapes.append(block.shape)
        return block.copy()

    monkeypatch.setattr(pocs_windows_module, "interpolate_pocs_block", identity_block)

    result = interpolate_pocs_volume(
        observed,
        mask,
        window_shape=(5, 4, 2, 2, 2),
        overlap=(4, 3, 1, 1, 1),
        **_pocs_parameters(),  # type: ignore[arg-type]
    )

    assert block_shapes == [(2, 1, 2, 1, 2), (2, 1, 2, 1, 2)]
    np.testing.assert_allclose(result.values, observed, rtol=0.0, atol=1.0e-6)


def test_empty_blocks_do_not_dilute_neighbors_and_uncovered_samples_stay_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.zeros((1, 1, 1, 1, 5), dtype=np.float64)
    observed[..., 0] = 7.0
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)
    mask[..., 0] = True
    call_count = 0

    def constant_prediction(
        block: np.ndarray,
        block_mask: np.ndarray,
        **parameters: object,
    ) -> np.ndarray:
        nonlocal call_count
        assert np.any(block_mask)
        call_count += 1
        return np.full(block.shape, 10.0, dtype=block.dtype)

    monkeypatch.setattr(
        pocs_windows_module,
        "interpolate_pocs_block",
        constant_prediction,
    )

    result = interpolate_pocs_volume(
        observed,
        mask,
        window_shape=(1, 1, 1, 1, 3),
        overlap=(0, 0, 0, 0, 2),
        **_pocs_parameters(),  # type: ignore[arg-type]
    )

    assert call_count == 1
    assert result.block_count == 3
    assert result.empty_block_count == 2
    assert result.uncovered_sample_count == 2
    np.testing.assert_array_equal(result.values.reshape(-1), [7.0, 10.0, 10.0, 0.0, 0.0])


def test_all_empty_windows_return_zero_and_count_every_sample_as_uncovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.full((2, 1, 1, 1, 5), np.nan, dtype=np.float32)
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)

    def unexpected_call(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("empty blocks must not be interpolated")

    monkeypatch.setattr(pocs_windows_module, "interpolate_pocs_block", unexpected_call)

    result = interpolate_pocs_volume(
        observed,
        mask,
        window_shape=(1, 1, 1, 1, 3),
        overlap=(0, 0, 0, 0, 2),
        **_pocs_parameters(),  # type: ignore[arg-type]
    )

    assert result.block_count == 6
    assert result.empty_block_count == 6
    assert result.uncovered_sample_count == observed.size
    assert result.values.dtype == observed.dtype
    np.testing.assert_array_equal(result.values, np.zeros_like(result.values))


def test_windowed_interpolation_ignores_missing_values_and_is_reproducible() -> None:
    rng = np.random.default_rng(82)
    source = rng.normal(size=(6, 2, 3, 2, 4)).astype(np.float32)
    mask = rng.random(source.shape[1:]) > 0.45
    variants = []
    for replacement in (0.0, 1.0e30, np.nan):
        observed = source.copy()
        observed[:, ~mask] = replacement
        original = observed.copy()
        result = interpolate_pocs_volume(
            observed,
            mask,
            window_shape=(4, 2, 2, 2, 3),
            overlap=(2, 1, 1, 1, 1),
            **_pocs_parameters(),  # type: ignore[arg-type]
        )
        assert result.values.dtype == observed.dtype
        np.testing.assert_array_equal(result.values[:, mask], source[:, mask])
        assert np.array_equal(observed, original, equal_nan=True)
        variants.append(result)

    np.testing.assert_array_equal(variants[0].values, variants[1].values)
    np.testing.assert_array_equal(variants[0].values, variants[2].values)
    repeated = interpolate_pocs_volume(
        np.where(mask[None], source, 0.0).astype(np.float32),
        mask,
        window_shape=(4, 2, 2, 2, 3),
        overlap=(2, 1, 1, 1, 1),
        **_pocs_parameters(),  # type: ignore[arg-type]
    )
    np.testing.assert_array_equal(repeated.values, variants[0].values)


@pytest.mark.parametrize(
    ("window_shape", "overlap", "message"),
    [
        (None, (0, 0, 0, 0, 0), "both"),
        ((2, 2, 2, 2, 2), None, "both"),
        ((2, 2, 2, 2), (0, 0, 0, 0, 0), "five"),
        ((2, 2, 2, 2, 2), (0, 0, 0, 0), "five"),
        ((2, 0, 2, 2, 2), (0, 0, 0, 0, 0), "positive"),
        ((2, True, 2, 2, 2), (0, 0, 0, 0, 0), "integers"),
        ((2, 2, 2, 2, 2), (0, -1, 0, 0, 0), "nonnegative"),
        ((2, 2, 2, 2, 2), (0, True, 0, 0, 0), "integers"),
        ((2, 2, 2, 2, 2), (0, 2, 0, 0, 0), "less"),
        ((2, 2, 2, 2, 2), (0, 3, 0, 0, 0), "less"),
    ],
)
def test_invalid_window_specifications_are_rejected(
    window_shape: tuple[int, ...] | None,
    overlap: tuple[int, ...] | None,
    message: str,
) -> None:
    observed = np.ones((3, 2, 2, 2, 2), dtype=np.float64)
    mask = np.ones(observed.shape[1:], dtype=np.bool_)

    with pytest.raises(ValueError, match=message):
        interpolate_pocs_volume(
            observed,
            mask,
            window_shape=window_shape,  # type: ignore[arg-type]
            overlap=overlap,  # type: ignore[arg-type]
            **_pocs_parameters(),  # type: ignore[arg-type]
        )


def test_windowed_volume_rejects_a_time_varying_mask() -> None:
    observed = np.ones((3, 2, 2, 2, 2), dtype=np.float64)
    mask = np.ones(observed.shape, dtype=np.bool_)

    with pytest.raises(ValueError, match="shape"):
        interpolate_pocs_volume(
            observed,
            mask,
            window_shape=None,
            overlap=None,
            **_pocs_parameters(),  # type: ignore[arg-type]
        )
