from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from seis_interp.processing import drr_windows
from seis_interp.processing.drr_windows import interpolate_drr_volume


def _parameters() -> dict[str, object]:
    return {
        "rank": 1,
        "damping_power": 4,
        "n_iterations": 2,
        "frequency_min_hz": 0.0,
        "frequency_max_hz": None,
    }


def test_no_window_and_covering_window_return_the_single_block_core_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.arange(5 * 3**4, dtype=np.float32).reshape(5, 3, 3, 3, 3)
    mask = np.ones(observed.shape[1:], dtype=np.bool_)
    mask[..., 1] = False
    time_s = np.arange(5) * 0.004
    expected = observed.copy()
    expected[:, ~mask] = -7.5
    calls = 0

    def core(
        block: np.ndarray,
        block_mask: np.ndarray,
        block_time: np.ndarray,
        **parameters: object,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert block is observed
        assert block_mask is mask
        assert block_time is time_s
        assert parameters == _parameters()
        return expected.copy()

    monkeypatch.setattr(drr_windows, "interpolate_drr_block", core)
    for window, overlap in ((None, None), ((5, 4, 5, 6), (4, 3, 4, 5))):
        result = interpolate_drr_volume(
            observed,
            mask,
            time_s,
            spatial_window_shape=window,
            spatial_overlap=overlap,
            **_parameters(),  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(result.values, expected)
        assert result.block_count == 1
        assert result.empty_block_count == 0
        assert result.uncovered_trace_count == 0
    assert calls == 2


def test_windows_slice_all_four_spatial_axes_in_lexicographic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (4, 6, 4, 4, 5)
    observed = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    mask = np.ones(shape[1:], dtype=np.bool_)
    time_s = 0.256 + np.arange(shape[0]) * 0.004
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def identity_core(
        block: np.ndarray,
        block_mask: np.ndarray,
        block_time: np.ndarray,
        **parameters: object,
    ) -> np.ndarray:
        assert block_time is time_s
        assert parameters == _parameters()
        calls.append((block.copy(), block_mask.copy()))
        return block.copy()

    monkeypatch.setattr(drr_windows, "interpolate_drr_block", identity_core)
    result = interpolate_drr_volume(
        observed,
        mask,
        time_s,
        spatial_window_shape=(3, 3, 3, 3),
        spatial_overlap=(1, 1, 1, 1),
        **_parameters(),  # type: ignore[arg-type]
    )

    expected_starts = list(product((0, 2, 3), (0, 1), (0, 1), (0, 2)))
    assert len(calls) == len(expected_starts)
    for (block, block_mask), starts in zip(calls, expected_starts, strict=True):
        spatial_slices = tuple(slice(start, start + 3) for start in starts)
        np.testing.assert_array_equal(block, observed[(slice(None), *spatial_slices)])
        np.testing.assert_array_equal(block_mask, mask[spatial_slices])
        assert block.shape[0] == shape[0]
    np.testing.assert_array_equal(result.values, observed)
    assert result.block_count == 24
    assert result.empty_block_count == 0
    assert result.uncovered_trace_count == 0


@pytest.mark.parametrize("constant", [0.0, -3.5, 8.0])
def test_positive_hann_blending_preserves_constant_missing_predictions_at_edges(
    monkeypatch: pytest.MonkeyPatch, constant: float
) -> None:
    observed = np.full((4, 4, 3, 3, 5), constant, dtype=np.float64)
    mask = np.ones(observed.shape[1:], dtype=np.bool_)
    mask[..., 1::2] = False

    def constant_core(block: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        return np.full_like(block, constant)

    monkeypatch.setattr(drr_windows, "interpolate_drr_block", constant_core)
    result = interpolate_drr_volume(
        observed,
        mask,
        np.arange(4) * 0.004,
        spatial_window_shape=(3, 3, 3, 3),
        spatial_overlap=(1, 1, 1, 1),
        **_parameters(),  # type: ignore[arg-type]
    )

    np.testing.assert_allclose(result.values, constant, rtol=0.0, atol=1e-14)
    assert result.uncovered_trace_count == 0


def test_blending_uses_interior_hann_weights_only_after_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.zeros((3, 1, 1, 1, 4), dtype=np.float64)
    observed[..., 0] = 7.0
    observed[..., 3] = -7.0
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)
    mask[..., (0, 3)] = True

    def constant_core(
        block: np.ndarray, block_mask: np.ndarray, *args: object, **kwargs: object
    ) -> np.ndarray:
        observed_value = block[:, block_mask]
        assert np.all(np.abs(observed_value) == 7.0)
        return np.full_like(block, 10.0 if observed_value[0, 0] > 0 else 20.0)

    monkeypatch.setattr(drr_windows, "interpolate_drr_block", constant_core)
    result = interpolate_drr_volume(
        observed,
        mask,
        np.arange(3) * 0.004,
        spatial_window_shape=(1, 1, 1, 3),
        spatial_overlap=(0, 0, 0, 2),
        **_parameters(),  # type: ignore[arg-type]
    )

    expected_trace_values = np.array([7.0, 40.0 / 3.0, 50.0 / 3.0, -7.0])
    np.testing.assert_allclose(
        result.values.reshape(3, 4), np.tile(expected_trace_values, (3, 1)), rtol=1e-15
    )
    np.testing.assert_array_equal(result.values[:, mask], observed[:, mask])


def test_oversized_axes_use_available_extents_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.ones((5, 1, 3, 1, 5), dtype=np.float32)
    mask = np.ones(observed.shape[1:], dtype=np.bool_)
    shapes: list[tuple[int, ...]] = []

    def core(block: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        shapes.append(block.shape)
        return block.copy()

    monkeypatch.setattr(drr_windows, "interpolate_drr_block", core)
    result = interpolate_drr_volume(
        observed,
        mask,
        np.arange(5) * 0.004,
        spatial_window_shape=(4, 5, 2, 3),
        spatial_overlap=(3, 4, 1, 1),
        **_parameters(),  # type: ignore[arg-type]
    )

    assert shapes == [(5, 1, 3, 1, 3)] * 2
    np.testing.assert_array_equal(result.values, observed)
    assert result.uncovered_trace_count == 0


def test_empty_blocks_do_not_dilute_neighbors_and_uncovered_count_is_per_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.zeros((4, 1, 1, 1, 5), dtype=np.float64)
    observed[..., 0] = 7.0
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)
    mask[..., 0] = True
    calls = 0

    def core(
        block: np.ndarray, block_mask: np.ndarray, *args: object, **kwargs: object
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert np.any(block_mask)
        return np.full_like(block, 10.0)

    monkeypatch.setattr(drr_windows, "interpolate_drr_block", core)
    result = interpolate_drr_volume(
        observed,
        mask,
        np.arange(4) * 0.004,
        spatial_window_shape=(1, 1, 1, 3),
        spatial_overlap=(0, 0, 0, 2),
        **_parameters(),  # type: ignore[arg-type]
    )

    assert calls == 1
    assert result.block_count == 3
    assert result.empty_block_count == 2
    assert result.uncovered_trace_count == 2
    np.testing.assert_array_equal(
        result.values.reshape(4, 5), np.tile([7.0, 10.0, 10.0, 0.0, 0.0], (4, 1))
    )


@pytest.mark.parametrize(
    ("window", "overlap", "block_count"),
    [(None, None, 1), ((1, 1, 1, 6), (0, 0, 0, 5), 1), ((1, 1, 1, 3), (0, 0, 0, 2), 3)],
)
def test_all_empty_inputs_skip_core_and_count_spatial_traces(
    monkeypatch: pytest.MonkeyPatch,
    window: tuple[int, ...] | None,
    overlap: tuple[int, ...] | None,
    block_count: int,
) -> None:
    observed = np.full((4, 1, 1, 1, 5), np.nan, dtype=np.float32)
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)

    def unexpected_call(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("empty windows must not call DRR")

    monkeypatch.setattr(drr_windows, "interpolate_drr_block", unexpected_call)
    result = interpolate_drr_volume(
        observed,
        mask,
        np.arange(4) * 0.004,
        spatial_window_shape=window,  # type: ignore[arg-type]
        spatial_overlap=overlap,  # type: ignore[arg-type]
        **_parameters(),  # type: ignore[arg-type]
    )

    assert result.block_count == result.empty_block_count == block_count
    assert result.uncovered_trace_count == 5
    assert result.values.dtype == observed.dtype
    np.testing.assert_array_equal(result.values, np.zeros_like(observed))


@pytest.mark.parametrize(
    ("window", "overlap", "match"),
    [
        (None, (0, 0, 0, 0), "both"),
        ((3, 3, 3, 3), None, "both"),
        ((3, 3, 3), (0, 0, 0, 0), "four"),
        ((3, 3, 3, 3), (0, 0, 0), "four"),
        ((3, 0, 3, 3), (0, 0, 0, 0), "positive"),
        ((3, -1, 3, 3), (0, 0, 0, 0), "positive"),
        ((3, True, 3, 3), (0, 0, 0, 0), "integers"),
        ((3, 3.0, 3, 3), (0, 0, 0, 0), "integers"),
        ((3, 3, 3, 3), (0, -1, 0, 0), "nonnegative"),
        ((3, 3, 3, 3), (0, True, 0, 0), "integers"),
        ((3, 3, 3, 3), (0, 3, 0, 0), "less"),
        ((3, 3, 3, 3), (0, 4, 0, 0), "less"),
        (np.array(3), (0, 0, 0, 0), "four"),
        ("3333", (0, 0, 0, 0), "four"),
    ],
)
def test_invalid_window_specifications_are_rejected(
    window: object, overlap: object, match: str
) -> None:
    observed = np.zeros((4, 3, 3, 3, 3), dtype=np.float32)
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)

    with pytest.raises(ValueError, match=match):
        interpolate_drr_volume(
            observed,
            mask,
            np.arange(4) * 0.004,
            spatial_window_shape=window,  # type: ignore[arg-type]
            spatial_overlap=overlap,  # type: ignore[arg-type]
            **_parameters(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"rank": 2}, "rank"),
        ({"damping_power": 0}, "damping_power"),
        ({"n_iterations": 0}, "n_iterations"),
        ({"frequency_min_hz": -1.0}, "frequency_min_hz"),
        ({"frequency_max_hz": 126.0}, "Nyquist"),
        ({"frequency_min_hz": 10.0, "frequency_max_hz": 11.0}, "no rFFT bins"),
    ],
)
def test_empty_windows_still_validate_drr_parameters_and_frequencies(
    changes: dict[str, object], match: str
) -> None:
    observed = np.zeros((4, 1, 1, 1, 5), dtype=np.float32)
    mask = np.zeros(observed.shape[1:], dtype=np.bool_)
    parameters = _parameters() | changes

    with pytest.raises(ValueError, match=match):
        interpolate_drr_volume(
            observed,
            mask,
            np.arange(4) * 0.004,
            spatial_window_shape=(1, 1, 1, 3),
            spatial_overlap=(0, 0, 0, 2),
            **parameters,  # type: ignore[arg-type]
        )


def test_oversized_window_rank_uses_available_extent_even_when_empty() -> None:
    observed = np.zeros((4, 1, 1, 1, 5), dtype=np.float32)
    parameters = _parameters() | {"rank": 3}

    with pytest.raises(ValueError, match="rank"):
        interpolate_drr_volume(
            observed,
            np.zeros(observed.shape[1:], dtype=np.bool_),
            np.arange(4) * 0.004,
            spatial_window_shape=(1, 1, 1, 9),
            spatial_overlap=(0, 0, 0, 8),
            **parameters,  # type: ignore[arg-type]
        )


def test_empty_volume_still_rejects_nonuniform_time_sampling() -> None:
    observed = np.zeros((4, 1, 1, 1, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="uniformly sampled"):
        interpolate_drr_volume(
            observed,
            np.zeros(observed.shape[1:], dtype=np.bool_),
            np.array([0.0, 0.004, 0.009, 0.012]),
            spatial_window_shape=None,
            spatial_overlap=None,
            **_parameters(),  # type: ignore[arg-type]
        )


def test_time_varying_mask_is_rejected() -> None:
    observed = np.ones((4, 1, 1, 1, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="four-dimensional spatial shape"):
        interpolate_drr_volume(
            observed,
            np.ones(observed.shape, dtype=np.bool_),
            np.arange(4) * 0.004,
            spatial_window_shape=None,
            spatial_overlap=None,
            **_parameters(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tiny_real_drr_windows_are_finite_reproducible_and_ignore_missing_values(
    dtype: type[np.floating],
) -> None:
    rng = np.random.default_rng(27)
    source = rng.normal(size=(5, 3, 1, 1, 5)).astype(dtype)
    mask = rng.random(source.shape[1:]) > 0.5
    mask[0] = True
    predictions = []
    for replacement in (0.0, 1e30, np.nan, 0.0):
        observed = source.copy()
        observed[:, ~mask] = replacement
        original = observed.copy()
        result = interpolate_drr_volume(
            observed,
            mask,
            np.arange(5) * 0.004,
            spatial_window_shape=(3, 1, 1, 3),
            spatial_overlap=(1, 0, 0, 1),
            **_parameters(),  # type: ignore[arg-type]
        )
        assert result.values.shape == observed.shape
        assert result.values.dtype == dtype
        assert np.all(np.isfinite(result.values))
        np.testing.assert_array_equal(result.values[:, mask], source[:, mask])
        np.testing.assert_array_equal(observed, original)
        predictions.append(result.values)
    for prediction in predictions[1:]:
        np.testing.assert_array_equal(prediction, predictions[0])
