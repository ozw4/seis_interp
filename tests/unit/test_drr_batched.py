import numpy as np
import pytest
import torch

from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
from seis_interp.processing.drr_nonoverlapping import interpolate_drr_nonoverlapping
from seis_interp.processing.drr_windows import interpolate_drr_volume


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("frequency_limits", [(0.0, None), (20.0, 100.0)])
def test_batched_drr_matches_reference_and_ignores_targets(device, windowed, frequency_limits):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(31)
    values = rng.normal(size=(7, 2, 4, 2, 5)).astype(np.float32)
    mask = rng.random(values.shape[1:]) < 0.4
    time_s = np.arange(7) * 0.002
    kwargs = dict(
        rank=2,
        damping_power=3,
        n_iterations=3,
        frequency_min_hz=frequency_limits[0],
        frequency_max_hz=frequency_limits[1],
        spatial_window_shape=(2, 3, 2, 3) if windowed else None,
        spatial_overlap=(1, 1, 1, 1) if windowed else None,
    )
    expected = interpolate_drr_volume(values, mask, time_s, **kwargs)
    values[:, ~mask] = np.nan
    actual = interpolate_drr_volume_batched(
        values, mask, time_s, **kwargs, device=device, batch_matrix_limit=4000
    )
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual.values[:, mask], values[:, mask])
    assert actual.block_count == expected.block_count
    assert actual.empty_block_count == expected.empty_block_count
    assert actual.uncovered_trace_count == expected.uncovered_trace_count
    if device == "cpu":
        parallel = interpolate_drr_volume_batched(
            values, mask, time_s, **kwargs, device=device, batch_matrix_limit=4000, workers=2
        )
        np.testing.assert_array_equal(parallel.values, actual.values)


@pytest.mark.parametrize("visible", [False, True])
def test_empty_and_complete_masks(visible):
    values = np.ones((5, 1, 4, 1, 4), dtype=np.float32)
    mask = np.full(values.shape[1:], visible)
    result = interpolate_drr_volume_batched(
        values,
        mask,
        np.arange(5) * 0.002,
        rank=1,
        damping_power=3,
        n_iterations=2,
        frequency_min_hz=0.0,
        frequency_max_hz=None,
        spatial_window_shape=(1, 4, 1, 4),
        spatial_overlap=(0, 0, 0, 0),
    )
    np.testing.assert_array_equal(result.values, values if visible else np.zeros_like(values))
    assert result.uncovered_trace_count == (0 if visible else mask.size)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_zero_observations_keep_singular_reductions_finite(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = np.zeros((5, 1, 4, 1, 4), dtype=np.float64)
    mask = np.zeros(values.shape[1:], dtype=bool)
    mask[:, ::2, :, ::2] = True
    values[:, ~mask] = np.nan
    result = interpolate_drr_volume_batched(
        values,
        mask,
        np.arange(5) * 0.002,
        rank=2,
        damping_power=3,
        n_iterations=3,
        frequency_min_hz=0.0,
        frequency_max_hz=None,
        spatial_window_shape=None,
        spatial_overlap=None,
        device=device,
    )
    np.testing.assert_array_equal(result.values, np.zeros_like(values))


def test_frequency_chunk_limit_bounds_each_svd_and_preserves_results(monkeypatch):
    from seis_interp.processing import drr_batched

    rng = np.random.default_rng(45)
    values = rng.normal(size=(31, 2, 4, 2, 4)).astype(np.float32)
    mask = rng.random(values.shape[1:]) < 0.5
    options = dict(
        rank=2,
        damping_power=2,
        n_iterations=3,
        frequency_min_hz=0.0,
        frequency_max_hz=None,
        spatial_window_shape=(2, 4, 2, 4),
        spatial_overlap=(0,) * 4,
    )
    expected = interpolate_drr_volume_batched(values, mask, np.arange(31) * 0.008, **options)
    matrix_size = drr_batched._hankel_indices(mask.shape)[0].size
    calls = []
    original = drr_batched._reduce_numpy

    def recorded(current, *args):
        calls.append(len(current))
        return original(current, *args)

    monkeypatch.setattr(drr_batched, "_reduce_numpy", recorded)
    actual = interpolate_drr_volume_batched(
        values,
        mask,
        np.arange(31) * 0.008,
        **options,
        batch_matrix_limit=matrix_size * 2,
    )
    assert len(calls) > 3 and max(calls) <= 2
    np.testing.assert_array_equal(actual.values, expected.values)
    previous = interpolate_drr_nonoverlapping(values, mask, np.arange(31) * 0.008, **options)
    np.testing.assert_array_equal(actual.values, previous.values)


@pytest.mark.parametrize("shape", [(3, 7, 3, 9), (6, 9, 4, 7)])
def test_point_half_overlap_clips_and_shifts_nondivisible_windows(shape):
    rng = np.random.default_rng(13)
    values = rng.normal(size=(5, *shape)).astype(np.float32)
    mask = rng.random(shape) < 0.6
    options = dict(
        rank=2,
        damping_power=2,
        n_iterations=1,
        frequency_min_hz=0.0,
        frequency_max_hz=None,
        spatial_window_shape=(4, 4, 4, 4),
        spatial_overlap=(0, 2, 0, 2),
    )
    expected = interpolate_drr_volume(values, mask, np.arange(5) * 0.004, **options)
    actual = interpolate_drr_volume_batched(
        values,
        mask,
        np.arange(5) * 0.004,
        **(options | {"spatial_overlap": (0,) * 4}),
        overlap_policy="point_half",
        boundary_rule="clip_and_shift",
        batch_matrix_limit=1000,
    )
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual.values[:, mask], values[:, mask])
    assert actual.block_count == expected.block_count
    assert actual.uncovered_trace_count == 0


@pytest.mark.parametrize(
    "window,expected",
    [
        ((4, 6, 4, 6), (0, 3, 0, 3)),
        ((4, 8, 4, 8), (0, 4, 0, 4)),
        ((6, 8, 6, 8), (0, 4, 0, 4)),
        ((8, 12, 8, 12), (0, 6, 0, 6)),
        ((8, 16, 8, 16), (0, 8, 0, 8)),
    ],
)
def test_declared_overlap_mapping(window, expected, monkeypatch):
    from seis_interp.processing import drr_batched

    overlaps = []
    original = drr_batched._axis_window_starts

    def recorded(length, width, shared):
        overlaps.append(shared)
        return original(length, width, shared)

    monkeypatch.setattr(drr_batched, "_axis_window_starts", recorded)
    interpolate_drr_volume_batched(
        np.zeros((3, 3, 3, 3, 3), dtype=np.float32),
        np.zeros((3,) * 4, dtype=bool),
        np.arange(3) * 0.008,
        rank=1,
        damping_power=2,
        n_iterations=1,
        frequency_min_hz=0.0,
        frequency_max_hz=None,
        spatial_window_shape=window,
        spatial_overlap=(0,) * 4,
        overlap_policy="point_half",
    )
    assert tuple(overlaps) == expected
