import numpy as np
import pytest
import torch

from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
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
