import numpy as np
import pytest
import torch

from seis_interp.processing.pocs_batched import interpolate_pocs_volume_batched
from seis_interp.processing.pocs_windows import interpolate_pocs_volume


@pytest.mark.parametrize("samples", [7, 8])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batched_pocs_matches_reference_and_ignores_targets(samples, windowed, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(51)
    values = rng.normal(size=(samples, 2, 4, 2, 3)).astype(np.float32)
    mask = rng.random(values.shape[1:]) < 0.4
    kwargs = dict(
        n_iterations=9,
        threshold_start=0.95,
        threshold_end=0.2,
        window_shape=(5, 2, 3, 1, 3) if windowed else None,
        overlap=(2, 1, 1, 0, 0) if windowed else None,
    )
    expected = interpolate_pocs_volume(values, mask, **kwargs)
    values[:, ~mask] = np.nan
    actual = interpolate_pocs_volume_batched(
        values, mask, **kwargs, device=device, batch_sample_limit=200
    )
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5, atol=1e-6)
    if device == "cpu":
        np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.values[:, mask], values[:, mask])
    assert actual.block_count == expected.block_count
    assert actual.empty_block_count == expected.empty_block_count
    assert actual.uncovered_sample_count == expected.uncovered_sample_count


@pytest.mark.parametrize("observed", [False, True])
def test_empty_and_complete_masks(observed):
    values = np.ones((4, 1, 2, 1, 2), dtype=np.float32)
    mask = np.full(values.shape[1:], observed)
    result = interpolate_pocs_volume_batched(
        values,
        mask,
        n_iterations=2,
        threshold_start=1,
        threshold_end=0.1,
        window_shape=(8, 2, 2, 2, 2),
        overlap=(7, 1, 1, 1, 1),
    )
    np.testing.assert_array_equal(result.values, values if observed else np.zeros_like(values))
    assert result.uncovered_sample_count == (0 if observed else values.size)


def test_numpy_batches_match_reference_at_maximum_threshold():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(16, 2, 8, 2, 8)).astype(np.float32)
    mask = rng.random(values.shape[1:]) < 0.3
    kwargs = dict(
        n_iterations=4,
        threshold_start=1.0,
        threshold_end=0.8,
        window_shape=(12, 1, 6, 1, 6),
        overlap=(6, 0, 4, 0, 4),
    )
    expected = interpolate_pocs_volume(values, mask, **kwargs)
    for limit in (432, 10_000):
        actual = interpolate_pocs_volume_batched(values, mask, **kwargs, batch_sample_limit=limit)
        np.testing.assert_array_equal(actual.values, expected.values)
