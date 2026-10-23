from itertools import product

import numpy as np
import pytest

from seis_interp.processing.drr import interpolate_drr_block
from seis_interp.processing.drr_nonoverlapping import interpolate_drr_nonoverlapping


def test_padded_nonoverlap_matches_independent_blocks_and_preserves_observations():
    rng = np.random.default_rng(4)
    shape = (3, 5, 2, 5)
    window = (2, 3, 2, 3)
    values = rng.normal(size=(7, *shape)).astype(np.float32)
    mask = rng.random(shape) < 0.6
    values[:, ~mask] = np.nan
    time = np.arange(7) * 0.008
    options = dict(
        rank=1, damping_power=2, n_iterations=2, frequency_min_hz=0.0, frequency_max_hz=None
    )
    result = interpolate_drr_nonoverlapping(
        values,
        mask,
        time,
        **options,
        spatial_window_shape=window,
        spatial_overlap=(0, 0, 0, 0),
        batch_matrix_limit=1,
        workers=2,
    )
    expected = np.zeros_like(values)
    coverage = np.zeros(shape, dtype=int)
    for location in product(*(range(0, n, w) for n, w in zip(shape, window, strict=True))):
        lengths = tuple(min(w, n - s) for n, w, s in zip(shape, window, location, strict=True))
        region = tuple(slice(s, s + n) for s, n in zip(location, lengths, strict=True))
        inside = tuple(slice(0, n) for n in lengths)
        block = np.zeros((7, *window), dtype=np.float32)
        visible = np.zeros(window, dtype=bool)
        block[(slice(None), *inside)] = np.where(
            mask[region][None], values[(slice(None), *region)], 0
        )
        visible[inside] = mask[region]
        predicted = interpolate_drr_block(block, visible, time, **options)
        expected[(slice(None), *region)] = predicted[(slice(None), *inside)]
        coverage[region] += 1
    np.testing.assert_allclose(result.values, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(result.values[:, mask], values[:, mask])
    assert np.all(coverage == 1)
    assert result.block_count == 8
    assert np.isfinite(result.values).all()


def test_nonzero_overlap_is_rejected():
    with pytest.raises(ValueError, match="zero overlap"):
        interpolate_drr_nonoverlapping(
            np.zeros((4, 2, 2, 2, 2), dtype=np.float32),
            np.ones((2, 2, 2, 2), dtype=bool),
            np.arange(4) * 0.008,
            rank=1,
            damping_power=2,
            n_iterations=1,
            frequency_min_hz=0,
            frequency_max_hz=None,
            spatial_window_shape=(2, 2, 2, 2),
            spatial_overlap=(1, 0, 0, 0),
        )
