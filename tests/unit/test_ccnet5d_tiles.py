from itertools import product

import numpy as np
import pytest

from seis_interp.processing.ccnet5d_tiles import iter_ccnet5d_tiles


@pytest.mark.parametrize("core", [(3, 2, 4, 2, 3), (20, 20, 20, 20, 20)])
@pytest.mark.parametrize("halo", [0, 2, 8])
def test_clipped_halos_cover_each_sample_exactly_once(core, halo) -> None:
    shape = (7, 3, 5, 2, 4)
    coverage = np.zeros(shape, dtype=int)
    values = np.arange(np.prod(shape)).reshape(shape)
    tiles = list(iter_ccnet5d_tiles(shape, core, halo_radius=halo))
    expected_starts = list(
        product(*(range(0, length, step) for length, step in zip(shape, core, strict=True)))
    )
    assert [tuple(axis.start for axis in tile.core_slices) for tile in tiles] == expected_starts
    for tile in tiles:
        coverage[tile.core_slices] += 1
        np.testing.assert_array_equal(
            values[tile.input_slices][tile.local_core_slices], values[tile.core_slices]
        )
        for length, output, source in zip(shape, tile.core_slices, tile.input_slices, strict=True):
            assert source.start == max(0, output.start - halo)
            assert source.stop == min(length, output.stop + halo)
    np.testing.assert_array_equal(coverage, 1)


@pytest.mark.parametrize("core", [(0, 1, 1, 1, 1), (1, 2), (True, 2, 3, 4, 5)])
def test_invalid_core_is_rejected(core) -> None:
    with pytest.raises(ValueError, match="core_shape"):
        list(iter_ccnet5d_tiles((5,) * 5, core, halo_radius=4))
