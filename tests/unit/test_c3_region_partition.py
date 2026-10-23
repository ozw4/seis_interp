import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.c3_fair_grid_contract import c3_trace_manifest, make_c3_outer_mask
from seis_interp.processing.c3_region_partition import partition_c3_regions, region_scoring_mask
from tests.fixtures.c3_fair_grid import SPATIAL_SHAPE, tiny_c3_contract, tiny_c3_volume_index

SPECIFICATION = {
    "source_block_shape": [1, 2],
    "block_counts": {"train": 4, "validation": 2, "test": 2},
}


def _parent():
    config = tiny_c3_contract()
    manifest = c3_trace_manifest(tiny_c3_volume_index(), SPATIAL_SHAPE, config["candidate_id"])
    return make_c3_outer_mask(manifest, config)


def test_regions_keep_sources_profiles_and_outer_mask_atomic_and_reproducible():
    parent = _parent()
    first = partition_c3_regions(parent, SPATIAL_SHAPE, SPECIFICATION, 7311)
    repeated = partition_c3_regions(
        parent.sample(frac=1, random_state=3), SPATIAL_SHAPE, SPECIFICATION, 7311
    )
    pd.testing.assert_frame_equal(first, repeated)
    np.testing.assert_array_equal(first.outer_split, parent.split)
    assert first.region_role.value_counts().to_dict() == {
        "train": 128,
        "validation": 64,
        "test": 64,
    }
    assert first.groupby("source_cell").region_role.nunique().eq(1).all()
    assert first.groupby("block_id").region_role.nunique().eq(1).all()
    assert first.loc[first.input_observed, "region_role"].eq("train").all()
    assert first.loc[first.input_observed, "outer_split"].eq("observed").all()
    second = partition_c3_regions(parent, SPATIAL_SHAPE, SPECIFICATION, 7312)
    assert not first.region_role.equals(second.region_role)
    np.testing.assert_array_equal(first.outer_split, second.outer_split)


def test_scoring_domains_have_identical_train_observations_and_disjoint_targets():
    partition = partition_c3_regions(_parent(), SPATIAL_SHAPE, SPECIFICATION, 7311)
    validation = region_scoring_mask(partition, ("validation",))
    final = region_scoring_mask(partition, ("validation", "test"))
    assert validation.split.eq("test").sum() == 64
    assert final.split.eq("test").sum() == 128
    np.testing.assert_array_equal(validation.split.eq("observed"), final.split.eq("observed"))
    assert validation.loc[partition.region_role.eq("test"), "split"].eq("excluded").all()


def test_invalid_region_counts_and_partial_blocks_fail():
    with pytest.raises(ValueError, match="divide"):
        partition_c3_regions(
            _parent(), SPATIAL_SHAPE, {**SPECIFICATION, "source_block_shape": [1, 3]}, 1
        )
    with pytest.raises(ValueError, match="cover"):
        partition_c3_regions(
            _parent(),
            SPATIAL_SHAPE,
            {**SPECIFICATION, "block_counts": {"train": 3, "validation": 2, "test": 2}},
            1,
        )
