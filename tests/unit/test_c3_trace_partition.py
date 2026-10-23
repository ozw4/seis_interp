import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.c3_fair_grid_contract import c3_trace_manifest, make_c3_outer_mask
from seis_interp.processing.c3_region_partition import region_scoring_mask
from seis_interp.processing.c3_region_search import c3_split_settings, validate_c3_experiment_plan
from seis_interp.processing.c3_trace_partition import partition_c3_traces
from seis_interp.processing.forge_mvp_contract import make_validation_holdout
from tests.fixtures.c3_fair_grid import SPATIAL_SHAPE, tiny_c3_contract, tiny_c3_volume_index


def _parent():
    contract = tiny_c3_contract()
    manifest = c3_trace_manifest(tiny_c3_volume_index(), SPATIAL_SHAPE, contract["candidate_id"])
    return make_c3_outer_mask(manifest, contract)


def _contract():
    contract = tiny_c3_contract()
    contract["validation_holdout"]["fraction"] = 0.1
    contract["validation_holdout"]["mask_seed"] = 21
    return contract


def test_trace_roles_keep_observed_holdout_separate_from_original_test_across_the_crop():
    parent = _parent()
    partition = partition_c3_traces(parent, _contract(), 7311)
    assert partition.trace_role.value_counts().to_dict() == {
        "train": 46,
        "validation": 5,
        "test": 205,
    }
    assert partition.loc[partition.input_observed, "outer_split"].eq("observed").all()
    assert partition.loc[partition.trace_role.eq("validation"), "outer_split"].eq("observed").all()
    np.testing.assert_array_equal(partition.trace_role.eq("test"), parent.split.eq("test"))
    assert partition.loc[partition.input_observed, "trace_role"].eq("train").all()
    assert not partition.loc[partition.trace_role.isin(("validation", "test"))].input_observed.any()
    assert partition.groupby("source_cell").trace_role.nunique().eq(3).any()
    assert "block_id" not in partition and "region_role" not in partition
    scoring = region_scoring_mask(partition, ("validation",), "trace_role")
    assert scoring.split.eq("observed").sum() == 46
    assert scoring.split.eq("test").sum() == 5
    assert scoring.loc[partition.trace_role.eq("test"), "split"].eq("excluded").all()
    final = region_scoring_mask(partition, ("validation", "test"), "trace_role")
    assert final.split.eq("test").sum() == 210


def test_assignment_uses_only_observed_cells_and_is_independent_of_row_order():
    parent = _parent()
    contract = _contract()
    partition = partition_c3_traces(parent, contract, 7311)
    repeated = partition_c3_traces(parent.sample(frac=1, random_state=9), contract, 7311)
    pd.testing.assert_frame_equal(partition, repeated)
    ranked = sorted(
        parent.loc[parent.split.eq("observed"), "cell_id"],
        key=lambda cell: (
            hashlib.sha256(
                json.dumps(
                    [
                        contract["study_id"],
                        contract["candidate_id"],
                        contract["validation_holdout"]["mask_id"],
                        7311,
                        int(cell),
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            int(cell),
        ),
    )
    assert set(partition.loc[partition.trace_role.eq("validation"), "cell_id"]) == set(ranked[:5])
    changed = partition_c3_traces(parent, contract, 7312)
    assert not partition.trace_role.equals(changed.trace_role)
    np.testing.assert_array_equal(partition.trace_role.eq("test"), changed.trace_role.eq("test"))


def test_search_mask_is_exactly_the_study_060_validation_holdout_rule():
    parent, contract = _parent(), _contract()
    partition = partition_c3_traces(parent, contract, 21)
    expected = make_validation_holdout(parent, contract)
    actual = region_scoring_mask(partition, ("validation",), "trace_role")
    pd.testing.assert_frame_equal(actual[expected.columns], expected)


@pytest.mark.parametrize("fraction", (-0.1, 0, 1, 2))
def test_invalid_validation_fractions_are_rejected(fraction):
    contract = _contract()
    contract["validation_holdout"]["fraction"] = fraction
    with pytest.raises(ValueError, match="strictly between"):
        partition_c3_traces(_parent(), contract, 7311)


def test_incomplete_crop_is_rejected():
    with pytest.raises(ValueError, match="complete C-ordered"):
        partition_c3_traces(_parent().iloc[1:], _contract(), 7311)


def test_experiment_requires_one_explicit_assignment_and_three_additional_seeds():
    from seis_interp.c3_fair_grid_config import C3_FAIR_GRID_METHODS

    contract = _contract()
    contract["trace_split"] = {
        "search_seed": 21,
        "evaluation_seeds": [7311, 7312, 7313],
    }
    grid = {"methods": dict.fromkeys(C3_FAIR_GRID_METHODS)}
    validate_c3_experiment_plan(contract, grid)
    assert c3_split_settings(contract) == ("trace", contract["trace_split"])
    contract["region_split"] = {}
    with pytest.raises(ValueError, match="exactly one"):
        validate_c3_experiment_plan(contract, grid)
    del contract["region_split"]
    contract["trace_split"]["evaluation_seeds"] = [21, 7311, 7312]
    with pytest.raises(ValueError, match="three distinct"):
        validate_c3_experiment_plan(contract, grid)
    contract["trace_split"]["evaluation_seeds"] = [7311, 7312, 7313]
    contract["validation_holdout"]["mask_seed"] = 22
    with pytest.raises(ValueError, match="must match"):
        validate_c3_experiment_plan(contract, grid)
