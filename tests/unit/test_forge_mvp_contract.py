import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.forge_mvp_contract import (
    make_geometry,
    make_outer_mask,
    make_validation_holdout,
    validate_mapping,
    validate_regsi_pair,
    validation_contract,
    validation_holdout_counts,
)
from tests.fixtures.forge_mvp import tiny_forge_study


def test_mask_is_exact_hash_order_and_row_order_independent(tmp_path):
    _, config, _, _, mapping, stations, cells, grids, _ = tiny_forge_study(tmp_path)
    manifest = validate_mapping(mapping, stations, cells, grids, config)
    mask = make_outer_mask(manifest, config)
    pd.testing.assert_frame_equal(mask, make_outer_mask(manifest.iloc[::-1], config))
    assert mask.split.value_counts().to_dict() == {"test": 28, "observed": 7}
    ranks = []
    for cell in manifest.cell_id:
        key = [config[k] for k in ("study_id", "candidate_id", "mask_id", "mask_seed")] + [
            int(cell)
        ]
        ranks.append(
            (hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest(), cell)
        )
    assert set(mask.loc[mask.split.eq("test"), "cell_id"]) == {r[1] for r in sorted(ranks)[:28]}
    assert 35 not in set(mask.cell_id)


def test_lower_missing_fraction_preserves_hash_order_and_nested_test_cells(tmp_path):
    _, config, _, _, mapping, stations, cells, grids, _ = tiny_forge_study(tmp_path)
    manifest = validate_mapping(mapping, stations, cells, grids, config)
    original = make_outer_mask(manifest, config)
    smaller_config = {
        **config,
        "missing_fraction": 0.7,
        "test_trace_count": round(len(manifest) * 0.7),
    }
    smaller_config["observed_trace_count"] = len(manifest) - smaller_config["test_trace_count"]
    smaller = make_outer_mask(manifest, smaller_config)
    pd.testing.assert_frame_equal(smaller.drop(columns="split"), original.drop(columns="split"))
    assert int(smaller.split.eq("test").sum()) == smaller_config["test_trace_count"]
    assert set(smaller.loc[smaller.split.eq("test"), "cell_id"]) < set(
        original.loc[original.split.eq("test"), "cell_id"]
    )
    pd.testing.assert_frame_equal(smaller, make_outer_mask(manifest.iloc[::-1], smaller_config))


@pytest.mark.parametrize("fault", ["collision", "counts", "axis", "coordinates", "distance"])
def test_mapping_rejects_integrity_faults(tmp_path, fault):
    _, config, _, _, mapping, stations, cells, grids, _ = tiny_forge_study(tmp_path)
    if fault == "collision":
        mapping.loc[1, "grid_cell"] = 0
    if fault == "counts":
        cells.loc[0, "retained_count"] = 0
    if fault == "axis":
        config["axis_order"] = config["axis_order"][::-1]
    if fault == "coordinates":
        mapping.loc[0, "source_x_m"] += 1
    if fault == "distance":
        mapping.loc[0, "receiver_normalized_distance"] += 1
    with pytest.raises(ValueError):
        validate_mapping(mapping, stations, cells, grids, config)


def test_grid_geometry_has_no_dependency_on_actual_coordinates(tmp_path):
    _, config, _, _, mapping, stations, cells, grids, _ = tiny_forge_study(tmp_path)
    manifest = validate_mapping(mapping, stations, cells, grids, config)
    real, grid = make_geometry(manifest, grids, "real"), make_geometry(manifest, grids, "grid")
    corrupted = manifest.drop(columns=["source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m"])
    pd.testing.assert_frame_equal(grid, make_geometry(corrupted, grids, "grid"))
    assert not real.equals(grid)
    assert np.any(real.source_point_continuous % 1 != 0)
    assert np.all(grid.source_point_continuous % 1 == 0)
    delta = (
        grid[["receiver_x_m", "receiver_y_m"]].to_numpy()
        - grid[["source_x_m", "source_y_m"]].to_numpy()
    )
    np.testing.assert_allclose(grid.offset_m, np.linalg.norm(delta, axis=1))


def test_regsi_pair_rejects_non_geometry_changes(tmp_path):
    configs = tiny_forge_study(tmp_path)[3]
    validate_regsi_pair(configs)
    changed = copy.deepcopy(configs)
    changed["regsi_grid"]["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="beyond geometry"):
        validate_regsi_pair(changed)


def test_regsi_pair_may_run_on_different_devices(tmp_path):
    configs = tiny_forge_study(tmp_path)[3]
    configs["regsi_real"]["device"] = "cuda:1"
    configs["regsi_grid"]["device"] = "cuda:0"
    validate_regsi_pair(configs)


def _holdout_config(config, **holdout):
    return {
        **config,
        "validation_holdout": {
            "fraction": 0.3,
            "mask_id": "tiny_validation",
            "mask_seed": 11,
            **holdout,
        },
    }


def test_validation_holdout_splits_outer_observed_cells_by_hash_rank(tmp_path):
    _, config, _, _, mapping, stations, cells, grids, _ = tiny_forge_study(tmp_path)
    manifest = validate_mapping(mapping, stations, cells, grids, config)
    config = _holdout_config(config)
    parent = make_outer_mask(manifest, config)
    mask = make_validation_holdout(parent, config)
    # round(0.3 x 7) = 2 validation cells; the 28 outer test cells are excluded.
    assert mask.split.value_counts().to_dict() == {"excluded": 28, "observed": 5, "test": 2}
    assert validation_holdout_counts(config) == {
        "observed_trace_count": 5,
        "test_trace_count": 2,
        "excluded_trace_count": 28,
    }
    pd.testing.assert_frame_equal(
        mask, make_validation_holdout(parent.sample(frac=1, random_state=3), config)
    )
    pd.testing.assert_frame_equal(
        mask.drop(
            columns=[
                "split",
                "mask_id",
                "mask_seed",
                "parent_split",
                "parent_mask_id",
                "parent_mask_seed",
                "validation_stable_hash",
                "validation_hash_rank",
            ]
        ),
        parent.drop(columns=["split", "mask_id", "mask_seed"]),
    )
    assert mask.parent_split.tolist() == parent.split.tolist()
    assert set(mask.loc[mask.split.eq("excluded"), "cell_id"]) == set(
        parent.loc[parent.split.eq("test"), "cell_id"]
    )
    assert mask.mask_id.eq("tiny_validation").all() and mask.mask_seed.eq(11).all()
    assert mask.parent_mask_id.eq(config["mask_id"]).all()
    observed = parent.loc[parent.split.eq("observed"), "cell_id"]
    ranks = sorted(
        (
            hashlib.sha256(
                json.dumps(
                    [config["study_id"], config["candidate_id"], "tiny_validation", 11, int(cell)],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            int(cell),
        )
        for cell in observed
    )
    assert set(mask.loc[mask.split.eq("test"), "cell_id"]) == {cell for _, cell in ranks[:2]}
    assert mask.loc[mask.split.eq("excluded"), "validation_hash_rank"].eq(0).all()
    assert mask.loc[mask.split.eq("excluded"), "validation_stable_hash"].isna().all()
    assert sorted(mask.loc[mask.parent_split.eq("observed"), "validation_hash_rank"]) == list(
        range(1, 8)
    )
    other = make_validation_holdout(parent, _holdout_config(config, mask_seed=12))
    assert other.split.value_counts().to_dict() == mask.split.value_counts().to_dict()
    assert other.loc[other.split.eq("excluded"), "cell_id"].tolist() == (
        mask.loc[mask.split.eq("excluded"), "cell_id"].tolist()
    )


def test_validation_contract_carries_holdout_counts_and_parent_identity(tmp_path):
    _, config, _, _, _, _, _, _, _ = tiny_forge_study(tmp_path)
    contract = validation_contract(_holdout_config(config))
    assert (contract["observed_trace_count"], contract["test_trace_count"]) == (5, 2)
    assert contract["excluded_trace_count"] == 28
    assert (contract["mask_id"], contract["mask_seed"]) == ("tiny_validation", 11)
    assert contract["selection_domain"] == "validation_holdout"
    assert contract["parent_mask"] == {
        "study_id": config["study_id"],
        "mask_id": config["mask_id"],
        "mask_seed": config["mask_seed"],
        "observed_trace_count": 7,
        "test_trace_count": 28,
    }


@pytest.mark.parametrize(
    "holdout",
    [
        {"fraction": 0.0},
        {"fraction": 1.0},
        {"fraction": 0.01},
        {"fraction": True},
        {"mask_seed": "11"},
        {"mask_id": "m1_random80_mvp_v1"},
        {"extra": 1},
    ],
)
def test_validation_holdout_rejects_invalid_settings(tmp_path, holdout):
    _, config, _, _, _, _, _, _, _ = tiny_forge_study(tmp_path)
    with pytest.raises(ValueError, match="validation"):
        validation_holdout_counts(_holdout_config(config, **holdout))
    with pytest.raises(ValueError, match="validation_holdout requires"):
        validation_holdout_counts(config)
