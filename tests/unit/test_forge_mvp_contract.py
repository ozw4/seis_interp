import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.forge_mvp_contract import (
    make_geometry,
    make_outer_mask,
    validate_mapping,
    validate_regsi_pair,
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
