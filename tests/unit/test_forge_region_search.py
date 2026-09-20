from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.processing.forge_region_search import (
    StationRegion,
    along_direction,
    compare_region_pair,
    fit_station_region,
    pair_cell_counts,
    pair_characteristics,
    rank_region_pairs,
    region_trace_mapping,
    search_station_regions,
    select_finalists,
)


@pytest.fixture
def config():
    repo = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo / "studies/study_051_forge_region_search/config.yaml").read_text())
    cfg["origin_xy_m"] = [0.0, 0.0]
    return cfg


def stations(n_line=3, n_point=8, angle=0.0):
    i, j = np.indices((n_line, n_point))
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    xy = j.ravel()[:, None] * 50 * direction + i.ravel()[:, None] * 200 * normal
    return pd.DataFrame(
        {
            "station_id": np.arange(n_line * n_point),
            "line": i.ravel(),
            "point": j.ravel(),
            "x_m": xy[:, 0],
            "y_m": xy[:, 1],
        }
    )


def fit(block, config, shape=(3, 4), center=None):
    direction = along_direction(block)
    if center is None:
        center = np.array([75, 200])
    result = fit_station_region(block, "source", shape, center, direction, config)
    assert result is not None
    result.metrics["candidate_id"] = "test"
    return result


def test_rotated_perfect_lattice_projection_and_id_independence(config):
    angle = np.radians(23)
    b = stations(angle=angle)
    d = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-d[1], d[0]])
    region = fit(b, config, center=75 * d + 200 * n)
    assert region.metrics["fill_fraction"] == 1
    assert region.metrics["across_spacing_m"] == pytest.approx(200)
    assert region.metrics["normalized_p95"] < 1e-12
    # Arbitrary numbering offsets on different lines must not move the grid.
    b.point += b.line * 100
    shifted = fit(b, config, center=75 * d + 200 * n)
    np.testing.assert_allclose(shifted.stations.grid_x_m, region.stations.grid_x_m)
    np.testing.assert_allclose(shifted.stations.grid_y_m, region.stations.grid_y_m)
    # Along-line numbering reversal is likewise an axial direction, not a reversal of geometry.
    b.point *= -1
    reversed_region = fit(b, config, center=75 * d + 200 * n)
    assert reversed_region.metrics["normalized_p95"] < 1e-12


def test_line_stagger_and_bending_are_not_removed_by_independent_shifts(config):
    b = stations()
    b.loc[b.line.eq(1), "x_m"] += 20
    region = fit(b, config)
    assert region.metrics["distance_p95_m"] > 10
    curved = stations()
    curved.loc[(curved.line == 1) & (curved.point == 2), "y_m"] += 80
    result = fit(curved, config)
    assert 10 in result.stations.station_id.to_list()
    assert result.metrics["distance_max_m"] > 50


def test_finite_grid_keeps_outlier_and_missing_point_decreases_fill(config):
    b = stations().drop(index=1)
    region = fit(b, config)
    assert region.metrics["fill_fraction"] == 11 / 12
    b = stations()
    b.loc[b.station_id == 0, "y_m"] = -500
    region = fit(b, config)
    row = region.stations.set_index("station_id").loc[0]
    assert row.edge_clamped
    assert row.distance_m > 400
    assert len(region.stations) == 12


def test_search_is_deterministic_and_unknown_coordinates_are_not_imputed(config):
    b = stations(n_line=4, n_point=12)
    b.loc[3, ["x_m", "y_m"]] = np.nan
    s = b.rename(
        columns={k: "source_" + k for k in ["line", "point", "x_m", "y_m"]}
        | {"station_id": "source_id"}
    )
    config["station_shortlist"] = 4
    a, selected = search_station_regions(s, "source", [3, 4], config)
    b, repeated = search_station_regions(s.sample(frac=1, random_state=9), "source", [3, 4], config)
    assert len(a) == len(b) > 0
    assert [x.metrics for x in selected] == [x.metrics for x in repeated]
    assert all(3 not in x.stations.station_id.to_list() for x in a)
    assert any(x.metrics["stage"] == "fine" for x in a)


def tiny_region(ids, cells, role):
    p = pd.DataFrame(
        {
            "station_id": ids,
            "cell": cells,
            "x_m": np.array(ids) * 50.0,
            "y_m": 0.0,
            "grid_line": 0,
            "grid_point": cells,
            "grid_x_m": np.array(cells) * 50.0,
            "grid_y_m": 0.0,
            "distance_m": 0.0,
            "normalized_distance": 0.0,
            "edge_clamped": False,
        }
    )
    return StationRegion(
        {
            "candidate_id": role,
            "cell_count": 3,
            "normalized_p95": 0.0,
            "distance_p95_m": 0.0,
            "fill_fraction": len(set(cells)) / 3,
        },
        p,
    )


def test_pair_occupancy_separates_collision_missing_and_qc_without_dropping_reviews(config):
    s = tiny_region([0, 1, 2], [0, 0, 2], "s")
    r = tiny_region([0, 1], [0, 2], "r")
    status = np.array([[3, 2], [4, 1], [0, 4]], dtype=np.uint8)
    _, _, count, recorded, support = pair_cell_counts(s, r, status)
    assert count.tolist() == [2, 0, 0, 0, 0, 0, 0, 0, 1]
    m = compare_region_pair(s, r, status, config, 4001)
    assert m["fill_fraction"] == 2 / 9
    assert m["collision_fraction"] == 2 / 3
    assert m["structural_empty_cells"] == 5
    assert m["unrecorded_supported_cells"] == 1
    assert m["qc_empty_cells"] == 1
    assert m["fixed_qc_excluded_traces"] == 1
    assert m["header_excluded_traces"] == 1
    assert m["review_fraction"] == 1 / 3
    assert m["dense_float32_bytes"] == 9 * 4001 * 4
    flags = {"dc_review": status == 3}
    assert pair_characteristics(s, r, status, flags)["dc_review_fraction"] == 1 / 3
    assert (count <= recorded).all() and (recorded <= support).all()


def test_ranking_ignores_quality_and_jaccard_uses_only_retained_records(config):
    s = tiny_region([0, 1], [0, 1], "s")
    r = tiny_region([0, 1], [0, 1], "r")
    rows = []
    status = np.full((2, 2), 4, dtype=np.uint8)
    for name in ["z", "a"]:
        row = compare_region_pair(s, r, status, config, 1)
        row.update(candidate_id=name, dc_review_fraction=0 if name == "z" else 1)
        rows.append(row)
    frame = pd.DataFrame(rows)
    assert rank_region_pairs(frame).candidate_id.tolist() == ["a", "z"]
    chosen = select_finalists(frame, {"s": s}, {"r": r}, status, config)
    assert chosen.candidate_id.tolist() == ["a"]


def test_mapping_retains_excluded_keys_and_uses_known_station_coordinates():
    s, r = tiny_region([0], [0], "s"), tiny_region([1], [2], "r")
    domain = pd.DataFrame(
        {
            "source_id": [0],
            "receiver_id": [1],
            "source_file": ["a"],
            "trace_index": [7],
            "source_x_m": [0.0],
            "source_y_m": [0.0],
            "receiver_x_m": [0.0],
            "receiver_y_m": [0.0],
            "eligible_after_fixed_qc": [False],
            "exclusion_reason": ["header_excluded"],
        }
    )
    mapping = region_trace_mapping(domain, s, r)
    assert len(mapping) == 1 and mapping.trace_index.iloc[0] == 7
    assert mapping.receiver_header_x_m.iloc[0] == 0
    assert mapping.receiver_x_m.iloc[0] == 50
    assert not mapping.eligible_after_fixed_qc.iloc[0]
    assert mapping.grid_cell.iloc[0] == 2
