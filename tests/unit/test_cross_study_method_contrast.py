"""The cross-study pair contrast reuses the study_067 estimator and checks pairing."""

import numpy as np
import pandas as pd
import pytest

from seis_interp.evaluation.cross_study_method_contrast import (
    adjust_declared_families,
    paired_method_contrast,
)
from seis_interp.evaluation.forge_fair_grid_statistics import cluster_bootstrap_weights
from seis_interp.evaluation.study067_statistics import paired_seed_bootstrap

SEEDS = (7321, 7322, 7323)
CLUSTERS = np.arange(4)
WEIGHTS = cluster_bootstrap_weights(len(CLUSTERS), 200, 670703)
COMMON = {"cluster_column": "source_cell", "left": "inr_real", "right": "nersi_profile"}


def _table(snr, relative_mse, *, cells=range(8)):
    cells = list(cells)
    return pd.DataFrame(
        {
            "cell_id": cells,
            "trace_id": [f"t{cell}" for cell in cells],
            "source_cell": [cell % len(CLUSTERS) for cell in cells],
            "clean_target": True,
            "snr_db": np.asarray(snr, dtype=np.float64),
            "relative_mse": np.asarray(relative_mse, dtype=np.float64),
        }
    )


def _pair(left_snr, right_snr, left_rel=0.4, right_rel=0.9):
    rng = np.random.default_rng(5)
    left, right = {}, {}
    for seed in SEEDS:
        left[seed] = _table(left_snr + rng.normal(0, 0.05, 8), np.full(8, left_rel))
        right[seed] = _table(right_snr + rng.normal(0, 0.05, 8), np.full(8, right_rel))
    return left, right


def _contrast(**overrides):
    left, right = overrides.pop("tables", _pair(3.0, 1.0))
    return paired_method_contrast(
        left, right, seeds=SEEDS, clusters=CLUSTERS, weights=WEIGHTS, **COMMON, **overrides
    )


def test_the_orientation_follows_each_metric():
    table = _contrast().set_index("metric")
    snr, relative = table.loc["snr_db"], table.loc["relative_mse"]
    assert snr.higher_is_better and not relative.higher_is_better
    assert snr.estimate > 0 and snr.direction == "inr_real 優位"
    assert relative.estimate < 0 and relative.direction == "inr_real 優位"
    assert snr.contrast == "inr_real_minus_nersi_profile"
    assert snr.status == "available" and snr.cluster_count == len(CLUSTERS)


def test_a_worse_left_method_is_reported_as_such():
    table = _contrast(tables=_pair(1.0, 3.0, left_rel=1.2, right_rel=0.9)).set_index("metric")
    assert table.loc["snr_db"].direction == "nersi_profile 優位"
    assert table.loc["relative_mse"].direction == "nersi_profile 優位"


def test_the_estimator_matches_the_study067_bootstrap():
    left, right = _pair(3.0, 1.0)
    table = _contrast(tables=(left, right)).set_index("metric")
    differences = [left[seed].snr_db.to_numpy() - right[seed].snr_db.to_numpy() for seed in SEEDS]
    ids = [left[seed].source_cell.to_numpy() for seed in SEEDS]
    expected = paired_seed_bootstrap(differences, ids, CLUSTERS, WEIGHTS)
    for key, value in expected.items():
        assert table.loc["snr_db"][key] == pytest.approx(value)


def test_a_diverged_seed_leaves_the_row_unavailable_without_averaging_the_rest():
    left, right = _pair(3.0, 1.0)
    left[SEEDS[1]] = None
    table = _contrast(tables=(left, right)).set_index("metric")
    row = table.loc["snr_db"]
    assert row.status.startswith("unavailable")
    assert row.direction == "算出不能"
    assert np.isnan(row.estimate) and np.isnan(row.bootstrap_p_value)
    assert np.isnan(row.delta_seed_2) and row.delta_seed_1 == pytest.approx(2.0, abs=0.2)
    assert row.trace_count_seed_2 is None


def test_the_validation_free_mask_drops_the_declared_trace_identities():
    left, right = _pair(3.0, 1.0)
    full = _contrast(tables=(left, right)).set_index("metric")
    reduced = _contrast(tables=(left, right), validation_sets=({"t0", "t1", "t2"},)).set_index(
        "metric"
    )
    assert full.loc["snr_db"].trace_count_seed_1 == 8
    assert reduced.loc["snr_db"].trace_count_seed_1 == 5


@pytest.mark.parametrize("column", ["cell_id", "trace_id", "source_cell", "clean_target"])
def test_unpaired_tables_are_rejected(column):
    left, right = _pair(3.0, 1.0)
    seed = SEEDS[0]
    broken = right[seed].copy()
    if column == "cell_id":
        broken["cell_id"] = broken.cell_id + 100
    elif column == "clean_target":
        broken["clean_target"] = False
    else:
        broken[column] = broken[column].astype(object).iloc[::-1].to_numpy()
    right[seed] = broken
    with pytest.raises(ValueError, match="unpaired test cells|paired"):
        _contrast(tables=(left, right))


def test_holm_runs_inside_each_declared_family_and_leaves_the_rest_descriptive():
    rows = pd.concat(
        [
            _contrast().assign(dataset="c3"),
            _contrast().assign(dataset="forge"),
            _contrast(tables=_pair(2.0, 1.5)).assign(
                dataset="c3", contrast="inr_real_minus_pocs_grid"
            ),
        ],
        ignore_index=True,
    )
    families = {
        "output_unit_snr": {"contrast": "inr_real_minus_nersi_profile", "metric": "snr_db"},
        "output_unit_relmse": {
            "contrast": "inr_real_minus_nersi_profile",
            "metric": "relative_mse",
        },
    }
    adjusted = adjust_declared_families(rows, families)
    registered = adjusted.loc[adjusted.family.ne("")]
    assert set(registered.family) == set(families)
    assert registered.groupby("family").size().eq(2).all()
    assert registered.holm_status.eq("available").all()
    assert (registered.holm_p_value >= registered.bootstrap_p_value).all()
    descriptive = adjusted.loc[adjusted.family.eq("")]
    assert descriptive.contrast.eq("inr_real_minus_pocs_grid").all()
    assert descriptive.holm_p_value.isna().all()
    assert descriptive.holm_status.eq("descriptive: no multiplicity adjustment").all()


def test_an_unmatched_family_declaration_is_rejected():
    with pytest.raises(ValueError, match="no rows match"):
        adjust_declared_families(
            _contrast(), {"missing": {"contrast": "regsi_real_minus_inr_real", "metric": "snr_db"}}
        )
