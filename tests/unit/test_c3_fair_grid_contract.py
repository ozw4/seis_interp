import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from seis_interp.c3_fair_grid_config import C3_FAIR_GRID_METHODS, validate_c3_fair_grid_configs
from seis_interp.evaluation.fair_grid_trace_snr import (
    paired_trace_snr_contrasts,
    seed_averaged_trace_snr,
    trace_snr_by_trace,
)
from seis_interp.evaluation.forge_fair_grid_statistics import cluster_bootstrap_weights
from seis_interp.processing.c3_fair_grid_contract import (
    c3_cell_ids,
    c3_geometry_table,
    c3_trace_manifest,
    make_c3_outer_mask,
)
from seis_interp.processing.forge_mvp_contract import make_validation_holdout
from tests.fixtures.c3_fair_grid import (
    SPATIAL_SHAPE,
    tiny_c3_configs,
    tiny_c3_contract,
    tiny_c3_volume_index,
)

# Columns, order and dtypes of a FORGE ``make_geometry`` table.
GEOMETRY_COLUMNS = [
    "cell_id",
    "source_grid_line",
    "source_grid_point",
    "receiver_grid_line",
    "receiver_grid_point",
    "source_x_m",
    "source_y_m",
    "source_line_continuous",
    "source_point_continuous",
    "receiver_x_m",
    "receiver_y_m",
    "receiver_line_continuous",
    "receiver_point_continuous",
    "offset_x_m",
    "offset_y_m",
    "offset_m",
    "azimuth_rad",
    "midpoint_x_m",
    "midpoint_y_m",
]
MASK_COLUMNS = [
    "candidate_id",
    "trace_id",
    "cell_id",
    "source_cell",
    "receiver_cell",
    "clean_target",
    "stable_hash",
    "hash_rank",
    "split",
    "mask_id",
    "mask_seed",
]


def test_cell_ids_follow_the_volume_index_c_order():
    index = tiny_c3_volume_index()
    cells = c3_cell_ids(index, SPATIAL_SHAPE)
    np.testing.assert_array_equal(cells, np.arange(len(index)))
    indices = index.iloc[:, 2:6].to_numpy()
    np.testing.assert_array_equal(np.ravel_multi_index(indices.T, SPATIAL_SHAPE), cells)
    with pytest.raises(ValueError, match="C order"):
        c3_cell_ids(index.iloc[::-1].reset_index(drop=True), SPATIAL_SHAPE)
    with pytest.raises(ValueError, match="cover the spatial grid"):
        c3_cell_ids(index.iloc[:-1], SPATIAL_SHAPE)
    with pytest.raises(ValueError, match="cover the spatial grid"):
        c3_cell_ids(index, [2, 8, 2, 4])


def test_geometry_has_the_forge_columns_and_relative_receivers():
    index = tiny_c3_volume_index()
    geometry = c3_geometry_table(index, SPATIAL_SHAPE)
    assert list(geometry.columns) == GEOMETRY_COLUMNS
    assert all(str(geometry[c].dtype) == "int64" for c in GEOMETRY_COLUMNS[:5])
    assert all(str(geometry[c].dtype) == "float64" for c in GEOMETRY_COLUMNS[5:])
    for axis, column in zip(GEOMETRY_COLUMNS[1:5], index.columns[2:6], strict=True):
        np.testing.assert_array_equal(geometry[axis], index[column])
    np.testing.assert_array_equal(geometry.source_x_m, index.source_x_m)
    np.testing.assert_array_equal(geometry.source_y_m, index.source_y_m)
    np.testing.assert_array_equal(
        geometry.receiver_x_m, index.source_x_m + index.relative_receiver_x_m
    )
    np.testing.assert_array_equal(
        geometry.receiver_y_m, index.source_y_m + index.relative_receiver_y_m
    )
    np.testing.assert_array_equal(geometry.offset_x_m, index.relative_receiver_x_m)
    np.testing.assert_array_equal(geometry.offset_y_m, index.relative_receiver_y_m)
    np.testing.assert_allclose(
        geometry.offset_m, np.hypot(index.relative_receiver_x_m, index.relative_receiver_y_m)
    )
    # FORGE convention: clockwise from north of the receiver - source vector.
    np.testing.assert_allclose(
        geometry.azimuth_rad,
        np.arctan2(index.relative_receiver_x_m, index.relative_receiver_y_m),
    )
    np.testing.assert_allclose(
        geometry.midpoint_x_m, index.source_x_m + index.relative_receiver_x_m / 2
    )
    # Continuous coordinates are the integer indices (C3 v3 index coordinates).
    for role, (line, point) in (
        ("source", ("source_grid_line", "source_grid_point")),
        ("receiver", ("receiver_grid_line", "receiver_grid_point")),
    ):
        np.testing.assert_array_equal(geometry[f"{role}_line_continuous"], geometry[line])
        np.testing.assert_array_equal(geometry[f"{role}_point_continuous"], geometry[point])


def test_trace_manifest_maps_cells_to_interim_rows():
    index = tiny_c3_volume_index()
    manifest = c3_trace_manifest(index, SPATIAL_SHAPE, "tiny_c3")
    assert list(manifest.columns) == [
        "cell_id",
        "array_row",
        "ffid",
        "trace_id",
        "source_cell",
        "receiver_cell",
        "candidate_id",
    ]
    np.testing.assert_array_equal(manifest.array_row, index.array_row)
    assert manifest.trace_id.tolist() == [f"c3na:{row}" for row in index.array_row]
    receivers = SPATIAL_SHAPE[2] * SPATIAL_SHAPE[3]
    np.testing.assert_array_equal(
        manifest.source_cell,
        index.source_line_index * SPATIAL_SHAPE[1] + index.shot_in_line_index,
    )
    np.testing.assert_array_equal(
        manifest.source_cell * receivers + manifest.receiver_cell, manifest.cell_id
    )
    duplicated = index.assign(array_row=index.array_row.where(index.index != 1, index.array_row[0]))
    with pytest.raises(ValueError, match="duplicate array_row"):
        c3_trace_manifest(duplicated, SPATIAL_SHAPE, "tiny_c3")


def _independent_ranks(config, cells):
    keys = [
        hashlib.sha256(
            json.dumps(
                [
                    config["mask_study_id"],
                    config["candidate_id"],
                    config["mask_id"],
                    config["mask_seed"],
                    int(c),
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for c in cells
    ]
    order = sorted(range(len(cells)), key=lambda i: (keys[i], cells[i]))
    ranks = np.empty(len(cells), dtype=np.int64)
    ranks[order] = np.arange(1, len(cells) + 1)
    return ranks


def test_outer_mask_is_an_exact_count_hash_rank_independent_of_row_order():
    config = tiny_c3_contract()
    manifest = c3_trace_manifest(tiny_c3_volume_index(), SPATIAL_SHAPE, config["candidate_id"])
    mask = make_c3_outer_mask(manifest, config)
    assert list(mask.columns) == MASK_COLUMNS
    assert mask.split.value_counts().to_dict() == {"test": 205, "observed": 51}
    assert mask.clean_target.all() and mask.clean_target.dtype == bool
    ranks = _independent_ranks(config, mask.cell_id.tolist())
    np.testing.assert_array_equal(mask.hash_rank, ranks)
    np.testing.assert_array_equal(mask.split.eq("test"), ranks <= config["test_trace_count"])
    shuffled = make_c3_outer_mask(manifest.sample(frac=1, random_state=1), config)
    pd.testing.assert_frame_equal(mask, shuffled)
    # The namespace is the mask study, not the study that uses the mask.
    other = make_c3_outer_mask(manifest, {**config, "study_id": "elsewhere"})
    pd.testing.assert_frame_equal(mask, other)
    moved = make_c3_outer_mask(manifest, {**config, "mask_study_id": "elsewhere"})
    assert not moved.split.equals(mask.split)
    for change, message in (
        ({"test_trace_count": 204}, "outer observed count"),
        ({"eligible_trace_count": 255}, "population/count"),
        ({"candidate_id": "other"}, "candidate"),
    ):
        with pytest.raises(ValueError, match=message):
            make_c3_outer_mask(manifest, {**config, **change})


def test_forge_validation_holdout_splits_the_c3_mask():
    config = tiny_c3_contract()
    manifest = c3_trace_manifest(tiny_c3_volume_index(), SPATIAL_SHAPE, config["candidate_id"])
    parent = make_c3_outer_mask(manifest, config)
    holdout = make_validation_holdout(parent, config)
    assert holdout.split.value_counts().to_dict() == {"excluded": 205, "observed": 41, "test": 10}
    assert set(holdout.cell_id[holdout.split.ne("excluded")]) == set(
        parent.cell_id[parent.split.eq("observed")]
    )
    assert holdout.clean_target.all()


@pytest.fixture
def valid():
    return tiny_c3_configs(), tiny_c3_contract()


def test_the_tiny_five_method_study_is_valid(valid):
    configs, contract = valid
    assert tuple(configs) == C3_FAIR_GRID_METHODS
    validate_c3_fair_grid_configs(configs, contract)


def _set(body, dotted, value):
    *path, key = dotted.split(".")
    for part in path:
        body = body[part]
    if value is _DELETE:
        del body[key]
    else:
        body[key] = value


_DELETE = object()


@pytest.mark.parametrize(
    "target,dotted,value,message",
    [
        ("contract", "dataset_id", _DELETE, "dataset_id"),
        ("contract", "preprocessing.amplitude_scaling", "trace_oracle", "observed global RMS"),
        ("contract", "preprocessing.trace_rms", True, "observed global RMS"),
        ("contract", "preprocessing.alignment", {"velocity_m_per_s": 500.0}, "no moveout"),
        ("contract", "time_sampling", {"decimation_factor": 1}, "no time_sampling"),
        ("contract", "matched_training", _DELETE, "matched_training"),
        ("contract", "eligible_trace_count", 255, "dense"),
        ("pocs_grid", "alignment", {"velocity_m_per_s": 500.0}, "shared preprocessing"),
        ("ccnet5d_grid", "geometry_mode", "real", "geometry mismatch"),
        ("regsi_real", "geometry_mode", "grid", "geometry mismatch"),
        ("drr_grid", "algorithm.frequency_min_hz", 5.0, "frequency filtering"),
        ("pocs_grid", "model_seed", 1, "deterministic"),
        ("ccnet5d_grid", "patches.shape", [16, 2, 8, 2, 8], "complete traces"),
        ("ccnet5d_grid", "training.supervised_traces_per_update", 3, "exactly the budgeted"),
        ("regsi_real", "training.optimizer", "adam", "optimizer"),
        ("regsi_real", "training.query_batch_size", 3, "budgeted traces"),
        ("regsi_real", "model_seed", 1, "model seed"),
        ("nersi_profile", "model.profile_shape", [16, 8], "complete traces"),
        ("nersi_profile", "training.max_steps", 5, "match the budget"),
        ("nersi_profile", "geometry_mode", "grid", "geometry mismatch"),
    ],
)
def test_config_validation_rejects_contract_drift(valid, target, dotted, value, message):
    configs, contract = valid
    _set(contract if target == "contract" else configs[target], dotted, value)
    with pytest.raises(ValueError, match=message):
        validate_c3_fair_grid_configs(configs, contract)


def test_config_validation_requires_exactly_the_five_methods(valid):
    configs, contract = valid
    with pytest.raises(ValueError, match="exactly"):
        validate_c3_fair_grid_configs({**configs, "nersi_real": {}}, contract)
    del configs["drr_grid"]
    with pytest.raises(ValueError, match="exactly"):
        validate_c3_fair_grid_configs(configs, contract)


def test_trace_snr_contrasts_keep_the_dB_orientation():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(6, 16))
    good = target + 0.1 * rng.normal(size=target.shape)
    poor = target + 0.5 * rng.normal(size=target.shape)
    snr = {name: trace_snr_by_trace(target, values) for name, values in (("a", good), ("b", poor))}
    expected = 10 * np.log10((target**2).sum(1) / ((good - target) ** 2).sum(1))
    np.testing.assert_allclose(snr["a"], expected)
    clusters = np.array([0, 0, 1, 1, 2, 2])
    weights = cluster_bootstrap_weights(3, 50, 1)
    table = paired_trace_snr_contrasts(snr, clusters, weights, [("a_vs_b", "a", "b", "role")])
    row = table.iloc[0]
    assert row.estimate == pytest.approx((snr["a"] - snr["b"]).mean()) and row.estimate > 0
    assert row.left_mean_trace_snr_db == pytest.approx(snr["a"].mean())
    assert row.fraction_left_higher_snr == pytest.approx(np.mean(snr["a"] > snr["b"]))
    assert "fraction_left_lower_error" not in table and "left_mean_error" not in table
    # A perfect trace (+inf dB) is counted upstream and rejected for seed means.
    perfect = trace_snr_by_trace(target, target)
    assert np.isposinf(perfect).all()
    with pytest.raises(ValueError, match="6 non-finite"):
        seed_averaged_trace_snr([perfect, snr["a"]])
    np.testing.assert_allclose(
        seed_averaged_trace_snr([snr["a"], snr["b"]]), (snr["a"] + snr["b"]) / 2
    )
