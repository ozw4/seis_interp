"""Reject unsupported experiment settings and frozen checkpoint overrides."""

from __future__ import annotations

import pytest

from seis_interp.configuration import ConfigurationError
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_prediction_config,
    validate_relational_trace_graph_training_config,
)
from tests.fixtures.relational_trace_graph_runs import (
    trace_graph_prediction_config,
    trace_graph_training_config,
)


def test_training_config_preserves_model_graph_and_separate_seed_contracts():
    config = trace_graph_training_config()
    model, graph, training = validate_relational_trace_graph_training_config(config)
    assert model["message_passing_rounds"] == 2
    assert graph.relation_scales_m == ((2000.0, 5000.0), (5000.0, 2000.0)) * 2
    assert training["random_seed"] == 7 and config["project"]["random_seed"] == 42
    assert training["episode_kind_probabilities"] == {"random_trace": 0.5, "random_whole_ffid": 0.5}


@pytest.mark.parametrize("threshold", [10000, 10000.0])
def test_training_physical_bound_is_optional_without_changing_model_or_trainer_options(threshold):
    config = trace_graph_training_config()
    expected = validate_relational_trace_graph_training_config(config)
    config["training_data"]["max_abs_amplitude"] = threshold

    assert validate_relational_trace_graph_training_config(config) == expected
    assert config["training_data"]["max_abs_amplitude"] == threshold


@pytest.mark.parametrize("value", [None, True, False, "10000", 0, -1, float("nan"), float("inf")])
def test_training_physical_bound_rejects_invalid_values(value):
    config = trace_graph_training_config()
    config["training_data"]["max_abs_amplitude"] = value
    with pytest.raises(ConfigurationError, match="training_data.max_abs_amplitude"):
        validate_relational_trace_graph_training_config(config)


def test_training_physical_bound_does_not_allow_other_training_data_options():
    config = trace_graph_training_config()
    config["training_data"].update(max_abs_amplitude=10000.0, clip=True)
    with pytest.raises(ConfigurationError, match="training_data"):
        validate_relational_trace_graph_training_config(config)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "name", "trace_graph"),
        ("model", "relation_fusion", "unknown"),
        ("model", "node_feature_names", ["observed"]),
        ("model", "message_passing_rounds", 4),
        ("graph", "rounds", 2),
        ("graph", "neighbors_per_relation", 0),
        ("geometry_features", "position_scale_m", 0),
        ("training_data", "pool", "test"),
        ("training_data", "time_samples", [2, 2]),
        ("training", "loss", "spectrum"),
        ("training", "random_seed", True),
        ("training_mask", "kinds", ["unknown"]),
        ("training_mask", "kind_probabilities", [1.0, 1.0]),
        ("training_mask", "missing_fractions", [1.0]),
        ("evaluation", "domain", "observed"),
    ],
)
def test_training_config_rejects_unknown_or_inconsistent_conditions(section, key, value):
    config = trace_graph_training_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_relational_trace_graph_training_config(config)


def test_graph_config_rejects_unknown_relation_and_scale_key():
    config = trace_graph_training_config()
    config["graph"]["relations"]["azimuth"] = config["graph"]["relations"].pop("source")
    with pytest.raises(ConfigurationError, match="relations"):
        validate_relational_trace_graph_training_config(config)
    config = trace_graph_training_config()
    config["graph"]["relations"]["receiver"]["offset_scale_m"] = 10
    with pytest.raises(ConfigurationError, match="receiver"):
        validate_relational_trace_graph_training_config(config)


@pytest.mark.parametrize(
    "override",
    [
        "model",
        "graph",
        "geometry_features",
        "amplitude",
        "training",
        "training_data",
        "training_mask",
    ],
)
def test_frozen_config_rejects_checkpoint_setting_overrides(override):
    config = trace_graph_prediction_config()
    config[override] = {}
    with pytest.raises(ConfigurationError, match="unsupported"):
        validate_relational_trace_graph_prediction_config(config)


def test_frozen_config_accepts_runtime_batch_and_target_case_only():
    validate_relational_trace_graph_prediction_config(trace_graph_prediction_config())


@pytest.mark.parametrize("variant", ["plain_gcn_row_normalized", "untyped_edge_conditioned"])
def test_comparison_config_records_model_topology_and_fixed_geometry(variant):
    config = trace_graph_training_config()
    config["model"].update(
        method_variant=variant, relation_fusion="mean", explicit_azimuth_features=False
    )
    config["graph"]["common_distance_scales_m"] = [1000.0, 1000.0]
    model, graph, _ = validate_relational_trace_graph_training_config(config)
    assert model["method_variant"] == variant
    assert model["explicit_azimuth_features"] is False
    assert graph.common_distance_scales_m == (1000.0, 1000.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"model": {"method_variant": "untyped_edge_conditioned"}},
        {"model": {"method_variant": "unsupported"}},
        {"model": {"explicit_azimuth_features": 0}},
        {"graph": {"topology": "single_4d", "common_distance_scales_m": [1000, 1000]}},
        {"graph": {"excluded_relation": "missing"}},
        {"model": {"method_variant": "untyped_edge_conditioned", "relation_fusion": "mean"}},
    ],
)
def test_impossible_variant_combinations_are_rejected(changes):
    config = trace_graph_training_config()
    for section, values in changes.items():
        config[section].update(values)
    with pytest.raises(ValueError):
        validate_relational_trace_graph_training_config(config)


def test_diagnostic_bands_are_explicit_fixed_config_and_optional():
    config = trace_graph_training_config()
    config["diagnostics"] = {
        "time_s": [0.1, 0.2],
        "offset_m": [100, 200],
        "azimuth_deg": [90, 180, 270],
    }
    _, _, options = validate_relational_trace_graph_training_config(config)
    assert tuple(options["diagnostic_bands"].time_s) == (0.1, 0.2)
    frozen = trace_graph_prediction_config()
    frozen["diagnostics"] = config["diagnostics"]
    validate_relational_trace_graph_prediction_config(frozen)
    config["diagnostics"]["offset_m"] = [200, 100]
    with pytest.raises(ValueError):
        validate_relational_trace_graph_training_config(config)
