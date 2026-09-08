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
