"""Finite pure training summaries and full-interval throughput."""

import json
from copy import deepcopy

import pytest

from seis_interp.training.trace_graph_training_profile import (
    optimizer_updates_per_second,
    summarize_trace_graph_training,
)


def _history():
    return [
        {
            "seconds": step,
            "batch_preparation_seconds": step / 4,
            "query_geometry_seconds": step / 32,
            "graph_build_seconds": step / 8,
            "input_assembly_seconds": step / 16,
            "label_read_seconds": step / 32,
            "optimization_seconds": step / 2,
            "subgraph_node_count": 10 * step,
            "subgraph_support_node_count": 10 * step - 2,
            "subgraph_edge_count": 20 * step,
            "subgraph_max_depth": step,
            "loss": 0.5,
        }
        for step in (1, 3)
    ]


def test_breakdown_fields_partition_the_recorded_preparation_interval():
    for row in _history():
        parts = (
            "query_geometry_seconds",
            "graph_build_seconds",
            "input_assembly_seconds",
            "label_read_seconds",
        )
        assert sum(row[key] for key in parts) == row["batch_preparation_seconds"]


def test_means_maxima_and_throughput_are_json_numbers_without_mutation():
    history = _history()
    before = deepcopy(history)
    profile = summarize_trace_graph_training(history)
    assert profile == {
        "optimizer_updates": 2,
        "mean_recorded_step_seconds": 2.0,
        "mean_batch_preparation_seconds": 0.5,
        "mean_query_geometry_seconds": 0.0625,
        "mean_graph_build_seconds": 0.25,
        "mean_input_assembly_seconds": 0.125,
        "mean_label_read_seconds": 0.0625,
        "mean_optimization_seconds": 1.0,
        "mean_subgraph_node_count": 20.0,
        "max_subgraph_node_count": 30,
        "mean_subgraph_support_node_count": 18.0,
        "mean_subgraph_edge_count": 40.0,
        "max_subgraph_edge_count": 60,
        "mean_subgraph_max_depth": 2.0,
    }
    assert history == before
    assert all(type(value) in (int, float) for value in profile.values())
    assert json.loads(json.dumps(profile, allow_nan=False)) == profile
    assert optimizer_updates_per_second(2, 8) == 0.25


@pytest.mark.parametrize(
    "kind", ["empty", "missing", "nan", "inf", "negative", "boolean", "fractional_count"]
)
def test_bad_history_fails_clearly(kind):
    history = _history()
    if kind == "empty":
        history = []
    elif kind == "missing":
        del history[1]["optimization_seconds"]
    elif kind == "fractional_count":
        history[0]["subgraph_node_count"] = 1.5
    else:
        history[0]["seconds"] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -1,
            "boolean": True,
        }[kind]
    with pytest.raises(ValueError, match="training history"):
        summarize_trace_graph_training(history)


@pytest.mark.parametrize(
    "updates,seconds",
    [(0, 1), (True, 1), (1, 0), (1, -1), (1, float("nan")), (1, float("inf")), (2, 5e-324)],
)
def test_invalid_throughput_is_rejected(updates, seconds):
    with pytest.raises(ValueError):
        optimizer_updates_per_second(updates, seconds)
