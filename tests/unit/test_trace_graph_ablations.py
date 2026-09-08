"""Independent geometry, feature and graph controls for trace comparisons."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import RELATION_NAMES, compute_trace_graph_geometry
from seis_interp.processing.trace_graph_neighbors import select_trace_graph_neighbors
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph
from tests.fixtures.relational_trace_graph import (
    make_relational_trace_domains,
    make_relational_trace_plan,
)


def _geometry(source, receiver):
    return compute_trace_graph_geometry(
        np.asarray(source, dtype=np.float64),
        np.asarray(receiver, dtype=np.float64),
        azimuth_min_offset_m=0.01,
    )


def _hash(tensor):
    return sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def test_single_4d_neighbors_match_equivalent_source_receiver_distance_order():
    source = np.array([[0.4, 0.1], [1.0, 0.3], [0.2, -0.1], [-0.3, 0.0], [0.7, 0.9]])
    receiver = np.array([[0.2, -0.3], [-0.4, 0.1], [0.3, 0.3], [-0.9, -0.4], [0.1, 0.4]])
    query_source, query_receiver = np.array([0.1, 0.2]), np.array([0.3, -0.5])
    query = _geometry(query_source[None], query_receiver[None])
    candidates = _geometry(source, receiver)
    ids = np.array([5, 9, 3, 2, 8])
    scale = 2.0
    common = (scale / np.sqrt(2), scale * np.sqrt(2))
    expected_d = np.sqrt(
        (
            np.sum((source - query_source) ** 2, axis=1)
            + np.sum((receiver - query_receiver) ** 2, axis=1)
        )
        / scale**2
    )
    midpoint_d = candidates.midpoint_xy_m - query.midpoint_xy_m
    offset_d = candidates.offset_xy_m - query.offset_xy_m
    np.testing.assert_allclose(
        expected_d**2,
        np.sum(midpoint_d**2, axis=1) / common[0] ** 2
        + np.sum(offset_d**2, axis=1) / common[1] ** 2,
        atol=1e-15,
    )
    observed = np.array([True, False, True, True, True])
    expected_rows = np.flatnonzero(observed & (expected_d <= 0.6))
    expected_rows = expected_rows[np.lexsort((ids[expected_rows], expected_d[expected_rows]))[:3]]
    for chunk_size in (1, 3, 100):
        result = select_trace_graph_neighbors(
            query,
            np.array([100]),
            candidates,
            ids,
            observed,
            relation_scales_m=np.ones((4, 2)),
            topology="single_4d",
            common_distance_scales_m=common,
            single_4d_neighbors=3,
            radius=0.6,
            candidate_chunk_size=chunk_size,
        )
        np.testing.assert_array_equal(result.sender_ids, ids[expected_rows])
        np.testing.assert_allclose(result.distances, expected_d[expected_rows], atol=1e-15)
        np.testing.assert_array_equal(
            result.edge_type, np.zeros(len(expected_rows), dtype=np.int64)
        )


@pytest.mark.parametrize("excluded_relation", RELATION_NAMES)
def test_relation_removal_changes_only_that_relation_and_preserves_stable_ids(excluded_relation):
    query = _geometry([[0, 0]], [[0, -10]])
    source = np.column_stack((np.linspace(0.1, 1.0, 12), np.zeros(12)))
    candidates = _geometry(source, source - [0, 10])
    arguments = (query, np.array([100]), candidates, np.arange(12), np.ones(12, bool))
    settings = {"relation_scales_m": np.ones((4, 2)), "neighbors_per_relation": 3}
    full = select_trace_graph_neighbors(*arguments, **settings)
    removed = select_trace_graph_neighbors(
        *arguments, excluded_relation=excluded_relation, **settings
    )
    keep = full.edge_type != RELATION_NAMES.index(excluded_relation)
    for name in ("sender_ids", "destination_ids", "edge_type", "distances"):
        np.testing.assert_array_equal(getattr(removed, name), getattr(full, name)[keep])
    plan = build_trace_graph_subgraph(
        *arguments, rounds=2, excluded_relation=excluded_relation, **settings
    )
    assert not np.any(plan.degree[:, RELATION_NAMES.index(excluded_relation)])
    assert not np.any(plan.coverage[:, RELATION_NAMES.index(excluded_relation)])
    assert plan.relation_names == RELATION_NAMES


def test_single_4d_degree_budget_and_common_union_distance_keep_distinct_meanings():
    source = np.column_stack((np.linspace(0.0, 1.0, 8), np.zeros(8)))
    candidate = _geometry(source, source - [0, 2])
    query = _geometry([[0.45, 0]], [[0.45, -2]])
    args = (query, np.array([100]), candidate, np.arange(8), np.ones(8, bool))
    settings = dict(
        rounds=1,
        relation_scales_m=np.full((4, 2), 2.0),
        neighbors_per_relation=2,
        common_distance_scales_m=(2.0, 4.0),
    )
    union = build_trace_graph_subgraph(*args, **settings)
    single = build_trace_graph_subgraph(
        *args, topology="single_4d", single_4d_neighbors=4, **settings
    )
    query_index = union.query_indices[0]
    assert union.degree[query_index].tolist() == [2, 2, 2, 2]
    assert union.diagnostics["typed_edge_count"] == 8
    assert union.diagnostics["unique_pair_count"] == 2
    assert single.relation_names == ("untyped",)
    assert single.degree[single.query_indices[0]].tolist() == [4, 0, 0, 0]
    assert single.coverage[single.query_indices[0], 0, 0] == 1
    assert single.diagnostics["typed_edge_count"] == single.diagnostics["unique_pair_count"] == 4
    np.testing.assert_allclose(single.common_edge_distances, single.edge_distances)
    for pair in np.unique(union.edge_index, axis=1).T:
        indices = np.all(union.edge_index == pair[:, None], axis=0)
        assert len(np.unique(union.common_edge_distances[indices])) == 1
        assert len(np.unique(union.edge_distances[indices])) > 1


def test_mean_and_gate_use_identical_graph_and_feature_hashes_while_azimuth_removal_is_exact():
    domain, training, values = make_relational_trace_domains()
    preprocessing = fit_trace_graph_preprocessing(
        training, values, position_scale_m=10, offset_scale_m=2, azimuth_min_offset_m=0.1
    )
    plan = make_relational_trace_plan(domain)
    inputs = MaskedTraceSource(domain, preprocessing, values).inputs(plan)
    original_nodes, original_edges = inputs.node_features.clone(), inputs.edge_features.clone()
    hashes = []
    for fusion in ("mean", "learned_gate"):
        model = RelationalTraceGraphInterpolator(width=8, relation_fusion=fusion)
        nodes, edges = model.input_features(inputs)
        hashes.append(
            (_hash(inputs.edge_index), _hash(inputs.edge_type), _hash(nodes), _hash(edges))
        )
    assert hashes[0] == hashes[1]
    model = RelationalTraceGraphInterpolator(width=8, explicit_azimuth_features=False)
    nodes, edges = model.input_features(inputs)
    assert torch.equal(nodes[:, 5:8], torch.zeros_like(nodes[:, 5:8]))
    assert torch.equal(edges[:, 11:14], torch.zeros_like(edges[:, 11:14]))
    assert torch.equal(nodes[:, [0, 1, 2, 3, 4, 8]], original_nodes[:, [0, 1, 2, 3, 4, 8]])
    assert torch.equal(edges[:, list(range(11)) + [14]], original_edges[:, list(range(11)) + [14]])
    assert _hash(inputs.edge_index) == hashes[0][0] and _hash(inputs.edge_type) == hashes[0][1]
    assert torch.equal(inputs.node_features, original_nodes) and torch.equal(
        inputs.edge_features, original_edges
    )
    assert model.constructor_config()["explicit_azimuth_features"] is False
    assert (
        "explicit_azimuth_features" not in RelationalTraceGraphInterpolator().constructor_config()
    )


@pytest.mark.parametrize(
    "settings,model",
    [
        ({"topology": "single_4d", "common_distance_scales_m": (1.0, 2.0)}, {}),
        (
            {"topology": "single_4d", "common_distance_scales_m": (1.0, 2.0)},
            {"method_variant": "plain_gcn_row_normalized"},
        ),
        ({}, {"method_variant": "untyped_edge_conditioned"}),
    ],
)
def test_invalid_graph_model_combinations_are_rejected(settings, model):
    graph = TraceGraphSettings(((1.0, 2.0),) * 4, **settings)
    with pytest.raises(ValueError, match="requires"):
        graph.validate_model_config(model)


@pytest.mark.parametrize(
    "settings",
    [
        {"topology": "single_4d"},
        {
            "topology": "single_4d",
            "common_distance_scales_m": (1.0, 2.0),
            "excluded_relation": "source",
        },
        {"excluded_relation": "bad"},
        {"common_distance_scales_m": (0, 1)},
        {"single_4d_neighbors": 64},
    ],
)
def test_invalid_ablation_settings_fail_without_silently_ignoring_options(settings):
    with pytest.raises(ValueError):
        TraceGraphSettings(((1.0, 2.0),) * 4, **settings)
