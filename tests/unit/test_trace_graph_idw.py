"""IDW weights unique direct observed senders in physical amplitude units."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_idw import predict_trace_graph_idw
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph


def _plan():
    source = np.array([[1.0, 0.0], [2.0, 0.0]])
    candidates = compute_trace_graph_geometry(source, source, azimuth_min_offset_m=0)
    queries = np.array([[0.0, 0.0], [1000.0, 0.0]])
    return build_trace_graph_subgraph(
        compute_trace_graph_geometry(queries, queries, azimuth_min_offset_m=0),
        [100, 101],
        candidates,
        [10, 20],
        np.ones(2, dtype=bool),
        rounds=1,
        relation_scales_m=np.full((4, 2), 10.0),
        neighbors_per_relation=2,
    )


def test_two_sender_idw_matches_hand_calculation_without_relation_double_count():
    plan = _plan()
    assert len(plan.edge_type) == 8
    values = np.array([[10.0, -2.0], [30.0, 6.0]], dtype=np.float32)
    predictions, flags = predict_trace_graph_idw(
        plan,
        observed_trace_ids=np.array([10, 20]),
        observed_waveforms=values,
        midpoint_scale_m=1.0,
        offset_scale_m=1.0,
    )
    # D0 is 1 and 2, so normalized weights are 0.8 and 0.2.
    np.testing.assert_allclose(predictions, [[14.0, -0.4], [0.0, 0.0]], atol=1e-7)
    assert flags.tolist() == [True, False]
    # Repeating only one sender's typed edges must not increase its weight.
    selected = np.flatnonzero(plan.trace_ids[plan.edge_index[0]] == 10)
    repeated = replace(
        plan,
        edge_index=np.concatenate((plan.edge_index, plan.edge_index[:, selected]), axis=1),
        edge_type=np.concatenate((plan.edge_type, plan.edge_type[selected])),
        edge_distances=np.concatenate((plan.edge_distances, plan.edge_distances[selected])),
    )
    actual, _ = predict_trace_graph_idw(
        repeated,
        observed_trace_ids=np.array([20, 10]),
        observed_waveforms=values[::-1],
        midpoint_scale_m=1.0,
        offset_scale_m=1.0,
    )
    np.testing.assert_array_equal(actual, predictions)


def test_idw_uses_full_offset_distance_and_never_accepts_query_waveforms():
    source = np.array([[0.0, 1.0], [0.0, 2.0]])
    candidates = compute_trace_graph_geometry(source, -source, azimuth_min_offset_m=0)
    zeros = np.zeros((1, 2))
    plan = build_trace_graph_subgraph(
        compute_trace_graph_geometry(zeros, zeros, azimuth_min_offset_m=0),
        [100],
        candidates,
        [10, 20],
        np.ones(2, dtype=bool),
        rounds=1,
        relation_scales_m=np.full((4, 2), 10.0),
        neighbors_per_relation=2,
    )
    values = np.array([[10.0], [30.0]], dtype=np.float32)
    prediction, _ = predict_trace_graph_idw(
        plan,
        observed_trace_ids=np.array([10, 20]),
        observed_waveforms=values,
        midpoint_scale_m=1,
        offset_scale_m=2,
    )
    np.testing.assert_array_equal(prediction, [[14.0]])
    with pytest.raises(ValueError, match="query IDs"):
        predict_trace_graph_idw(
            plan,
            observed_trace_ids=np.array([10, 20, 100]),
            observed_waveforms=np.array([[10.0], [30.0], [999.0]], dtype=np.float32),
            midpoint_scale_m=1,
            offset_scale_m=2,
        )
