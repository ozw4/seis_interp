"""Aggregation controls on nonzero waveforms and complete dependency closures."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from seis_interp.data.masked_trace_source import assemble_masked_trace_graph_inputs
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.models.trace_graph_comparison import TraceGraphComparisonBlock
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph


class _Zero(nn.Module):
    def forward(self, values):
        return torch.zeros_like(values)


def _geometry(x):
    x = np.asarray(x, dtype=np.float64)
    source = np.column_stack((x, 0.07 * np.sin(x)))
    return compute_trace_graph_geometry(source, source - [10.0, 2.0], azimuth_min_offset_m=0.01)


def _inputs(query_x=(0.0, 2.2), query_ids=(100, 101), *, rounds=2, topology="multi_relation"):
    candidates = _geometry([0.8, 1.55, 2.3, 3.05])
    settings = TraceGraphSettings(
        ((1.4, 1.4), (1.4, 1.4), (1.0, 1.0), (1.0, 1.0)),
        neighbors_per_relation=2,
        common_distance_scales_m=(1.0, 1.0),
        topology=topology,
        single_4d_neighbors=2 if topology == "single_4d" else 32,
    )
    plan = build_trace_graph_subgraph(
        _geometry(query_x),
        np.array(query_ids),
        candidates,
        np.arange(1, 5),
        np.ones(4, bool),
        rounds=rounds,
        **settings.subgraph_kwargs(),
    )
    time_s = np.arange(9, dtype=np.float64)
    phase = np.linspace(0.2, 3.0, 9)[None]
    values = np.sin(phase * np.arange(2, 6)[:, None]) + np.arange(1, 5)[:, None] * 0.1
    preprocessing = TraceGraphPreprocessing(
        amplitude_scale=1.0,
        midpoint_origin_m=(0.0, 0.0),
        position_scale_m=10.0,
        offset_scale_m=10.0,
        azimuth_min_offset_m=0.01,
        time_s=tuple(time_s),
        fit_domain={},
    )
    inputs = assemble_masked_trace_graph_inputs(
        plan,
        observed_trace_ids=np.arange(1, 5),
        observed_waveforms=values.astype(np.float32),
        time_s=time_s,
        preprocessing=preprocessing,
    )
    return inputs, plan


def _model(variant):
    torch.manual_seed(37)
    model = RelationalTraceGraphInterpolator(
        width=8,
        stem_kernel_size=3,
        temporal_kernel_size=3,
        attention_width=5,
        relation_embedding_dim=3,
        method_variant=variant,
    )
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(std=0.1)
        model.decoder.head[-1].bias.fill_(0.03)
    return model


def test_row_normalized_gcn_matches_unique_observed_neighbors_plus_one_local_self_term():
    block = TraceGraphComparisonBlock(
        8,
        method_variant="plain_gcn_row_normalized",
        temporal_kernel_size=3,
        temporal_dilation=1,
        attention_width=5,
    )
    block.temporal_update = _Zero()
    block.value_projection = nn.Identity()
    block.message_update = nn.Identity()
    latents = torch.tensor([1.0, 3.0, 8.0])[:, None, None].expand(-1, 8, 3).clone()
    # Four typed copies of 0->2, two copies of 1->2, and 0->1.
    edges = torch.tensor([[0, 0, 0, 0, 1, 1, 0], [2, 2, 2, 2, 2, 2, 1]])
    types = torch.tensor([0, 1, 2, 3, 0, 3, 2])
    output = block(latents, edges, types, torch.zeros(7, 15))
    expected = torch.tensor([1 + 1, 3 + (3 + 1) / 2, 8 + (8 + 1 + 3) / 3])
    torch.testing.assert_close(output, expected[:, None, None].expand_as(output))
    assert not torch.any(edges[0] == 2)  # The query's local self term creates no sending edge.


@pytest.mark.parametrize("variant", ["plain_gcn_row_normalized", "untyped_edge_conditioned"])
def test_comparison_ignores_relation_labels_and_multiplicity(variant):
    model = _model(variant)
    inputs, _ = _inputs()
    before, _ = model(inputs)
    types = 3 - inputs.edge_type
    coverage = inputs.coverage.flip(1)
    features = inputs.edge_features.clone()
    features[:, -1] = 3.0 + types  # A relation's D must never enter the untyped MLP.
    renamed = replace(inputs, edge_type=types, coverage=coverage, edge_features=features)
    after, _ = model(renamed)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    repeated = replace(
        inputs,
        edge_index=inputs.edge_index.repeat(1, 3),
        edge_type=inputs.edge_type.repeat(3),
        edge_features=inputs.edge_features.repeat(3, 1),
        common_edge_distances=inputs.common_edge_distances.repeat(3),
    )
    # Coverage describes relation availability; comparison ignores its degree values.
    duplicate, _ = model(repeated)
    torch.testing.assert_close(before, duplicate, atol=0, rtol=0)
    assert before.abs().max() > 1e-4
    assert not any(
        "relation_embedding" in name or "relation_gate" in name
        for name, _ in model.named_parameters()
    )


@pytest.mark.parametrize(
    "variant,topology",
    [
        ("plain_gcn_row_normalized", "multi_relation"),
        ("untyped_edge_conditioned", "multi_relation"),
        ("untyped_edge_conditioned", "single_4d"),
    ],
)
def test_comparison_full_closure_query_split_and_addition_agree_for_nonzero_predictions(
    variant, topology
):
    model = _model(variant)
    joint, _ = _inputs(topology=topology)
    prediction, _ = model(joint)
    full, _ = _inputs(rounds=6, topology=topology)
    torch.testing.assert_close(prediction, model(full)[0], atol=1e-6, rtol=1e-5)
    pieces = [
        model(_inputs((x,), (identity,), topology=topology)[0])[0]
        for x, identity in ((0.0, 100), (2.2, 101))
    ]
    torch.testing.assert_close(prediction, torch.cat(pieces), atol=1e-6, rtol=1e-5)
    reverse, _ = _inputs((2.2, 0.0), (101, 100), topology=topology)
    torch.testing.assert_close(prediction.flip(0), model(reverse)[0], atol=1e-6, rtol=1e-5)
    added, _ = _inputs((0.0, 2.2, 100.0), (100, 101, 102), topology=topology)
    extended, context = model(added)
    torch.testing.assert_close(prediction, extended[:2], atol=1e-6, rtol=1e-5)
    assert not context[-1] and torch.equal(extended[-1], torch.zeros_like(extended[-1]))
    assert prediction.abs().max() > 1e-4
    altered = joint.waveforms.flip(1)
    assert (model(replace(joint, waveforms=altered))[0] - prediction).abs().max() > 1e-5


@pytest.mark.parametrize("variant", ["plain_gcn_row_normalized", "untyped_edge_conditioned"])
def test_comparison_diagnostics_do_not_change_outputs_gradients_or_count_shared_support(variant):
    model = _model(variant)
    joint, _ = _inputs((0.0, 2.2, 100.0), (100, 101, 102))
    diagnostics = {}
    output, _ = model(joint, diagnostics=diagnostics)
    output.square().sum().backward()
    gradients = {name: value.grad.clone() for name, value in model.named_parameters()}
    model.zero_grad()
    plain, _ = model(joint)
    plain.square().sum().backward()
    assert torch.equal(output, plain)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.grad, gradients[name])
    assert diagnostics["gate_sum"].shape == (2, 1)
    assert torch.equal(diagnostics["context_query_count"], torch.tensor([2, 2]))
    assert torch.equal(diagnostics["no_context_query_count"], torch.tensor([1, 1]))
    pieces = []
    for x, identity in ((0.0, 100), (2.2, 101), (100.0, 102)):
        summary = {}
        model(_inputs((x,), (identity,))[0], diagnostics=summary)
        pieces.append(summary)
    for name, value in diagnostics.items():
        torch.testing.assert_close(value, sum(piece[name] for piece in pieces))


def test_untyped_requires_common_distances_and_comparisons_reject_relation_gate():
    inputs, _ = _inputs()
    with pytest.raises(ValueError, match="common_edge_distances"):
        _model("untyped_edge_conditioned")(replace(inputs, common_edge_distances=None))
    for variant in ("plain_gcn_row_normalized", "untyped_edge_conditioned"):
        with pytest.raises(ValueError, match="relation_fusion=mean"):
            RelationalTraceGraphInterpolator(method_variant=variant, relation_fusion="learned_gate")
