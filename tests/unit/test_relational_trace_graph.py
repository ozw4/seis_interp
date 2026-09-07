"""Direct waveforms and finite dependency invariance for trace graph models."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_inputs import MaskedTraceGraphInputs
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.models.trace_codec import TraceNodeDecoder, TraceNodeEncoder
from seis_interp.processing.trace_graph_geometry import (
    build_trace_graph_edge_features,
    build_trace_graph_node_features,
    compute_trace_graph_geometry,
)
from seis_interp.processing.trace_graph_subgraphs import (
    TraceGraphPlan,
    build_trace_graph_subgraph,
)


def _model(fusion: str = "mean", *, nonzero_head: bool = True, **kwargs):
    torch.manual_seed(37)
    model = RelationalTraceGraphInterpolator(
        **{
            "width": 8,
            "stem_kernel_size": 3,
            "temporal_kernel_size": 3,
            "attention_width": 5,
            "relation_embedding_dim": 3,
            "relation_fusion": fusion,
            **kwargs,
        }
    )
    if nonzero_head:
        with torch.no_grad():
            model.decoder.head[-1].weight.normal_(std=0.1)
            model.decoder.head[-1].bias.fill_(0.03)
    return model


def _geometry(x):
    x = np.asarray(x, dtype=np.float64)
    source = np.column_stack((x, 0.07 * np.sin(x)))
    return compute_trace_graph_geometry(source, source - [10.0, 2.0], azimuth_min_offset_m=0.01)


def _plan(query_x=(0.0, 2.2), query_ids=(100, 101), *, rounds=2):
    return build_trace_graph_subgraph(
        _geometry(query_x),
        np.array(query_ids, dtype=np.int64),
        _geometry([0.8, 1.55, 2.3, 3.05]),
        np.arange(1, 5, dtype=np.int64),
        np.ones(4, dtype=bool),
        rounds=rounds,
        relation_scales_m=np.array([[1.4, 1.4], [1.4, 1.4], [1.0, 1.0], [1.0, 1.0]]),
        neighbors_per_relation=2,
    )


def _inputs(plan: TraceGraphPlan, *, time=9) -> MaskedTraceGraphInputs:
    node_features = build_trace_graph_node_features(
        plan.geometry,
        midpoint_origin_m=np.zeros(2),
        position_scale_m=10,
        offset_scale_m=10,
        observed_mask=plan.observed_mask,
    )
    edge_features = build_trace_graph_edge_features(
        plan.geometry,
        sender_indices=plan.edge_index[0],
        destination_indices=plan.edge_index[1],
        relation_distances=plan.edge_distances,
        position_scale_m=10,
        offset_scale_m=10,
    )
    phase = torch.linspace(0.2, 3.0, time)[None]
    ids = torch.from_numpy(plan.trace_ids).float()[:, None]
    waveforms = torch.sin(phase * (ids + 1)) + 0.1 * ids
    waveforms[~plan.observed_mask] = 0
    return MaskedTraceGraphInputs(
        waveforms=waveforms,
        node_features=torch.from_numpy(node_features),
        edge_features=torch.from_numpy(edge_features),
        edge_index=torch.from_numpy(plan.edge_index),
        edge_type=torch.from_numpy(plan.edge_type),
        observed_mask=torch.from_numpy(plan.observed_mask),
        query_indices=torch.from_numpy(plan.query_indices),
        coverage=torch.from_numpy(plan.coverage),
    )


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
@pytest.mark.parametrize("time,factor", [(9, 2), (1, 2), (1, 1), (7, 3)])
def test_direct_shape_crop_and_zero_initialized_decoder(fusion, time, factor) -> None:
    model = _model(fusion, nonzero_head=False, time_downsample_factor=factor)
    inputs = _inputs(_plan(), time=time)
    prediction, context = model(inputs)
    assert isinstance(model.encoder, TraceNodeEncoder)
    assert isinstance(model.decoder, TraceNodeDecoder)
    assert prediction.shape == (2, time)
    assert torch.equal(prediction, torch.zeros_like(prediction))
    assert torch.equal(context, torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
def test_observed_waveforms_affect_nonzero_predictions(fusion) -> None:
    model = _model(fusion)
    inputs = _inputs(_plan())
    prediction, _ = model(inputs)
    changed = inputs.waveforms.flip(dims=[1]).clone()
    changed[inputs.observed_mask, 3] += 2
    altered, _ = model(replace(inputs, waveforms=changed))
    assert prediction.abs().max() > 1e-4
    assert (prediction - altered).abs().max() > 1e-5


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
def test_two_round_closure_matches_full_graph_with_excluded_boundary_edges(fusion) -> None:
    model = _model(fusion)
    closed = _plan((0.0,), (100,), rounds=2)
    full = _plan((0.0,), (100,), rounds=6)
    assert (closed.depth == 2).any()
    leaves = closed.trace_ids[closed.depth == 2]
    assert not np.isin(closed.trace_ids[closed.edge_index[1]], leaves).any()
    assert np.isin(full.trace_ids[full.edge_index[1]], leaves).any()
    short_prediction, _ = model(_inputs(closed))
    full_prediction, _ = model(_inputs(full))
    assert short_prediction.abs().max() > 1e-4
    torch.testing.assert_close(short_prediction, full_prediction, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
def test_query_order_split_addition_and_node_order_preserve_predictions(fusion) -> None:
    model = _model(fusion)
    if fusion == "learned_gate":
        with torch.no_grad():
            for block in model.rounds:
                block.relation_gate[-1].weight.normal_(std=0.4)
    joint = _inputs(_plan())
    prediction, _ = model(joint)
    separate = torch.cat(
        [model(_inputs(_plan((x,), (trace_id,))))[0] for x, trace_id in [(0.0, 100), (2.2, 101)]]
    )
    reversed_prediction, _ = model(_inputs(_plan((2.2, 0.0), (101, 100))))
    order = torch.arange(len(joint.waveforms) - 1, -1, -1)
    inverse = torch.argsort(order)
    permuted = replace(
        joint,
        waveforms=joint.waveforms[order],
        node_features=joint.node_features[order],
        observed_mask=joint.observed_mask[order],
        coverage=joint.coverage[order],
        edge_index=inverse[joint.edge_index],
        query_indices=inverse[joint.query_indices],
    )
    node_prediction, _ = model(permuted)
    assert prediction.abs().max() > 1e-4
    torch.testing.assert_close(prediction, separate, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(prediction.flip(0), reversed_prediction, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(prediction, node_prediction, atol=1e-6, rtol=1e-5)
    # The added query gives the original depth-two leaf incoming edges.
    single = _plan((0.0,), (100,))
    leaves = single.trace_ids[single.depth == 2]
    together = _plan()
    assert np.isin(together.trace_ids[together.edge_index[1]], leaves).any()


@pytest.mark.parametrize("factor", [1, 2])
def test_single_zero_context_trace_with_one_time_sample_is_exact_zero(factor) -> None:
    model = _model("learned_gate", time_downsample_factor=factor)
    plan = _plan((100.0,), (100,))
    inputs = _inputs(plan, time=1)
    assert len(inputs.waveforms) == 1 and inputs.edge_index.shape[1] == 0
    prediction, context = model(inputs)
    assert torch.equal(prediction, torch.zeros(1, 1))
    assert torch.equal(context, torch.tensor([False]))


def test_zero_context_query_is_retained_alongside_context_query() -> None:
    inputs = _inputs(_plan((0.0, 100.0), (100, 101)))
    prediction, context = _model()(inputs)
    assert prediction.shape == (2, 9)
    assert torch.equal(context, torch.tensor([True, False]))
    assert prediction[0].abs().max() > 1e-4
    assert torch.equal(prediction[1], torch.zeros(9))


def test_observed_zero_and_missing_are_distinguished_by_mask_features() -> None:
    model = _model()
    inputs = _inputs(_plan())
    features = inputs.node_features.clone()
    features[1] = features[0]
    features[0, -1] = 1
    features[1, -1] = 0
    embeddings = model.node_embedding(features[:2])
    assert not torch.allclose(embeddings[0], embeddings[1])
    observed_zero = replace(inputs, waveforms=torch.zeros_like(inputs.waveforms))
    prediction, context = model(observed_zero)
    assert context.all() and torch.isfinite(prediction).all()


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
def test_tiny_analytic_waveform_learning_reduces_loss(fusion) -> None:
    model = _model(fusion, nonzero_head=False)
    inputs = _inputs(_plan())
    target = 0.3 * torch.sin(torch.linspace(0.0, 3.0, 9))[None].expand(2, -1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = torch.mean((model(inputs)[0] - target).square()).item()
    for _ in range(16):
        optimizer.zero_grad()
        prediction, _ = model(inputs)
        loss = torch.mean((prediction - target).square())
        loss.backward()
        optimizer.step()
    final = torch.mean((model(inputs)[0] - target).square()).item()
    assert final < initial * 0.5
    assert model.encoder.stem.weight.grad.abs().sum() > 0
    assert model.rounds[0].gamma[0].weight.grad.abs().sum() > 0
    if fusion == "learned_gate":
        assert model.rounds[0].relation_gate[0].weight.grad.abs().sum() > 0


def test_gate_initialization_matches_mean_model_with_shared_parameters() -> None:
    mean = _model("mean")
    learned = _model("learned_gate")
    missing, unexpected = learned.load_state_dict(mean.state_dict(), strict=False)
    assert missing and all("relation_gate" in name for name in missing)
    assert not unexpected
    inputs = _inputs(_plan())
    torch.testing.assert_close(mean(inputs)[0], learned(inputs)[0], atol=1e-6, rtol=1e-5)


def test_diagnostics_are_detached_compact_and_do_not_change_prediction_or_gradient() -> None:
    model = _model("learned_gate")
    inputs = _inputs(_plan())
    diagnostics = {}
    diagnostic_prediction, _ = model(inputs, diagnostics=diagnostics)
    diagnostic_prediction.square().sum().backward()
    gradients = {name: value.grad.clone() for name, value in model.named_parameters()}
    model.zero_grad()
    plain_prediction, _ = model(inputs)
    plain_prediction.square().sum().backward()
    assert torch.equal(diagnostic_prediction, plain_prediction)
    assert set(diagnostics) == {"gate_mean", "available_count"}
    assert all(value.shape == (2, 4) for value in diagnostics.values())
    assert all(value.grad_fn is None and not value.requires_grad for value in diagnostics.values())
    for name, value in model.named_parameters():
        assert torch.equal(gradients[name], value.grad)


def test_constructor_config_roundtrips_pure_values_and_is_independent() -> None:
    model = _model("learned_gate")
    config = model.constructor_config()
    restored = RelationalTraceGraphInterpolator(**json.loads(json.dumps(config)))
    assert restored.constructor_config() == config
    config["temporal_dilations"][0] = 99
    assert model.constructor_config()["temporal_dilations"] == [1, 2]


def test_empty_query_selection_returns_empty_waveforms_and_context() -> None:
    inputs = _inputs(_plan())
    empty = replace(inputs, query_indices=torch.empty(0, dtype=torch.int64))
    prediction, context = _model()(empty)
    assert prediction.shape == (0, 9) and context.shape == (0,)


def test_rejects_dilation_count_mismatch() -> None:
    with pytest.raises(ValueError, match="message_passing_rounds"):
        _model(message_passing_rounds=3)


def test_forward_rejects_nonzero_hidden_waveforms_and_hidden_senders() -> None:
    model = _model()
    inputs = _inputs(_plan())
    corrupted = inputs.waveforms.clone()
    corrupted[inputs.query_indices] = 1
    with pytest.raises(ValueError, match="exactly zero"):
        model(replace(inputs, waveforms=corrupted))
    edges = inputs.edge_index.clone()
    edges[0, 0] = inputs.query_indices[-1]
    with pytest.raises(ValueError, match="sender"):
        model(replace(inputs, edge_index=edges))
