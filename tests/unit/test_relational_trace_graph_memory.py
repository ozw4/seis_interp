"""Exact memory options of the relational trace graph model against its original path."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_inputs import MaskedTraceGraphInputs
from seis_interp.models import relational_trace_graph
from seis_interp.models.relational_trace_graph import (
    TRACE_GRAPH_MEMORY_OPTIONS,
    RelationalTraceGraphInterpolator,
)
from seis_interp.processing.trace_graph_geometry import (
    build_trace_graph_edge_features,
    build_trace_graph_node_features,
    compute_trace_graph_geometry,
)
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph

_SWITCH_COMBINATIONS = [
    dict(zip(TRACE_GRAPH_MEMORY_OPTIONS, flags, strict=True))
    for flags in itertools.product((False, True), repeat=3)
    if any(flags)
]


def _geometry(points: np.ndarray):
    return compute_trace_graph_geometry(points[:, :2], points[:, 2:], azimuth_min_offset_m=0.01)


def _traces(rng: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    source = rng.uniform(low, high, size=(count, 2))
    return np.column_stack((source, source + rng.uniform(-3, 3, size=(count, 2))))


def _inputs(
    neighbors: int,
    rounds: int,
    *,
    queries: int = 5,
    time: int = 12,
    dtype: torch.dtype = torch.float64,
    seed: int = 3,
) -> MaskedTraceGraphInputs:
    rng = np.random.default_rng(seed)
    candidates = _traces(rng, 60, 0.0, 4.0)
    query_points = _traces(rng, queries, 0.5, 3.5)
    plan = build_trace_graph_subgraph(
        _geometry(query_points),
        np.arange(1000, 1000 + queries),
        _geometry(candidates),
        np.arange(1, 61),
        np.ones(60, dtype=bool),
        rounds=rounds,
        relation_scales_m=np.full((4, 2), 1.5),
        neighbors_per_relation=neighbors,
        common_distance_scales_m=(1.5, 1.5),
    )
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
    waveforms = torch.from_numpy(rng.standard_normal((len(plan.trace_ids), time)))
    waveforms[~torch.from_numpy(plan.observed_mask)] = 0
    return MaskedTraceGraphInputs(
        waveforms=waveforms.to(dtype),
        node_features=torch.from_numpy(node_features).to(dtype),
        edge_features=torch.from_numpy(edge_features).to(dtype),
        edge_index=torch.from_numpy(plan.edge_index),
        edge_type=torch.from_numpy(plan.edge_type),
        observed_mask=torch.from_numpy(plan.observed_mask),
        query_indices=torch.from_numpy(plan.query_indices),
        coverage=torch.from_numpy(plan.coverage).to(dtype),
        dependency_rounds=plan.dependency_rounds,
        common_edge_distances=torch.from_numpy(plan.common_edge_distances).to(dtype),
    )


def _model(dtype: torch.dtype = torch.float64, **config) -> RelationalTraceGraphInterpolator:
    torch.manual_seed(11)
    model = RelationalTraceGraphInterpolator(
        **{
            "width": 8,
            "stem_kernel_size": 3,
            "temporal_kernel_size": 3,
            "attention_width": 5,
            "relation_embedding_dim": 3,
            **config,
        }
    )
    # The decoder head is zero-initialized; perturb it so every layer gets gradients.
    with torch.no_grad():
        model.decoder.head[-1].weight.copy_(
            0.1 * torch.randn(model.decoder.head[-1].weight.shape, generator=torch.Generator())
        )
    return model.to(dtype)


def _evaluate(model: RelationalTraceGraphInterpolator, inputs: MaskedTraceGraphInputs):
    diagnostics = {}
    prediction, context = model(inputs, diagnostics=diagnostics)
    weights = torch.linspace(-1, 1, prediction.numel(), dtype=prediction.dtype)
    (prediction * weights.reshape(prediction.shape)).sum().backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    return prediction.detach(), context, diagnostics, gradients


def _assert_equivalent(reference, result, rtol: float) -> None:
    torch.testing.assert_close(result[0], reference[0], rtol=rtol, atol=rtol)
    assert torch.equal(result[1], reference[1])
    assert result[2].keys() == reference[2].keys()
    for name, value in reference[2].items():
        torch.testing.assert_close(result[2][name], value, rtol=rtol, atol=rtol)
    scale = max(float(gradient.abs().max()) for gradient in reference[3].values())
    assert result[3].keys() == reference[3].keys()
    for name, gradient in reference[3].items():
        torch.testing.assert_close(result[3][name], gradient, rtol=rtol, atol=rtol * scale)


_EQUIVALENCE_CASES = [
    dict(neighbors=1, message_passing_rounds=1, temporal_dilations=(1,)),
    dict(
        neighbors=2,
        relation_fusion="learned_gate",
        attention_pooling="rms",
        node_fourier_components=4,
    ),
    dict(
        neighbors=3,
        message_passing_rounds=3,
        temporal_dilations=(2, 1, 3),
        relation_fusion="learned_gate",
        relation_gate_pooling="rms",
    ),
    dict(
        neighbors=2,
        message_passing_rounds=3,
        temporal_dilations=(1, 3, 2),
        attention_pooling="rms",
        node_fourier_components=3,
    ),
    dict(
        neighbors=3,
        temporal_dilations=(3, 6),
        relation_fusion="learned_gate",
        attention_pooling="rms",
        relation_gate_pooling="rms",
        node_fourier_components=4,
    ),
    dict(neighbors=1, message_passing_rounds=3, temporal_dilations=(1, 1, 1)),
]


@pytest.mark.parametrize("case", _EQUIVALENCE_CASES)
def test_every_switch_combination_matches_original_predictions_and_gradients(case) -> None:
    config = dict(case)
    neighbors = config.pop("neighbors")
    inputs = _inputs(neighbors, config.get("message_passing_rounds", 2))
    reference_model = _model(**config)
    reference = _evaluate(reference_model, inputs)
    assert reference[0].abs().max() > 1e-3
    for switches in _SWITCH_COMBINATIONS:
        model = _model(**config, **switches)
        for (name, value), expected in zip(
            model.state_dict().items(), reference_model.state_dict().items(), strict=True
        ):
            assert name == expected[0] and torch.equal(value, expected[1])
        _assert_equivalent(reference, _evaluate(model, inputs), rtol=1e-12)


@pytest.mark.parametrize(
    "switches",
    [
        {"receptive_field_pruning": True},
        {"receptive_field_pruning": True, "activation_checkpointing": True},
    ],
)
def test_pruning_supports_trace_amplitudes_and_edge_lags(switches) -> None:
    config = dict(
        relation_fusion="learned_gate",
        amplitude_mode="observed_trace_rms",
        max_edge_time_shift_samples=3,
    )
    inputs = _inputs(2, 2)
    reference_model = _model(**config)
    with torch.no_grad():
        reference_model.edge_time_shift_weights.copy_(torch.tensor([0.4, -0.3, 0.2, 0.5]))
    model = _model(**config, **switches)
    model.load_state_dict(reference_model.state_dict())
    _assert_equivalent(_evaluate(reference_model, inputs), _evaluate(model, inputs), rtol=1e-12)


def test_spectral_input_block_is_supported_in_float32() -> None:
    inputs = _inputs(2, 2, dtype=torch.float32)
    reference = _evaluate(_model(torch.float32, spectral_input_block=True), inputs)
    switches = dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True)
    result = _evaluate(_model(torch.float32, spectral_input_block=True, **switches), inputs)
    _assert_equivalent(reference, result, rtol=2e-5)


def test_inference_and_diagnostics_match_without_gradients() -> None:
    inputs = _inputs(3, 2)
    switches = dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True)
    outputs = []
    for model in (
        _model(relation_fusion="learned_gate"),
        _model(relation_fusion="learned_gate", **switches),
    ):
        diagnostics = {}
        with torch.inference_mode():
            prediction, context = model.eval()(inputs, diagnostics=diagnostics)
        outputs.append((prediction, context, diagnostics))
    torch.testing.assert_close(outputs[1][0], outputs[0][0], rtol=1e-12, atol=1e-12)
    assert torch.equal(outputs[1][1], outputs[0][1])
    for name, value in outputs[0][2].items():
        torch.testing.assert_close(outputs[1][2][name], value, rtol=1e-12, atol=1e-12)


def test_pruning_updates_only_states_that_later_rounds_read() -> None:
    inputs = _inputs(2, 2)
    rows = {}
    for pruning in (False, True):
        model = _model(receptive_field_pruning=pruning)
        seen = []
        for block in model.rounds:
            block.message_update.register_forward_hook(
                lambda module, args, output, seen=seen: seen.append(args[0].shape[0])
            )
        model(inputs)
        rows[pruning] = seen
    node_count = inputs.waveforms.shape[0]
    assert rows[False] == [node_count, node_count]
    assert rows[True][-1] == inputs.query_indices.numel()
    assert inputs.query_indices.numel() < rows[True][0] < node_count


def test_pruning_with_query_order_split_and_empty_queries() -> None:
    inputs = _inputs(2, 2)
    model = _model(receptive_field_pruning=True)
    full, _ = model(inputs)
    reversed_queries = inputs.query_indices.flip(0)
    reordered, _ = model(replace(inputs, query_indices=reversed_queries))
    torch.testing.assert_close(reordered, full.flip(0), rtol=1e-12, atol=1e-12)
    single, _ = model(replace(inputs, query_indices=inputs.query_indices[2:3]))
    torch.testing.assert_close(single, full[2:3], rtol=1e-12, atol=1e-12)
    diagnostics = {}
    empty = replace(inputs, query_indices=torch.empty(0, dtype=torch.int64))
    prediction, context = model(empty, diagnostics=diagnostics)
    assert prediction.shape == (0, 12) and context.shape == (0,)
    assert torch.equal(diagnostics["context_query_count"], torch.zeros(2, dtype=torch.int64))


def test_switches_select_their_mechanisms(monkeypatch) -> None:
    calls = {"aggregate": 0, "receptive_field": 0}
    aggregate = relational_trace_graph.aggregate_edge_messages
    receptive_field = relational_trace_graph._receptive_field

    def counted_aggregate(*args):
        calls["aggregate"] += 1
        return aggregate(*args)

    def counted_receptive_field(*args):
        calls["receptive_field"] += 1
        return receptive_field(*args)

    monkeypatch.setattr(relational_trace_graph, "aggregate_edge_messages", counted_aggregate)
    monkeypatch.setattr(relational_trace_graph, "_receptive_field", counted_receptive_field)
    inputs = _inputs(2, 2)
    model = _model(**dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True))
    encoder_calls = []
    model.encoder.register_forward_pre_hook(lambda *args: encoder_calls.append(1))
    prediction, _ = model(inputs)
    prediction.square().sum().backward()
    assert calls == {"aggregate": 2, "receptive_field": 1}
    # The checkpointed encoder starts again during backward (recomputation may stop early).
    assert len(encoder_calls) == 2


def test_default_path_uses_no_memory_mechanism(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("default path must keep the original computation")

    for name in ("aggregate_edge_messages", "_receptive_field", "checkpoint"):
        monkeypatch.setattr(relational_trace_graph, name, unexpected)
    prediction, _ = _model(relation_fusion="learned_gate")(_inputs(2, 2))
    prediction.square().sum().backward()


# Output of the implementation before the memory options existed (float64, CPU). The
# tolerance only absorbs BLAS rounding across CPUs; the freezing machine matched bitwise.
_FROZEN_PREDICTION = [
    [
        0.02979927379724483,
        -0.015909837942297604,
        0.029334751883996788,
        -0.043190713666333314,
        0.0378408449906229,
        -0.09555688204666991,
    ],
    [
        0.020381965197726174,
        -0.014597295814179553,
        0.022580476043154593,
        -0.03858929296631488,
        0.03574507377412275,
        -0.0915044598784643,
    ],
]
_FROZEN_GRADIENT_NORM = 0.76985724968064


def test_default_path_matches_frozen_previous_implementation_output() -> None:
    inputs = _inputs(2, 2, queries=2, time=6)
    model = _model(
        relation_fusion="learned_gate", attention_pooling="rms", node_fourier_components=2
    )
    prediction, _, _, gradients = _evaluate(model, inputs)
    expected = torch.tensor(_FROZEN_PREDICTION, dtype=torch.float64)
    torch.testing.assert_close(prediction, expected, rtol=1e-12, atol=1e-15)
    norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients.values()))
    frozen_norm = torch.tensor(_FROZEN_GRADIENT_NORM, dtype=torch.float64)
    torch.testing.assert_close(norm, frozen_norm, rtol=1e-12, atol=0)


def test_memory_options_add_no_parameters_or_rng_and_record_enabled_switches() -> None:
    torch.manual_seed(5)
    base = RelationalTraceGraphInterpolator(relation_fusion="learned_gate")
    base_rng = torch.random.get_rng_state()
    torch.manual_seed(5)
    explicit_off = RelationalTraceGraphInterpolator(
        relation_fusion="learned_gate", **dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, False)
    )
    torch.manual_seed(5)
    enabled = RelationalTraceGraphInterpolator(
        relation_fusion="learned_gate", **dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True)
    )
    assert torch.equal(torch.random.get_rng_state(), base_rng)
    assert explicit_off.constructor_config() == base.constructor_config()
    assert enabled.constructor_config() == {
        **base.constructor_config(),
        **dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True),
    }
    for model in (explicit_off, enabled):
        assert list(model.state_dict()) == list(base.state_dict())
        assert all(
            torch.equal(value, base.state_dict()[name])
            for name, value in model.state_dict().items()
        )
    restored = RelationalTraceGraphInterpolator(
        **json.loads(json.dumps(enabled.constructor_config()))
    )
    assert restored.constructor_config() == enabled.constructor_config()


def test_c3_width_128_parameter_count_is_unchanged() -> None:
    config = dict(
        width=128,
        relation_fusion="learned_gate",
        node_fourier_components=16,
        **dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True),
    )
    model = RelationalTraceGraphInterpolator(**config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 363_269


def test_forge_method_initialization_passes_switches_without_changing_weights() -> None:
    from seis_interp.training.forge_mvp_methods import initialize_model

    config = {"method_id": "regsi_real", "model_seed": 7301, "model": {"width": 16}}
    base = initialize_model(config)
    config["model"].update(dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True))
    switched = initialize_model(config)
    assert switched.receptive_field_pruning and switched.activation_checkpointing
    assert all(block.fused_edge_aggregation for block in switched.rounds)
    for (name, value), (expected_name, expected) in zip(
        switched.state_dict().items(), base.state_dict().items(), strict=True
    ):
        assert name == expected_name and torch.equal(value, expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"receptive_field_pruning": 1}, "receptive_field_pruning must be boolean"),
        ({"fused_edge_aggregation": "true"}, "fused_edge_aggregation must be boolean"),
        ({"activation_checkpointing": None}, "activation_checkpointing must be boolean"),
        (
            {"receptive_field_pruning": True, "method_variant": "plain_gcn_row_normalized"},
            "require method_variant=relational",
        ),
        (
            {"fused_edge_aggregation": True, "max_edge_time_shift_samples": 2},
            "max_edge_time_shift_samples=0",
        ),
    ],
)
def test_invalid_memory_options_are_rejected_before_rng_consumption(kwargs, match) -> None:
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match=match):
        RelationalTraceGraphInterpolator(**kwargs)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="bf16 autocast smoke test requires a bf16-capable CUDA device",
)
def test_bf16_autocast_training_and_prediction_with_all_switches_are_finite() -> None:
    device = torch.device("cuda")
    inputs = _inputs(3, 2, dtype=torch.float32, time=32).to(device)
    switches = dict.fromkeys(TRACE_GRAPH_MEMORY_OPTIONS, True)
    model = _model(torch.float32, width=16, relation_fusion="learned_gate", **switches).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction, _ = model(inputs)
    loss = prediction.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predicted, _ = model.eval()(inputs, diagnostics={})
    assert torch.isfinite(predicted).all()
