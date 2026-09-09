"""Physical frozen predictions keep one observed domain across query batches."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def _inputs(fusion="learned_gate"):
    domain, training, amplitudes = make_relational_trace_domains()
    preprocessing = fit_trace_graph_preprocessing(
        training,
        amplitudes,
        position_scale_m=10,
        offset_scale_m=10,
        azimuth_min_offset_m=0.1,
    )
    torch.manual_seed(12)
    model = RelationalTraceGraphInterpolator(
        width=8, attention_width=5, relation_embedding_dim=3, relation_fusion=fusion
    )
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(std=0.15)
        model.decoder.head[-1].bias.fill_(0.03)
        if fusion == "learned_gate":
            for block in model.rounds:
                block.relation_gate[-1].weight.normal_(std=0.4)
    return model, domain, preprocessing, amplitudes


def _settings(**overrides):
    return TraceGraphSettings(
        **{
            "relation_scales_m": ((1.0, 1.0),) * 4,
            "neighbors_per_relation": 2,
            "candidate_chunk_size": 2,
            **overrides,
        }
    )


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
@pytest.mark.parametrize("neighbor_search", ["brute_force", "exact_index"])
def test_physical_prediction_query_split_order_and_gate_totals(fusion, neighbor_search):
    model, domain, preprocessing, amplitudes = _inputs(fusion)
    arguments = {
        "graph_settings": _settings(neighbor_search=neighbor_search),
        "amplitudes": amplitudes,
    }
    results = [
        predict_relational_trace_graph(
            model, domain, preprocessing, query_batch_size=batch, **arguments
        )
        for batch in (1, 2, 20)
    ]
    expected = results[-1]
    assert np.abs(expected.prediction).max() > 1e-4
    for actual in results:
        np.testing.assert_allclose(actual.prediction, expected.prediction, atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(actual.has_observed_context, expected.has_observed_context)
        np.testing.assert_allclose(
            actual.diagnostics["gate_sum"], expected.diagnostics["gate_sum"], atol=1e-6, rtol=1e-5
        )
        for name in ("available_query_count", "context_query_count", "no_context_query_count"):
            assert actual.diagnostics[name] == expected.diagnostics[name]
    reversed_result = predict_relational_trace_graph(
        model, domain, preprocessing, query_trace_ids=expected.query_trace_ids[::-1], **arguments
    )
    np.testing.assert_allclose(
        reversed_result.prediction[::-1], expected.prediction, atol=1e-6, rtol=1e-5
    )
    np.testing.assert_array_equal(expected.query_trace_ids, domain.trace_ids[domain.query_mask])
    np.testing.assert_array_equal(expected.source_xy_m, domain.source_xy_m[domain.query_mask])


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
@pytest.mark.parametrize("neighbor_search", ["brute_force", "exact_index"])
def test_off_grid_queries_without_any_array_rows_and_query_addition(fusion, neighbor_search):
    model, domain, preprocessing, amplitudes = _inputs(fusion)
    waveforms = amplitudes[domain.array_rows[domain.observed_mask], 1:6]
    domain = replace(domain, array_rows=None)
    sources = np.array([[0.73, 0.04], [1.43, 0.1], [500.0, 9.0]])
    receivers = sources - [0.3, 2.1]
    arguments = {
        "graph_settings": _settings(neighbor_search=neighbor_search),
        "observed_waveforms": waveforms,
        "query_trace_ids": np.array([101, 102, 103]),
        "query_source_xy_m": sources,
        "query_receiver_xy_m": receivers,
    }
    result = predict_relational_trace_graph(model, domain, preprocessing, **arguments)
    assert result.prediction.shape == (3, 5)
    assert result.has_observed_context.tolist() == [True, True, False]
    assert np.array_equal(result.prediction[-1], np.zeros(5, dtype=np.float32))
    assert np.abs(result.prediction[:2]).max() > 1e-4
    assert result.diagnostics["no_context_query_count"] == [1, 1]
    arguments.update(
        query_trace_ids=np.array([101]),
        query_source_xy_m=sources[:1],
        query_receiver_xy_m=receivers[:1],
    )
    alone = predict_relational_trace_graph(model, domain, preprocessing, **arguments)
    np.testing.assert_allclose(alone.prediction, result.prediction[:1], atol=1e-6, rtol=1e-5)
    changed_waveforms = waveforms.copy()
    changed_waveforms[:, 2] += 5
    arguments["observed_waveforms"] = changed_waveforms
    changed = predict_relational_trace_graph(model, domain, preprocessing, **arguments)
    assert np.max(np.abs(changed.prediction - alone.prediction)) > 1e-5


@pytest.mark.parametrize("neighbor_search", ["brute_force", "exact_index"])
def test_frozen_prediction_reads_only_support_and_preserves_parameters_and_gradients(
    monkeypatch, neighbor_search
):
    model, domain, preprocessing, amplitudes = _inputs()
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    states = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    reads = []
    original = MaskedTraceSource.read_observed_rows

    def observed_only(source, ids):
        assert np.isin(ids, domain.trace_ids[domain.observed_mask]).all()
        reads.extend(ids.tolist())
        assert not torch.is_grad_enabled()
        assert not model.training
        return original(source, ids)

    monkeypatch.setattr(MaskedTraceSource, "read_observed_rows", observed_only)
    arguments = {
        "graph_settings": _settings(neighbor_search=neighbor_search),
        "amplitudes": amplitudes,
        "query_batch_size": 1,
    }
    result = predict_relational_trace_graph(model, domain, preprocessing, **arguments)
    changed = amplitudes.copy()
    changed[domain.array_rows[domain.query_mask]] = np.nan
    arguments["amplitudes"] = changed
    hidden_changed = predict_relational_trace_graph(model, domain, preprocessing, **arguments)
    np.testing.assert_array_equal(result.prediction, hidden_changed.prediction)
    assert reads
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, states[name], rtol=0, atol=0)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


def test_physical_scale_applied_once_after_normalized_forward(monkeypatch):
    model, domain, preprocessing, amplitudes = _inputs("mean")
    normalized_outputs = []
    original = model.forward

    def capture(inputs, **kwargs):
        prediction, context = original(inputs, **kwargs)
        normalized_outputs.append(prediction.cpu().numpy().copy())
        return prediction, context

    monkeypatch.setattr(model, "forward", capture)
    result = predict_relational_trace_graph(
        model, domain, preprocessing, graph_settings=_settings(), amplitudes=amplitudes
    )
    expected = (normalized_outputs[0].astype(np.float64) * preprocessing.amplitude_scale).astype(
        np.float32
    )
    np.testing.assert_array_equal(result.prediction, expected)


def test_time_grid_mismatch_rejected_before_support_read(monkeypatch):
    model, domain, preprocessing, amplitudes = _inputs()

    def forbidden(*args, **kwargs):
        pytest.fail("read before validating time grid")

    monkeypatch.setattr(MaskedTraceSource, "read_observed_rows", forbidden)
    with pytest.raises(ValueError, match="time grid"):
        predict_relational_trace_graph(
            model,
            replace(domain, time_s=domain.time_s + 0.001),
            preprocessing,
            graph_settings=_settings(),
            amplitudes=amplitudes,
        )


@pytest.mark.parametrize("batch", [0, -1, True, 2.5])
def test_invalid_query_batch_rejected(batch):
    model, domain, preprocessing, amplitudes = _inputs()
    with pytest.raises(ValueError, match="query_batch_size"):
        predict_relational_trace_graph(
            model,
            domain,
            preprocessing,
            graph_settings=_settings(),
            amplitudes=amplitudes,
            query_batch_size=batch,
        )


def test_explicit_query_cannot_alias_any_observed_pair_even_outside_support_radius():
    model, domain, preprocessing, amplitudes = _inputs()
    with pytest.raises(ValueError, match="duplicates an observed"):
        predict_relational_trace_graph(
            model,
            domain,
            preprocessing,
            graph_settings=_settings(radius=0.01),
            amplitudes=amplitudes,
            query_trace_ids=np.array([101]),
            query_source_xy_m=domain.source_xy_m[:1],
            query_receiver_xy_m=domain.receiver_xy_m[:1],
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"relation_scales_m": ((0.0, 1.0),) * 4},
        {"relation_scales_m": ((1.0, float("inf")),) * 4},
        {"relation_scales_m": ((1.0, 1.0),) * 3},
        {"radius": 0},
        {"neighbors_per_relation": True},
        {"candidate_chunk_size": 1.5},
        {"neighbor_search": "approximate"},
    ],
)
def test_invalid_fixed_graph_settings_rejected(settings):
    with pytest.raises(ValueError):
        _settings(**settings)


def test_graph_settings_copy_mutable_config_and_have_no_independent_round_count():
    scales = np.ones((4, 2))
    settings = _settings(relation_scales_m=scales)
    scales[:] = 5
    values = settings.constructor_config()
    values["relation_scales_m"][0][0] = 7
    assert settings.relation_scales_m == ((1.0, 1.0),) * 4
    assert "rounds" not in settings.subgraph_kwargs()


def test_exact_index_preserves_physical_predictions_context_and_diagnostics():
    model, domain, preprocessing, amplitudes = _inputs()
    outputs = [
        predict_relational_trace_graph(
            model,
            domain,
            preprocessing,
            graph_settings=_settings(neighbor_search=search),
            amplitudes=amplitudes,
            query_batch_size=2,
            measure_resources=False,
        )
        for search in ("brute_force", "exact_index")
    ]
    np.testing.assert_array_equal(outputs[0].prediction, outputs[1].prediction)
    np.testing.assert_array_equal(outputs[0].has_observed_context, outputs[1].has_observed_context)
    assert outputs[0].diagnostics == outputs[1].diagnostics
