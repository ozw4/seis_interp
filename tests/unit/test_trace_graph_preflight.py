"""Bounded CPU preflight reports measured limits without changing the model."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.evaluation.trace_graph_diagnostic_metrics import TraceGraphDiagnosticBands
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training import trace_graph_preflight as preflight
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def _arguments():
    domain, training, amplitudes = make_relational_trace_domains()
    fixed = fit_trace_graph_preprocessing(
        training,
        amplitudes,
        position_scale_m=10,
        offset_scale_m=10,
        azimuth_min_offset_m=0.1,
    )
    torch.manual_seed(3)
    model = RelationalTraceGraphInterpolator(width=8, relation_fusion="learned_gate")
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(std=0.05)
    settings = TraceGraphSettings(((1.0, 1.0),) * 4, neighbors_per_relation=2)
    return model, domain, fixed, {"graph_settings": settings, "amplitudes": amplitudes}


@pytest.mark.parametrize("neighbor_search", ["brute_force", "exact_index"])
def test_cpu_preflight_scores_same_queries_for_model_zero_idw_and_fixed_bands(neighbor_search):
    model, domain, fixed, kwargs = _arguments()
    kwargs["graph_settings"] = replace(kwargs["graph_settings"], neighbor_search=neighbor_search)
    report = preflight.run_trace_graph_preflight(
        model,
        domain,
        fixed,
        **kwargs,
        query_trace_ids=domain.trace_ids[domain.query_mask],
        query_limit=2,
        evaluate_baselines=True,
        bands=TraceGraphDiagnosticBands(time_s=(0.025,)),
    )
    assert report["status"] == "success" and not report["blockers"]
    assert report["geometry"]["query_count"] == 2
    assert report["resources"]["cuda_max_memory_allocated_bytes"] is None
    assert report["geometry_graph_seconds"] > 0
    timings = report["prediction_diagnostics"]["timings"]
    assert all(value > 0 for value in timings.values())
    model_metrics = report["model_metrics"]["evaluation_target"]
    for baseline in ("zero", "idw"):
        scores = report["baselines"][baseline]["evaluation_target"]
        for name in ("trace_count", "sample_count", "reference_energy"):
            assert scores[name] == model_metrics[name]
    assert report["error_bands"]["sample_count"] == model_metrics["sample_count"]
    json.dumps(report, allow_nan=False)


def test_timing_instrumentation_preserves_predictions_gradients_and_rng():
    model, domain, fixed, kwargs = _arguments()
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    states = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    before_rng = torch.get_rng_state().clone()
    first = predict_relational_trace_graph(model, domain, fixed, **kwargs, measure_resources=False)
    second = predict_relational_trace_graph(model, domain, fixed, **kwargs, measure_resources=True)
    np.testing.assert_array_equal(first.prediction, second.prediction)
    assert "timings" not in first.diagnostics
    assert second.diagnostics["gate_relation_names"] == [
        "source",
        "receiver",
        "cmp",
        "offset_azimuth",
    ]
    assert first.diagnostics["gate_sum"] == second.diagnostics["gate_sum"]
    torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
    assert model.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, states[name], atol=0, rtol=0)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)


def test_measured_graph_limit_blocks_before_forward_or_evaluation(monkeypatch):
    model, domain, fixed, kwargs = _arguments()
    monkeypatch.setattr(
        preflight,
        "predict_relational_trace_graph",
        lambda *a, **kw: pytest.fail("blocked geometry must not run prediction"),
    )
    report = preflight.run_trace_graph_preflight(
        model,
        domain,
        fixed,
        **kwargs,
        query_trace_ids=np.array([20]),
        max_graph_seconds=1e-20,
        evaluate_baselines=True,
    )
    assert report["status"] == "blocked" and report["blocked_stage"] == "geometry"
    assert report["blockers"][0]["reason"] == "measured_limit_exceeded"
    assert report["blockers"][0]["measured"] > report["blockers"][0]["limit"]
    assert report["prediction_diagnostics"] is None
    assert "baselines" not in report


def test_exact_index_construction_time_counts_toward_geometry_budget(monkeypatch):
    model, domain, fixed, kwargs = _arguments()
    kwargs["graph_settings"] = replace(kwargs["graph_settings"], neighbor_search="exact_index")
    native_builder = preflight.FixedTraceGraphSubgraphBuilder
    clock = [0.0]

    def build_index(*args, **settings):
        clock[0] += 7.0
        return native_builder(*args, **settings)

    monkeypatch.setattr(preflight, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(preflight, "FixedTraceGraphSubgraphBuilder", build_index)
    monkeypatch.setattr(
        preflight,
        "build_trace_graph_subgraph",
        lambda *a, **kw: pytest.fail("exact_index preflight must use the indexed builder"),
    )
    monkeypatch.setattr(
        preflight,
        "predict_relational_trace_graph",
        lambda *a, **kw: pytest.fail("index construction exceeded the geometry budget"),
    )
    report = preflight.run_trace_graph_preflight(
        model,
        domain,
        fixed,
        **kwargs,
        query_trace_ids=np.array([20]),
        max_graph_seconds=3.0,
    )
    assert report["status"] == "blocked" and report["blocked_stage"] == "geometry"
    assert report["geometry_graph_seconds"] == 7.0
    assert report["blockers"][0]["measured"] == 7.0
    assert report["prediction_diagnostics"] is None


@pytest.mark.parametrize(
    "stage,error", [("geometry", MemoryError), ("prediction", torch.cuda.OutOfMemoryError)]
)
def test_resource_allocation_failure_is_reported_as_blocker(monkeypatch, stage, error):
    model, domain, fixed, kwargs = _arguments()

    def failed(*args, **kwargs):
        raise error("test allocation failed")

    name = "build_trace_graph_subgraph" if stage == "geometry" else "predict_relational_trace_graph"
    monkeypatch.setattr(preflight, name, failed)
    report = preflight.run_trace_graph_preflight(
        model,
        domain,
        fixed,
        **kwargs,
        query_trace_ids=np.array([20]),
    )
    assert report["status"] == "blocked" and report["blocked_stage"] == stage
    assert report["blockers"][0]["reason"] == error.__name__
    assert report["blockers"][0]["measured"] is None
    json.dumps(report, allow_nan=False)


def test_query_limit_is_enforced_before_graph_construction(monkeypatch):
    model, domain, fixed, kwargs = _arguments()
    monkeypatch.setattr(
        preflight,
        "build_trace_graph_subgraph",
        lambda *a, **kw: pytest.fail("must enforce explicit query limit"),
    )
    with pytest.raises(ValueError, match="query_limit"):
        preflight.run_trace_graph_preflight(
            model,
            domain,
            fixed,
            **kwargs,
            query_trace_ids=np.array([20, 30]),
            query_limit=1,
        )


def test_empty_query_prediction_still_checks_model_graph_compatibility():
    model, domain, fixed, kwargs = _arguments()
    settings = TraceGraphSettings(
        ((1.0, 1.0),) * 4,
        topology="single_4d",
        common_distance_scales_m=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="single_4d|untyped"):
        predict_relational_trace_graph(
            model,
            domain,
            fixed,
            graph_settings=settings,
            amplitudes=kwargs["amplitudes"],
            query_trace_ids=np.array([], dtype=np.int64),
        )
