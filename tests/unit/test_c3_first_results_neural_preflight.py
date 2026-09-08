"""Episode visibility/RNG isolation and scope-aware resource extrapolation."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training import c3_first_results_neural_preflight as preflight
from seis_interp.training.trace_graph_episodes import TraceGraphEpisodeGenerator
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def test_disposable_training_batch_uses_episode_support_and_restores_rng(monkeypatch):
    _, training, amplitudes = make_relational_trace_domains()
    training = replace(training, pool="all_train_traces")
    fixed = fit_trace_graph_preprocessing(
        training, amplitudes, position_scale_m=10, offset_scale_m=10, azimuth_min_offset_m=0.1
    )
    options = {
        "random_seed": 20260908,
        "episode_kind_probabilities": {"random_trace": 1.0},
        "missing_fractions": (0.8,),
        "query_batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    episode = TraceGraphEpisodeGenerator(
        training,
        random_seed=20260908,
        kind_probabilities={"random_trace": 1.0},
        missing_fractions=(0.8,),
    ).next_episode()
    native_inputs = preflight.MaskedTraceSource.inputs
    captured = []

    def inputs(source, plan):
        support = plan.trace_ids[plan.observed_mask]
        assert np.isin(support, training.trace_ids[episode.visible_mask]).all()
        assert not np.isin(support, episode.hidden_trace_ids).any()
        captured.append(support.copy())
        return native_inputs(source, plan)

    monkeypatch.setattr(preflight.MaskedTraceSource, "inputs", inputs)
    before_rng = torch.get_rng_state().clone()
    reports = [
        preflight.measure_c3_first_results_graph_training_batch(
            training,
            fixed,
            model_config={"width": 8, "attention_width": 8},
            graph_settings=TraceGraphSettings(((2.0, 2.0),) * 4, neighbors_per_relation=2),
            training_options=options,
            device=torch.device("cpu"),
            amplitudes=amplitudes,
        )
        for _ in range(2)
    ]
    assert captured
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert reports[0]["query_trace_ids"] == episode.query_trace_ids.tolist()
    assert reports[0]["normalized_training_loss"] == reports[1]["normalized_training_loss"]
    assert reports[0]["visible_trace_count"] == 1
    assert reports[0]["sample_count"] == 10
    assert not reports[0]["state_reused_by_pilot"]
    json.dumps(reports, allow_nan=False)


def test_graph_estimate_counts_validation_best_baseline_and_final_scopes():
    measured = {
        "status": "success",
        "total_case_query_count": 33,
        "sampled_query_trace_ids": list(range(8)),
        "prediction_diagnostics": {"timings": {"graph_build_seconds": 2, "forward_seconds": 1}},
    }
    report = preflight.estimate_c3_first_results_graph_budget(
        {"timings": {"graph_build_seconds": 1, "forward_seconds": 2}},
        measured,
        max_steps=200,
        validation_interval=200,
        validation_query_batch_size=8,
    )
    assert report["validation_batch_count"] == 5
    assert report["estimated_validation_prediction_seconds"] == 15
    assert report["trainer_validation_passes"] == 1
    assert report["native_baseline_pass_equivalent_allowance"] == 0
    assert report["estimated_train_action_seconds"] == 630
    assert report["estimated_final_predict_action_seconds"] == 15
    assert report["estimated_combined_seconds"] == 645
    measured["sampled_query_trace_ids"] = [0]
    with pytest.raises(ValueError, match="full validation batch"):
        preflight.estimate_c3_first_results_graph_budget(
            {"timings": {}},
            measured,
            max_steps=200,
            validation_interval=200,
            validation_query_batch_size=8,
        )
