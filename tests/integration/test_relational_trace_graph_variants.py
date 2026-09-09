"""Fixed graph/model ablations through real training and frozen checkpoints."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
    save_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.relational_trace_graph_trainer import train_relational_trace_graph
from tests.fixtures.trace_graph_training import make_trace_graph_training_domains

VARIANTS = [
    ({"relation_fusion": "mean"}, {}),
    ({"relation_fusion": "learned_gate"}, {}),
    ({"method_variant": "plain_gcn_row_normalized"}, {}),
    ({"method_variant": "untyped_edge_conditioned"}, {}),
    ({"method_variant": "untyped_edge_conditioned"}, {"topology": "single_4d"}),
    *[
        ({}, {"excluded_relation": name})
        for name in ("source", "receiver", "cmp", "offset_azimuth")
    ],
    ({"explicit_azimuth_features": False}, {}),
    (
        {"relation_fusion": "learned_gate", "amplitude_mode": "observed_trace_rms"},
        {"neighbor_search": "exact_index"},
    ),
    ({"max_edge_time_shift_samples": 4}, {}),
    (
        {
            "relation_fusion": "learned_gate",
            "amplitude_mode": "observed_trace_rms",
            "max_edge_time_shift_samples": 4,
        },
        {"neighbor_search": "exact_index"},
    ),
]


def _setup(model_options, graph_options):
    train, validation, values = make_trace_graph_training_domains()
    # Distinct physical signals prevent equal-neighbor or zero-head agreement.
    values *= np.linspace(0.6, 1.8, len(values), dtype=np.float32)[:, None]
    fixed = fit_trace_graph_preprocessing(
        train, values, position_scale_m=2, offset_scale_m=2, azimuth_min_offset_m=0.01
    )
    torch.manual_seed(18)
    model = RelationalTraceGraphInterpolator(
        width=8,
        attention_width=4,
        relation_embedding_dim=3,
        stem_kernel_size=3,
        temporal_kernel_size=3,
        **model_options,
    )
    settings = TraceGraphSettings(
        ((2.0, 2.0),) * 4,
        neighbors_per_relation=2,
        common_distance_scales_m=(2.0, 2.0),
        **graph_options,
    )
    return model, settings, train, validation, values, fixed


@pytest.mark.parametrize("model_options,graph_options", VARIANTS)
def test_all_ablations_round_trip_nonzero_frozen_prediction(model_options, graph_options, tmp_path):
    model, settings, _, validation, values, fixed = _setup(model_options, graph_options)
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(std=0.15)
        if model_options.get("max_edge_time_shift_samples", 0):
            model.edge_time_shift_weights.copy_(torch.tensor([0.125, -0.05, 0.1, -0.2]))
    before = predict_relational_trace_graph(
        model, validation, fixed, graph_settings=settings, amplitudes=values, query_batch_size=4
    )
    assert np.abs(before.prediction).max() > 1e-4
    checkpoint = tmp_path / "best.pt"
    save_relational_trace_graph_checkpoint(
        checkpoint,
        model_config=model.constructor_config(),
        state_dict=model.state_dict(),
        preprocessing=fixed,
        graph_settings=settings,
        training_mask={
            "kinds": ["random_trace"],
            "kind_probabilities": [1.0],
            "missing_fractions": [0.5],
        },
        training_provenance={"source_inputs_lock": {"partition": "train"}},
        training_random_seed=18,
        checkpoint_role="best_validation",
        global_step=2,
        selection_metrics={"evaluation_target": {"error_energy": 12.0}},
    )
    loaded = load_relational_trace_graph_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["relation_names"] == (
        ["untyped"]
        if settings.topology == "single_4d"
        else ["source", "receiver", "cmp", "offset_azimuth"]
    )
    assert loaded.model.constructor_config() == model.constructor_config()
    assert loaded.graph_settings == settings
    after = predict_relational_trace_graph(
        loaded.model,
        validation,
        loaded.preprocessing,
        graph_settings=loaded.graph_settings,
        amplitudes=values,
        query_batch_size=1,
    )
    np.testing.assert_allclose(after.prediction, before.prediction, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(after.has_observed_context, before.has_observed_context)
    json.dumps(after.diagnostics, allow_nan=False)


@pytest.mark.parametrize("method_variant", ["plain_gcn_row_normalized", "untyped_edge_conditioned"])
def test_comparison_models_learn_in_existing_masked_trainer(method_variant):
    model, settings, train, validation, values, fixed = _setup(
        {"method_variant": method_variant}, {}
    )
    result = train_relational_trace_graph(
        model,
        train,
        validation,
        fixed,
        graph_settings=settings,
        episode_kind_probabilities={"random_trace": 1.0},
        missing_fractions=[0.5],
        random_seed=18,
        max_steps=3,
        query_batch_size=4,
        validation_interval=2,
        learning_rate=0.01,
        weight_decay=0,
        training_amplitudes=values,
        validation_amplitudes=values,
    )
    prediction = predict_relational_trace_graph(
        model, validation, fixed, graph_settings=settings, amplitudes=values
    )
    assert result.steps_completed == 3 and result.best_step in (2, 3)
    assert np.abs(prediction.prediction).max() > 1e-4
    assert model.encoder.stem.weight.grad.abs().sum() > 0
    assert model.decoder.head[-1].weight.abs().sum() > 0


@pytest.mark.parametrize("max_shift", [0, 4])
def test_observed_rms_model_trains_with_physical_mse_and_responds_to_visible_gain(max_shift):
    histories = []
    for amplitude_mode in ("train_global_rms", "observed_trace_rms"):
        model, settings, train, validation, values, fixed = _setup(
            {
                "relation_fusion": "learned_gate",
                "amplitude_mode": amplitude_mode,
                "max_edge_time_shift_samples": max_shift,
            },
            {"neighbor_search": "exact_index"},
        )
        result = train_relational_trace_graph(
            model,
            train,
            validation,
            fixed,
            graph_settings=settings,
            episode_kind_probabilities={"random_trace": 1.0},
            missing_fractions=[0.5],
            random_seed=18,
            max_steps=6,
            query_batch_size=3,
            validation_interval=3,
            learning_rate=0.01,
            weight_decay=0,
            training_amplitudes=values,
            validation_amplitudes=values,
        )
        histories.append(result.training_history)
    # Both zero-initialized decoders start from the same global-normalized
    # physical errors. The new mode does not replace the objective by unit RMS MSE.
    assert histories[0][0]["batch_loss"] == histories[1][0]["batch_loss"]
    assert model.encoder.stem.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.rounds[0].gamma[0].weight.grad).all()
    if max_shift:
        assert torch.isfinite(model.edge_time_shift_weights.grad).all()
        assert model.edge_time_shift_weights.grad.abs().sum() > 0
        assert model.edge_time_shift_weights.abs().sum() > 0
    original = predict_relational_trace_graph(
        model, validation, fixed, graph_settings=settings, amplitudes=values, query_batch_size=3
    )
    assert np.abs(original.prediction).max() > 1e-4
    changed = values.copy()
    changed[validation.array_rows[validation.observed_mask]] *= 3.0
    changed[validation.array_rows[validation.query_mask]] = np.nan
    scaled = predict_relational_trace_graph(
        model, validation, fixed, graph_settings=settings, amplitudes=changed, query_batch_size=1
    )
    np.testing.assert_allclose(scaled.prediction, original.prediction * 3.0, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(scaled.has_observed_context, original.has_observed_context)
