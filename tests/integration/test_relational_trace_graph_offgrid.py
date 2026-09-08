"""Small analytic-coordinate smoke; no numerical propagation or C3 accuracy claim."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.evaluation.physical_amplitude_metrics import (
    physical_amplitude_energies,
    physical_amplitude_target_metrics,
)
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
    save_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.relational_trace_graph_trainer import train_relational_trace_graph
from tests.fixtures.analytic_trace_field import (
    analytic_offgrid_query_coordinates,
    analytic_trace_waveforms,
    make_analytic_trace_domains,
)


def test_analytic_targets_are_evaluated_at_each_query_coordinate() -> None:
    source, receiver = analytic_offgrid_query_coordinates()
    time_s = np.arange(33, dtype=np.float64) * 0.004
    reference = analytic_trace_waveforms(source, receiver, time_s)
    moved = analytic_trace_waveforms(source + [4.0, -2.0], receiver + [4.0, -2.0], time_s)
    assert np.max(np.abs(reference - moved)) > 0.05
    for row in range(len(source)):
        np.testing.assert_array_equal(
            reference[row],
            analytic_trace_waveforms(source[row : row + 1], receiver[row : row + 1], time_s)[0],
        )


@pytest.mark.parametrize("mask_kind", ["random_trace", "random_whole_ffid"])
def test_analytic_offgrid_train_checkpoint_predict_physical_targets(tmp_path, mask_kind) -> None:
    training, validation, values = make_analytic_trace_domains(mask_kind)
    assert len(set(np.unique(training.ffids, return_counts=True)[1])) > 1
    if mask_kind == "random_whole_ffid":
        assert not np.intersect1d(
            validation.ffids[validation.query_mask], validation.ffids[validation.observed_mask]
        ).size
    fixed = fit_trace_graph_preprocessing(
        training,
        values,
        position_scale_m=1000.0,
        offset_scale_m=1000.0,
        azimuth_min_offset_m=1.0,
    )
    settings = TraceGraphSettings(((160.0, 640.0), (640.0, 160.0)) * 2, neighbors_per_relation=2)
    torch.manual_seed(42)
    model = RelationalTraceGraphInterpolator(
        width=8, attention_width=4, relation_embedding_dim=3, relation_fusion="learned_gate"
    )
    trained = train_relational_trace_graph(
        model,
        training,
        validation,
        fixed,
        graph_settings=settings,
        episode_kind_probabilities={mask_kind: 1.0},
        missing_fractions=(0.5,),
        random_seed=42,
        max_steps=3,
        query_batch_size=4,
        validation_interval=1,
        learning_rate=0.01,
        weight_decay=0.0,
        training_amplitudes=values,
        validation_amplitudes=values,
    )
    assert trained.steps_completed == 3
    assert all(row["kind"] == mask_kind for row in trained.episode_history)
    path = tmp_path / "best.pt"
    save_relational_trace_graph_checkpoint(
        path,
        model_config=model.constructor_config(),
        state_dict=trained.best_state_dict,
        preprocessing=fixed,
        graph_settings=settings,
        training_mask={
            "kinds": [mask_kind],
            "kind_probabilities": [1.0],
            "missing_fractions": [0.5],
        },
        training_provenance={
            "fixture": "analytic_ricker_arrivals",
            "training_inputs_lock": training.inputs_lock,
        },
        training_random_seed=42,
        checkpoint_role="best_validation",
        global_step=trained.best_step,
        selection_metrics=trained.best_validation_metrics,
    )
    checkpoint = load_relational_trace_graph_checkpoint(path)
    source, receiver = analytic_offgrid_query_coordinates()
    # No stored row or target waveform enters arbitrary-coordinate inference.
    observed_domain = replace(training, array_rows=None)
    assert observed_domain.array_rows is None
    arguments = {
        "graph_settings": checkpoint.graph_settings,
        "observed_waveforms": values[training.array_rows],
        "query_trace_ids": np.arange(-4, 0, dtype=np.int64),
        "query_source_xy_m": source,
        "query_receiver_xy_m": receiver,
    }
    prediction = predict_relational_trace_graph(
        checkpoint.model, observed_domain, checkpoint.preprocessing, query_batch_size=4, **arguments
    )
    split = predict_relational_trace_graph(
        checkpoint.model, observed_domain, checkpoint.preprocessing, query_batch_size=1, **arguments
    )
    assert np.max(np.abs(prediction.prediction[:3])) > 1e-4
    np.testing.assert_allclose(split.prediction, prediction.prediction, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(prediction.has_observed_context, [True, True, True, False])
    np.testing.assert_array_equal(prediction.prediction[-1], np.zeros(len(training.time_s)))
    reference = analytic_trace_waveforms(source, receiver, training.time_s)
    assert np.square(reference[-1].astype(np.float64)).sum() > 0.1
    reference_energy, error_energy = physical_amplitude_energies(reference, prediction.prediction)
    metrics = physical_amplitude_target_metrics(
        trace_count=len(source),
        sample_count=reference.size,
        reference_energy=reference_energy,
        error_energy=error_energy,
    )
    assert metrics["trace_count"] == 4
    assert metrics["sample_count"] == 4 * 33
    assert metrics["reference_energy"] > 0
    assert metrics["error_energy"] >= float(np.square(reference[-1].astype(np.float64)).sum())
    assert metrics["snr_status"] == "finite"
    (tmp_path / "offgrid_metrics.json").write_text(json.dumps(metrics, allow_nan=False) + "\n")
    np.save(tmp_path / "offgrid_prediction.npy", prediction.prediction, allow_pickle=False)
