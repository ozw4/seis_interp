"""Prediction, reference evaluation and selected-domain boundaries stay distinct."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.data.trace_graph_domain import (
    load_benchmark_trace_graph_domain,
    load_training_trace_graph_domain,
)
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import (
    TraceGraphPreprocessing,
    fit_trace_graph_preprocessing,
)
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from tests.fixtures.c3_volume_run_artifacts import prepare_c3_volume_run_artifacts
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def test_target_row_reader_is_used_only_after_prediction_completes() -> None:
    domain, training, amplitudes = make_relational_trace_domains()
    preprocessing = fit_trace_graph_preprocessing(
        training,
        amplitudes,
        position_scale_m=1.0,
        offset_scale_m=2.0,
        azimuth_min_offset_m=0.1,
    )
    phase = "prediction"
    reads: list[tuple[str, np.ndarray]] = []

    class PhaseRows:
        shape, ndim, dtype = amplitudes.shape, amplitudes.ndim, amplitudes.dtype

        def __getitem__(self, selection: tuple[np.ndarray, slice]) -> np.ndarray:
            rows, time = selection
            role = domain.observed_mask if phase == "prediction" else domain.query_mask
            assert np.all(np.isin(rows, domain.array_rows[role]))
            assert time == slice(*domain.time_samples)
            reads.append((phase, rows.copy()))
            return amplitudes[selection]

    torch.manual_seed(3)
    model = RelationalTraceGraphInterpolator(
        width=8,
        message_passing_rounds=2,
        attention_width=4,
        relation_embedding_dim=3,
        temporal_dilations=(1, 1),
    )
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(std=0.1)
    result = predict_relational_trace_graph(
        model,
        domain,
        preprocessing,
        graph_settings=TraceGraphSettings(((1.0, 1.0),) * 4, neighbors_per_relation=2),
        amplitudes=PhaseRows(),
        query_batch_size=1,
    )
    assert np.any(result.prediction != 0)
    assert reads and all(name == "prediction" for name, _ in reads)
    phase = "evaluation"
    metrics = evaluate_trace_graph_prediction(
        result.prediction,
        domain,
        query_trace_ids=result.query_trace_ids,
        has_observed_context=result.has_observed_context,
        amplitudes=PhaseRows(),
    )
    assert metrics["evaluation_target"]["trace_count"] == 2
    assert reads[-1][0] == "evaluation"
    np.testing.assert_array_equal(reads[-1][1], domain.array_rows[domain.query_mask])


def test_selected_volume_cannot_replenish_context_from_crop_exterior_or_training(
    tmp_path: Path,
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path, time_sample_count=3)
    paths = {
        "interim_dir": artifacts.interim,
        "processed_dir": artifacts.processed,
        "mask_dir": artifacts.mask,
        "case_dir": artifacts.case,
    }
    cropped = load_benchmark_trace_graph_domain(**paths, volume_dir=artifacts.volume)
    native = load_benchmark_trace_graph_domain(**paths, time_samples=(0, 3))
    training = load_training_trace_graph_domain(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        pool="all_train_traces",
        time_samples=(0, 3),
    )
    preprocessing = TraceGraphPreprocessing(
        amplitude_scale=1.0,
        midpoint_origin_m=(0.0, 0.0),
        position_scale_m=100.0,
        offset_scale_m=100.0,
        azimuth_min_offset_m=0.1,
        time_s=tuple(cropped.time_s),
        fit_domain={"partition": "train"},
    )
    source = MaskedTraceSource(cropped, preprocessing)
    exterior = native.trace_ids[
        native.observed_mask & ~np.isin(native.trace_ids, cropped.trace_ids)
    ]
    assert exterior.size and training.trace_ids.size
    for disallowed in (exterior[:1], training.trace_ids[:1]):
        with pytest.raises(ValueError, match="outside the observed domain"):
            source.read_observed_rows(disallowed)
    with pytest.raises(ValueError, match="time_s"):
        MaskedTraceSource(
            native,
            preprocessing=TraceGraphPreprocessing(
                amplitude_scale=1.0,
                midpoint_origin_m=(0.0, 0.0),
                position_scale_m=100.0,
                offset_scale_m=100.0,
                azimuth_min_offset_m=0.1,
                time_s=(0.0, 0.01, 0.02),
                fit_domain={"partition": "train"},
            ),
        )
