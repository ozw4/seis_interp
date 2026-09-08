from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.c3_benchmark_inputs import (
    load_c3_benchmark_graph_domain,
    load_c3_benchmark_supervised_source,
    load_c3_benchmark_training_graph_domain,
    load_c3_benchmark_volume_inputs,
)
from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.models.siren import Siren
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.c3_volume_siren_data import (
    build_c3_volume_siren_data,
    build_c3_volume_siren_sampler,
)
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume
from seis_interp.training.fixed_step_siren import train_siren_fixed_steps
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)
from tests.fixtures.relational_trace_graph import make_relational_trace_plan

DIMENSIONS = C3BenchmarkDimensions((0, 4), (1, 2), (4, 2, 2, 2, 4))


def test_five_method_inputs_target_poisoning_and_query_batch_invariance(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _check_conformance(tmp_path)
    finally:
        torch.set_num_threads(old_threads)


def _check_conformance(tmp_path):
    loaded = []
    fit = {
        "time": [0, 4],
        "source_line": [0, 1],
        "shot_in_line": [0, 1],
        "relative_receiver_x": [3, 5],
        "relative_receiver_y": [32, 36],
    }
    selection = {**fit, "shot_in_line": [1, 2]}
    for name in ("original", "poisoned"):
        interim = make_benchmark_interim(tmp_path / name)
        if name == "poisoned":
            mask = pd.read_parquet(
                tmp_path / "original/suite/masks/test_random_trace/observation_mask.parquet"
            )
            rows = mask.loc[mask.observation_role.eq("evaluation_target"), "array_row"].to_numpy()
            amplitudes = np.load(interim / "amplitudes.npy")
            amplitudes[rows, :4] = 99.0
            np.save(interim / "amplitudes.npy", amplitudes)
        suite_dir = tmp_path / name / "suite"
        prepare_c3_benchmark_artifacts(
            interim,
            suite_dir,
            config=synthetic_benchmark_config(),
            inputs={"cases": synthetic_benchmark_cases()},
            dimensions=DIMENSIONS,
        )
        volume = load_c3_benchmark_volume_inputs(
            suite_dir, "test_random_trace", dimensions=DIMENSIONS
        )
        graph = load_c3_benchmark_graph_domain(
            suite_dir,
            "test_random_trace",
            volume_dir=suite_dir / "volumes/test_random_trace",
            dimensions=DIMENSIONS,
        )
        training = load_c3_benchmark_training_graph_domain(suite_dir, dimensions=DIMENSIONS)
        teacher = load_c3_benchmark_supervised_source(
            suite_dir, fit_region=fit, selection_region=selection, dimensions=DIMENSIONS
        )
        preprocessing = fit_trace_graph_preprocessing(
            training, position_scale_m=1000, offset_scale_m=1000, azimuth_min_offset_m=0.1
        )
        siren = build_c3_volume_siren_data(volume.observed_volume, volume.index_table)
        dense = volume.observed_volume
        assert np.any(np.all(dense.values[:, dense.observed_trace_mask] == 0, axis=0))
        assert np.array_equal(dense.time_s, graph.time_s)
        assert dense.time_s[0] == 0.125
        assert set(dense.array_rows[dense.observed_trace_mask]) == set(
            graph.array_rows[graph.observed_mask]
        )
        assert set(dense.array_rows[dense.evaluation_target_trace_mask]) == set(
            graph.array_rows[graph.query_mask]
        )
        assert volume.case["case_id"] == graph.inputs_lock["benchmark_case"]["case_id"]
        actual = volume.index_table.set_index("array_row").loc[graph.array_rows]
        np.testing.assert_array_equal(
            actual[["source_x_m", "source_y_m"]].to_numpy(), graph.source_xy_m
        )
        np.testing.assert_array_equal(
            actual[["relative_receiver_x_m", "relative_receiver_y_m"]].to_numpy()
            + graph.source_xy_m,
            graph.receiver_xy_m,
        )
        # Partition observations outside the crop cannot become graph senders.
        mask = pd.read_parquet(suite_dir / "masks/test_random_trace/observation_mask.parquet")
        outside = set(mask.loc[mask.observation_role.eq("observed"), "array_row"]) - set(
            graph.array_rows
        )
        assert outside
        plan = make_relational_trace_plan(graph)
        assert set(plan.trace_ids[plan.observed_mask]) <= set(graph.trace_ids[graph.observed_mask])
        assert not set(plan.trace_ids) & outside
        graph_inputs = MaskedTraceSource(graph, preprocessing).inputs(plan)
        loaded.append((volume, graph, preprocessing, siren, teacher, graph_inputs, interim))
    first, second = loaded
    np.testing.assert_array_equal(first[0].observed_volume.values, second[0].observed_volume.values)
    assert first[2].amplitude_scale == second[2].amplitude_scale
    assert first[2].midpoint_origin_m == second[2].midpoint_origin_m
    assert first[4].amplitude_rms == second[4].amplitude_rms
    for field in fields(first[3]):
        a, b = getattr(first[3], field.name), getattr(second[3], field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    for field in fields(first[5]):
        a, b = getattr(first[5], field.name), getattr(second[5], field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        else:
            assert a == b
    siren_states = []
    for item in loaded:
        torch.manual_seed(17)
        model = Siren(input_features=6, hidden_width=8, hidden_layers=1)
        sampler = build_c3_volume_siren_sampler(item[3], random_seed=19)
        train_siren_fixed_steps(
            model,
            sampler,
            device="cpu",
            learning_rate=1e-3,
            batch_size=8,
            max_steps=3,
            report_interval=3,
        )
        siren_states.append(model.state_dict())
    for key, value in siren_states[0].items():
        assert torch.equal(value, siren_states[1][key])
    torch.manual_seed(17)
    cnn = CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=1)
    predictions = [
        predict_ccnet5d_volume(
            cnn,
            item[0].observed_volume,
            amplitude_rms=item[4].amplitude_rms,
            core_shape=(4, 2, 2, 2, 4),
            device="cpu",
        ).values
        for item in loaded
    ]
    np.testing.assert_array_equal(*predictions)
    graph_model = RelationalTraceGraphInterpolator(
        width=8, attention_width=5, relation_embedding_dim=3
    )
    with torch.no_grad():
        graph_model.decoder.head[-1].weight.normal_(std=0.15)
    settings = TraceGraphSettings(
        relation_scales_m=((1000.0, 1000.0),) * 4, neighbors_per_relation=2, candidate_chunk_size=16
    )
    graph_predictions = []
    for item, batch in ((first, 3), (first, 100), (second, 3)):
        graph_predictions.append(
            predict_relational_trace_graph(
                graph_model,
                item[1],
                item[2],
                graph_settings=settings,
                query_batch_size=batch,
                device="cpu",
            )
        )
    for result in graph_predictions[1:]:
        np.testing.assert_allclose(
            result.prediction, graph_predictions[0].prediction, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_array_equal(result.query_trace_ids, graph_predictions[0].query_trace_ids)
    energies = [
        evaluate_c3_volume_prediction(
            predictions[i],
            item[0].observed_volume,
            interim_dir=item[6],
            volume_metadata=item[0].volume_metadata,
        )["evaluation_target"]["error_energy"]
        for i, item in enumerate(loaded)
    ]
    assert energies[0] != energies[1]
    with pytest.raises(ValueError, match="fixed volume_dir"):
        load_c3_benchmark_graph_domain(
            tmp_path / "original/suite", "test_random_trace", volume_dir=None, dimensions=DIMENSIONS
        )
    with pytest.raises(ValueError, match="authorized training time"):
        load_c3_benchmark_supervised_source(
            tmp_path / "original/suite",
            fit_region={**fit, "time": [0, 8]},
            selection_region=selection,
            dimensions=DIMENSIONS,
        )
