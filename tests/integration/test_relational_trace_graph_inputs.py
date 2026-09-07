"""Verified native artifacts through fixed preprocessing and query prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.data.trace_graph_domain import (
    TraceGraphDomain,
    load_benchmark_trace_graph_domain,
    load_training_trace_graph_domain,
)
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_subgraphs import TraceGraphPlan, build_trace_graph_subgraph
from tests.fixtures.benchmark_case_artifacts import prepare_benchmark_case_artifacts


def _plan(
    domain: TraceGraphDomain, query_rows: np.ndarray, *, add_remote_query: bool = False
) -> TraceGraphPlan:
    source = domain.source_xy_m[query_rows]
    receiver = domain.receiver_xy_m[query_rows]
    query_ids = domain.trace_ids[query_rows]
    if add_remote_query:
        source = np.concatenate((source, [[1e6, 2e6]]))
        receiver = np.concatenate((receiver, [[1e6 + 100, 2e6 + 200]]))
        query_ids = np.append(query_ids, domain.trace_ids.max() + 1000)
    query_geometry = compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.1)
    observed_geometry = compute_trace_graph_geometry(
        domain.source_xy_m, domain.receiver_xy_m, azimuth_min_offset_m=0.1
    )
    return build_trace_graph_subgraph(
        query_geometry,
        query_ids,
        observed_geometry,
        domain.trace_ids,
        domain.observed_mask,
        rounds=2,
        relation_scales_m=np.full((4, 2), 1000.0),
        neighbors_per_relation=1,
        candidate_chunk_size=1,
    )


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
def test_native_domains_to_model_preserve_queries_and_read_only_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fusion: str
) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    case = processed / "cases" / "test"
    prepare_benchmark_case(interim, processed, mask, case, case_id="test")
    training = load_training_trace_graph_domain(
        interim_dir=interim,
        processed_dir=processed,
        pool="all_train_traces",
        time_samples=(1, 4),
    )
    benchmark = load_benchmark_trace_graph_domain(
        interim_dir=interim,
        processed_dir=processed,
        mask_dir=mask,
        case_dir=case,
        time_samples=(1, 4),
    )
    fixed = fit_trace_graph_preprocessing(
        training,
        position_scale_m=1000,
        offset_scale_m=1000,
        azimuth_min_offset_m=0.1,
        row_chunk_size=2,
    )
    assert fixed.fit_domain["inputs_lock"]["partition"] == "train"
    assert benchmark.inputs_lock["partition"] == "test"
    assert not np.intersect1d(training.trace_ids, benchmark.trace_ids).size

    reads: list[tuple[np.ndarray, slice]] = []
    original_getitem = np.memmap.__getitem__

    def record_rows(array: np.memmap, selection: tuple[np.ndarray, slice]) -> np.ndarray:
        rows, times = selection
        reads.append((rows.copy(), times))
        return original_getitem(array, selection)

    monkeypatch.setattr(np.memmap, "__getitem__", record_rows)
    source = MaskedTraceSource(benchmark, fixed)
    torch.manual_seed(43)
    model = RelationalTraceGraphInterpolator(
        width=8,
        stem_kernel_size=3,
        temporal_kernel_size=3,
        attention_width=4,
        relation_embedding_dim=3,
        relation_fusion=fusion,
    ).eval()
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(std=0.2)
        if fusion == "learned_gate":
            for block in model.rounds:
                block.relation_gate[-1].weight.normal_(std=0.2)

    plans = []

    def predict(plan: TraceGraphPlan) -> tuple[torch.Tensor, torch.Tensor]:
        plans.append(plan)
        inputs = source.inputs(plan)
        assert inputs.waveforms.shape[1] == 3
        assert torch.equal(
            inputs.waveforms[inputs.query_indices], torch.zeros(len(plan.query_indices), 3)
        )
        assert source.preprocessing is fixed
        with torch.no_grad():
            return model(inputs)

    query_rows = benchmark.query_indices
    assert len(query_rows) == 2
    joint, context = predict(_plan(benchmark, query_rows))
    split = torch.cat([predict(_plan(benchmark, row[None]))[0] for row in query_rows])
    reversed_prediction, _ = predict(_plan(benchmark, query_rows[::-1]))
    augmented, augmented_context = predict(_plan(benchmark, query_rows, add_remote_query=True))

    assert joint.shape == (2, 3) and context.all()
    assert torch.all(joint.abs().amax(dim=1) > 1e-4)
    torch.testing.assert_close(joint, split, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(joint.flip(0), reversed_prediction, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(joint, augmented[:-1], atol=1e-6, rtol=1e-5)
    assert torch.equal(augmented_context, torch.tensor([True, True, False]))
    assert torch.equal(augmented[-1], torch.zeros(3))

    assert len(reads) == len(plans)
    for (rows, times), plan in zip(reads, plans, strict=True):
        # Native stable IDs are the original array rows, including support leaves.
        np.testing.assert_array_equal(rows, plan.trace_ids[plan.observed_mask])
        assert np.isin(rows, benchmark.array_rows[benchmark.observed_mask]).all()
        assert not np.isin(rows, benchmark.array_rows[benchmark.query_mask]).any()
        assert times == slice(1, 4)
