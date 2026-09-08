"""Target-only physical energies and strict-JSON trace graph evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from seis_interp.data.trace_graph_domain import TraceGraphDomain, build_trace_graph_domain
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction


class _TargetRows:
    def __init__(self, values: np.ndarray, allowed_rows: np.ndarray) -> None:
        self.values = values
        self.allowed_rows = allowed_rows
        self.shape, self.ndim, self.dtype = values.shape, values.ndim, values.dtype
        self.reads: list[tuple[np.ndarray, slice]] = []

    def __getitem__(self, selection: tuple[np.ndarray, slice]) -> np.ndarray:
        rows, time = selection
        assert np.all(np.isin(rows, self.allowed_rows)), "only target rows may be read"
        assert time == slice(1, 3), "only selected samples may be read"
        self.reads.append((rows.copy(), time))
        return self.values[selection]


def _example() -> tuple[TraceGraphDomain, np.ndarray, np.ndarray, np.ndarray, _TargetRows]:
    domain = build_trace_graph_domain(
        trace_ids=np.array([40, 10, 90, 20]),
        source_xy_m=np.column_stack((np.arange(4.0), np.zeros(4))),
        receiver_xy_m=np.column_stack((np.arange(4.0), np.ones(4))),
        ffids=np.arange(4),
        array_rows=np.array([4, 0, 6, 1]),
        observed_mask=np.array([True, False, False, False]),
        time_s=np.array([0.01, 0.02]),
        time_samples=(1, 3),
        inputs_lock={"partition": "validation"},
    )
    query_ids = np.array([90, 20, 10])
    values = np.full((7, 5), np.nan, dtype=np.float32)
    values[[6, 1, 0], 1:3] = [[3, 4], [0, 12], [1, 2]]
    prediction = np.array([[2, 4], [0, 0], [1, 1]], dtype=np.float32)
    context = np.array([True, False, True])
    return domain, query_ids, prediction, context, _TargetRows(values, np.array([6, 1, 0]))


def test_global_physical_energy_includes_zero_context_queries_and_ignores_other_rows() -> None:
    domain, ids, prediction, context, amplitudes = _example()
    assert amplitudes.reads == []
    result = evaluate_trace_graph_prediction(
        prediction,
        domain,
        query_trace_ids=ids,
        has_observed_context=context,
        amplitudes=amplitudes,
        row_chunk_size=2,
    )
    target = result["evaluation_target"]
    assert target["trace_count"] == 3
    assert target["sample_count"] == 6
    assert target["reference_energy"] == 174.0
    assert target["error_energy"] == 146.0
    assert target["snr_db"] == pytest.approx(10 * math.log10(174 / 146))
    assert target["rmse"] == pytest.approx(math.sqrt(146 / 6))
    assert target["relative_l2"] == pytest.approx(math.sqrt(146 / 174))
    assert result["with_observed_context"]["error_energy"] == 2.0
    assert result["without_observed_context"]["error_energy"] == 144.0
    assert result["zero_context_query_count"] == 1
    assert result["zero_context_query_fraction"] == 1 / 3
    np.testing.assert_array_equal(np.concatenate([read[0] for read in amplitudes.reads]), [6, 1, 0])
    json.dumps(result, allow_nan=False)


def test_energy_accumulation_is_independent_of_read_chunks_and_prediction_batch_splits() -> None:
    domain, ids, prediction, context, amplitudes = _example()

    def evaluate(selection: slice, chunk: int) -> dict[str, object]:
        return evaluate_trace_graph_prediction(
            prediction[selection],
            domain,
            query_trace_ids=ids[selection],
            has_observed_context=context[selection],
            amplitudes=amplitudes,
            row_chunk_size=chunk,
        )

    full = evaluate(slice(None), 3)
    assert full == evaluate(slice(None), 1)
    first, second = evaluate(slice(0, 1), 1), evaluate(slice(1, None), 2)
    for group in ("evaluation_target", "with_observed_context", "without_observed_context"):
        for key in ("reference_energy", "error_energy", "sample_count", "trace_count"):
            assert full[group][key] == first[group][key] + second[group][key]


@pytest.mark.parametrize("kind", ["perfect", "zero_target", "zero_both"])
def test_degenerate_snr_preserves_meaning_in_strict_json(kind: str) -> None:
    domain, ids, prediction, context, amplitudes = _example()
    if kind == "perfect":
        prediction = amplitudes.values[[6, 1, 0], 1:3].copy()
        status = "perfect_reconstruction"
    else:
        amplitudes.values[[6, 1, 0], 1:3] = 0.0
        if kind == "zero_both":
            prediction[:] = 0.0
        status = "undefined_zero_reference"
    result = evaluate_trace_graph_prediction(
        prediction,
        domain,
        query_trace_ids=ids,
        has_observed_context=context,
        amplitudes=amplitudes,
    )
    assert result["evaluation_target"]["snr_db"] is None
    assert result["evaluation_target"]["snr_status"] == status
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("context_value", [True, False])
def test_empty_context_subgroups_have_no_queries_status(context_value: bool) -> None:
    domain, ids, prediction, context, amplitudes = _example()
    context[:] = context_value
    result = evaluate_trace_graph_prediction(
        prediction,
        domain,
        query_trace_ids=ids,
        has_observed_context=context,
        amplitudes=amplitudes,
    )
    empty = result["without_observed_context" if context_value else "with_observed_context"]
    assert empty["sample_count"] == 0
    assert empty["snr_status"] == "no_queries"
    assert empty["rmse"] is None
    assert result["evaluation_target"]["sample_count"] == 6
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure", ["observed", "unknown", "duplicates", "padding", "nan"])
def test_invalid_predictions_rejected_before_target_reads(failure: str) -> None:
    domain, ids, prediction, context, amplitudes = _example()
    if failure == "observed":
        ids[0] = 40
    elif failure == "unknown":
        ids[0] = 100
    elif failure == "duplicates":
        ids[0] = ids[1]
    elif failure == "padding":
        prediction = np.pad(prediction, ((0, 0), (0, 1)))
    else:
        prediction[0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_trace_graph_prediction(
            prediction,
            domain,
            query_trace_ids=ids,
            has_observed_context=context,
            amplitudes=amplitudes,
        )
    assert amplitudes.reads == []


def test_arbitrary_queries_without_target_rows_cannot_be_scored() -> None:
    domain, ids, prediction, context, amplitudes = _example()
    with pytest.raises(ValueError, match="mapped target"):
        evaluate_trace_graph_prediction(
            prediction,
            replace(domain, array_rows=None),
            query_trace_ids=ids,
            has_observed_context=context,
            amplitudes=amplitudes,
        )
    assert amplitudes.reads == []
