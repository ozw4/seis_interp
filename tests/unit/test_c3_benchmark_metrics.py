from __future__ import annotations

import json

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.trace_graph_domain import build_trace_graph_domain
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction


@pytest.mark.parametrize("mode", ["half", "zero_fill", "perfect", "zero_reference"])
def test_dense_and_query_metrics_conform_in_physical_units(tmp_path, mode):
    # One observed trace and three targets, stored in a deliberately different row order.
    rows = np.array([4, 1, 6, 2])
    observed = np.array([True, False, False, False])
    references = np.array([[8, 9], [1, 2], [2, 4], [3, 6]], dtype=np.float32)
    if mode == "zero_reference":
        references[1:] = 0
    values = np.full((8, 7), 1e9, dtype=np.float32)
    values[rows, 2:4] = references
    np.save(tmp_path / "amplitudes.npy", values)
    source = np.column_stack((np.arange(4.0), np.zeros(4)))
    times = np.array([0.45, 0.46])
    domain = build_trace_graph_domain(
        trace_ids=rows,
        source_xy_m=source,
        receiver_xy_m=source + [0, -10],
        ffids=np.array([5, 5, 9, 9]),
        observed_mask=observed,
        time_s=times,
        array_rows=rows,
        time_samples=(2, 4),
    )
    shape = (1, 1, 1, 4)
    waveform = np.where(observed[:, None], references, 0).T.reshape((2, *shape))
    volume = ObservedC3Volume(
        values=waveform,
        time_s=times,
        array_rows=rows.reshape(shape),
        observed_trace_mask=observed.reshape(shape),
        evaluation_target_trace_mask=(~observed).reshape(shape),
    )
    normalized = references.astype(np.float64) / 13.0
    prediction = (
        normalized * (1.0 if mode == "perfect" else 0.0 if mode == "zero_fill" else 0.5)
    ) * 13.0
    prediction[0] += 1234  # observed errors are auxiliary only
    dense = prediction.T.reshape((2, *shape))
    dense_metrics = evaluate_c3_volume_prediction(
        dense, volume, interim_dir=tmp_path, volume_metadata={"selection": {"time": [2, 4]}}
    )["evaluation_target"]
    order = np.array([3, 1, 2])
    for chunk in (1, 2, 5):
        query_metrics = evaluate_trace_graph_prediction(
            prediction[order],
            domain,
            query_trace_ids=rows[order],
            has_observed_context=np.array([False, True, False]),
            row_chunk_size=chunk,
            amplitudes=values,
        )["evaluation_target"]
        for key in (
            "reference_energy",
            "error_energy",
            "trace_count",
            "sample_count",
            "rmse",
            "relative_l2",
            "snr_db",
        ):
            if dense_metrics[key] is None:
                assert query_metrics[key] is None
            else:
                assert query_metrics[key] == pytest.approx(dense_metrics[key], rel=1e-12, abs=1e-12)
        assert query_metrics["snr_status"] == dense_metrics["snr_status"]
        json.dumps(query_metrics, allow_nan=False)
    if mode == "half":
        assert dense_metrics["reference_energy"] == 5 * (1 + 4 + 9)
        assert dense_metrics["error_energy"] == pytest.approx(1.25 * (1 + 4 + 9))
        assert dense_metrics["snr_db"] == pytest.approx(10 * np.log10(4))


def test_exact_single_trace_energy_example(tmp_path):
    domain = build_trace_graph_domain(
        trace_ids=np.array([0, 1]),
        source_xy_m=np.array([[0, 0], [1, 0]]),
        receiver_xy_m=np.array([[0, -1], [1, -1]]),
        ffids=np.array([1, 1]),
        observed_mask=np.array([True, False]),
        array_rows=np.array([0, 1]),
        time_s=np.array([0.1, 0.2]),
    )
    result = evaluate_trace_graph_prediction(
        np.array([[0.5, 1.0]]),
        domain,
        query_trace_ids=np.array([1]),
        has_observed_context=np.array([False]),
        amplitudes=np.array([[99, 99], [1, 2]], dtype=np.float32),
    )
    assert result["evaluation_target"]["reference_energy"] == 5
    assert result["evaluation_target"]["error_energy"] == 1.25
