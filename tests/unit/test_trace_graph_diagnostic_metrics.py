"""Fixed bands conserve energy/counts and deterministic baselines share targets."""

from __future__ import annotations

import json

import numpy as np
import pytest

from seis_interp.evaluation import trace_graph_diagnostic_metrics as diagnostic_metrics
from seis_interp.evaluation.trace_graph_diagnostic_metrics import (
    TraceGraphDiagnosticBands,
    evaluate_trace_graph_baselines,
    evaluate_trace_graph_diagnostic_bands,
    merge_trace_graph_prediction_band_summaries,
    summarize_trace_graph_prediction_bands,
)
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def _band_example():
    reference = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    source = np.array([[0, 0], [0, 2], [2, 0], [-2, 0]], dtype=np.float64)
    return reference, {
        "time_s": np.array([0, 0.1, 0.2, 0.3]),
        "source_xy_m": source,
        "receiver_xy_m": np.zeros_like(source),
        "bands": TraceGraphDiagnosticBands((0.1, 0.25), (1, 2, 3), (90, 180, 270)),
        "azimuth_min_offset_m": 0.01,
    }


def test_fixed_bands_cover_all_samples_and_undefined_azimuth():
    reference, arguments = _band_example()
    summary = summarize_trace_graph_prediction_bands(reference / 2, reference, **arguments)
    assert summary["query_count"] == 4
    assert summary["sample_count"] == 16
    assert summary["reference_energy"] == 1496
    assert summary["error_energy"] == 374
    for axis in ("time_s", "offset_m", "azimuth_deg"):
        rows = summary[axis]
        for name in ("sample_count", "reference_energy", "error_energy"):
            assert sum(row[name] for row in rows) == summary[name]
        if axis != "time_s":
            assert sum(row["query_count"] for row in rows) == 4
    assert [row["sample_count"] for row in summary["time_s"]] == [4, 8, 4]
    assert [row["query_count"] for row in summary["offset_m"]] == [1, 0, 3, 0]
    azimuth = summary["azimuth_deg"]
    assert [row["query_count"] for row in azimuth] == [1, 1, 0, 1, 1]
    assert azimuth[-1]["undefined_azimuth"]
    assert azimuth[-1]["reference_energy"] == 30
    json.dumps(summary, allow_nan=False)


def test_split_band_summaries_are_additive_and_never_refit_cuts():
    reference, arguments = _band_example()
    expected = summarize_trace_graph_prediction_bands(reference / 2, reference, **arguments)
    summaries = [
        summarize_trace_graph_prediction_bands(
            reference[rows] / 2,
            reference[rows],
            **{
                **arguments,
                "source_xy_m": arguments["source_xy_m"][rows],
                "receiver_xy_m": arguments["receiver_xy_m"][rows],
            },
        )
        for rows in (slice(0, 1), slice(1, 4))
    ]
    assert merge_trace_graph_prediction_band_summaries(summaries) == expected
    changed = {**summaries[0], "bands": TraceGraphDiagnosticBands().constructor_config()}
    with pytest.raises(ValueError, match="identical fixed cuts"):
        merge_trace_graph_prediction_band_summaries([summaries[0], changed])


def test_domain_band_reader_only_reads_selected_targets_in_prediction_order():
    domain, _, amplitudes = make_relational_trace_domains()
    ids = np.array([30, 20])
    prediction = np.zeros((2, 5), dtype=np.float32)
    expected_rows = [2, 1]

    class TargetReader:
        shape, ndim = amplitudes.shape, 2
        reads = []

        def __getitem__(self, selection):
            rows, samples = selection
            assert len(rows) == 1
            assert rows[0] == expected_rows[len(self.reads)]
            assert samples == slice(1, 6)
            self.reads.extend(rows.tolist())
            return amplitudes[selection]

    reader = TargetReader()
    summary = evaluate_trace_graph_diagnostic_bands(
        prediction,
        domain,
        query_trace_ids=ids,
        bands=TraceGraphDiagnosticBands(time_s=(0.025,), offset_m=(2,)),
        azimuth_min_offset_m=0.1,
        amplitudes=reader,
        row_chunk_size=1,
    )
    assert reader.reads == expected_rows
    assert summary["query_count"] == 2
    assert summary["sample_count"] == 10
    assert (
        summary["reference_energy"]
        == np.square(amplitudes[expected_rows, 1:6].astype(np.float64)).sum()
    )


def test_zero_and_idw_share_physical_target_domain_without_label_input(monkeypatch):
    domain, training, amplitudes = make_relational_trace_domains()
    fixed = fit_trace_graph_preprocessing(
        training,
        amplitudes,
        position_scale_m=10,
        offset_scale_m=10,
        azimuth_min_offset_m=0.1,
    )
    settings = TraceGraphSettings(((1.0, 1.0),) * 4, neighbors_per_relation=2)
    outputs = []
    evaluator = diagnostic_metrics.evaluate_trace_graph_prediction

    def capture(values, target_domain, **kwargs):
        assert target_domain is domain
        outputs.append(values.copy())
        return evaluator(values, target_domain, **kwargs)

    monkeypatch.setattr(diagnostic_metrics, "evaluate_trace_graph_prediction", capture)
    first = evaluate_trace_graph_baselines(
        domain,
        fixed,
        graph_settings=settings,
        query_batch_size=1,
        amplitudes=amplitudes,
    )
    altered = amplitudes.copy()
    altered[domain.array_rows[domain.query_mask]] *= 5
    second = evaluate_trace_graph_baselines(
        domain,
        fixed,
        graph_settings=settings,
        query_batch_size=2,
        amplitudes=altered,
    )
    np.testing.assert_array_equal(outputs[0], np.zeros((2, 5), dtype=np.float32))
    np.testing.assert_array_equal(outputs[0], outputs[2])
    np.testing.assert_array_equal(outputs[1], outputs[3])
    assert np.abs(outputs[1]).max() > 1
    for name in ("zero", "idw"):
        scores = first[name]["evaluation_target"]
        assert scores["trace_count"] == 2 and scores["sample_count"] == 10
        assert (
            second[name]["evaluation_target"]["reference_energy"] == 25 * scores["reference_energy"]
        )
    assert (
        first["zero"]["evaluation_target"]["reference_energy"]
        == first["idw"]["evaluation_target"]["reference_energy"]
    )
    assert first["distance"]["midpoint_scale_m"] == fixed.position_scale_m
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_s": (0.2, 0.1)},
        {"offset_m": (-1,)},
        {"azimuth_deg": (361,)},
        {"time_s": (float("nan"),)},
        {"offset_m": (True,)},
    ],
)
def test_invalid_band_boundaries_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TraceGraphDiagnosticBands(**kwargs)
