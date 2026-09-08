"""Query-order native artifacts and exact observed dense reconstruction."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp import run_records
from seis_interp.data.c3_volume_index_store import OUTPUT_FILE_NAMES, write_c3_volume_index
from seis_interp.data.trace_graph_domain import build_trace_graph_domain
from seis_interp.data.trace_graph_prediction_store import (
    save_trace_graph_prediction,
    trace_graph_prediction_to_volume,
    trace_graph_query_index,
)
from seis_interp.processing.c3_volume_index import (
    INDEX_CONTRACT,
    VOLUME_AXIS_ORDER,
    VOLUME_INDEX_COLUMNS,
)
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def _volume_example(tmp_path: Path) -> tuple:
    rows = np.array([5, 1, 4, 0])
    index = pd.DataFrame(
        [
            [int(row), 10, 0, 0, 0, local, 100.0, 20.0, 0.0, float(local)]
            for local, row in enumerate(rows)
        ],
        columns=VOLUME_INDEX_COLUMNS,
    )
    volume_dir = tmp_path / "volume"
    metadata = write_c3_volume_index(
        volume_dir,
        index,
        {
            "volume_id": "shuffled",
            "dataset_id": "synthetic",
            "partition": "test",
            "config_source": "studies/synthetic/config.yaml",
            "axis_order": list(VOLUME_AXIS_ORDER),
            "selection": {
                "time": [1, 4],
                "source_line": [0, 1],
                "shot_in_line": [0, 1],
                "relative_receiver_x": [0, 1],
                "relative_receiver_y": [0, 4],
            },
            "shape": [3, 1, 1, 1, 4],
            "trace_count": 4,
            "role_counts": {"observed": 2, "evaluation_target": 2},
            "index_contract": dict(INDEX_CONTRACT),
            "benchmark_case": {
                "case_id": "example",
                "file": "benchmark_case.json",
                "sha256": "a" * 64,
            },
        },
    )
    # Domain order, storage order and dense index order are all different.
    ordering = np.array([2, 0, 3, 1])
    source = np.tile([100.0, 20.0], (4, 1))
    receiver = source + np.column_stack((np.zeros(4), np.arange(4.0)))
    domain = build_trace_graph_domain(
        trace_ids=np.array([40, 50, 10, 20]),
        source_xy_m=source[ordering],
        receiver_xy_m=receiver[ordering],
        ffids=np.full(4, 10),
        array_rows=rows[ordering],
        observed_mask=np.array([True, True, False, False]),
        time_s=np.array([0.01, 0.02, 0.03]),
        time_samples=(1, 4),
        inputs_lock={
            "partition": "test",
            "benchmark_case": metadata["benchmark_case"],
            "benchmark_volume": {
                "volume_id": metadata["volume_id"],
                "files": run_records.file_hashes(volume_dir, OUTPUT_FILE_NAMES),
                "array_rows": rows.tolist(),
                "selection": metadata["selection"],
            },
        },
    )
    amplitudes = np.full((7, 6), np.nan, dtype=np.float32)
    amplitudes[4, 1:4] = [0.1, -0.0, 1.0e-18]
    amplitudes[5, 1:4] = [-3.14, 1.0e18, 0.7]
    predictions = np.array([[11, 12, 13], [21, 22, 23]], dtype=np.float32)
    return domain, index, metadata, volume_dir, amplitudes, predictions


class _ObservedRows:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape, self.ndim, self.dtype = values.shape, values.ndim, values.dtype
        self.reads: list[np.ndarray] = []

    def __getitem__(self, selection: tuple[np.ndarray, slice]) -> np.ndarray:
        rows, time = selection
        assert set(rows) <= {4, 5}
        assert time == slice(1, 4)
        self.reads.append(rows.copy())
        return self.values[selection]


def test_dense_mapping_uses_existing_index_order_and_exact_observed_values(tmp_path: Path) -> None:
    domain, index, metadata, _, amplitudes, prediction = _volume_example(tmp_path)
    reader = _ObservedRows(amplitudes)
    first = trace_graph_prediction_to_volume(
        prediction,
        domain,
        query_trace_ids=np.array([20, 10]),
        index_table=index,
        volume_metadata=metadata,
        amplitudes=reader,
    )
    second = trace_graph_prediction_to_volume(
        -prediction * 100,
        domain,
        query_trace_ids=np.array([20, 10]),
        index_table=index,
        volume_metadata=metadata,
        amplitudes=reader,
    )
    assert first.shape == (3, 1, 1, 1, 4)
    expected = np.stack([amplitudes[5, 1:4], prediction[0], amplitudes[4, 1:4], prediction[1]])
    np.testing.assert_array_equal(first.reshape(3, 4).T, expected)
    assert first[..., [0, 2]].tobytes() == second[..., [0, 2]].tobytes()
    np.testing.assert_array_equal(np.concatenate(reader.reads), [4, 5, 4, 5])


def test_native_artifacts_preserve_query_order_without_reading_any_waveform(tmp_path: Path) -> None:
    domain, index, metadata, _, amplitudes, prediction = _volume_example(tmp_path)
    reader = _ObservedRows(amplitudes)
    summary = save_trace_graph_prediction(
        tmp_path / "native",
        prediction,
        domain,
        query_trace_ids=np.array([20, 10]),
        has_observed_context=np.array([False, True]),
        amplitudes=reader,
    )
    assert reader.reads == []
    assert summary["axis_order"] == ["query", "time"]
    assert summary["layout"] == "native_trace_list"
    np.testing.assert_array_equal(np.load(tmp_path / "native/prediction.npy"), prediction)
    saved = pd.read_parquet(tmp_path / "native/query_index.parquet")
    assert saved["trace_id"].tolist() == [20, 10]
    assert saved["array_row"].tolist() == [1, 0]
    assert saved["has_observed_context"].tolist() == [False, True]
    np.testing.assert_array_equal(saved["receiver_y_m"], [21, 23])
    json.dumps(summary, allow_nan=False)


def test_dense_artifacts_record_layout_and_query_correspondence(tmp_path: Path) -> None:
    domain, _, _, volume_dir, amplitudes, prediction = _volume_example(tmp_path)
    summary = save_trace_graph_prediction(
        tmp_path / "dense",
        prediction,
        domain,
        query_trace_ids=np.array([20, 10]),
        has_observed_context=np.array([False, True]),
        volume_dir=volume_dir,
        amplitudes=amplitudes,
    )
    assert summary["axis_order"] == list(VOLUME_AXIS_ORDER)
    assert summary["shape"] == [3, 1, 1, 1, 4]
    assert summary["layout"] == "dense_volume"
    assert len(pd.read_parquet(tmp_path / "dense/query_index.parquet")) == 2
    json.dumps(summary, allow_nan=False)


def test_arbitrary_query_mapping_requires_no_storage_row() -> None:
    domain, _, _ = make_relational_trace_domains()
    domain = replace(domain, array_rows=None)
    result = trace_graph_query_index(
        domain,
        query_trace_ids=np.array([30, 20]),
        has_observed_context=np.array([False, True]),
    )
    assert "array_row" not in result
    assert result["trace_id"].tolist() == [30, 20]


@pytest.mark.parametrize("kind", ["outside_crop", "train_partition", "unselected", "time"])
def test_dense_reconstruction_rejects_mismatched_prediction_domain_before_reads(
    tmp_path: Path, kind: str
) -> None:
    domain, index, metadata, _, amplitudes, prediction = _volume_example(tmp_path)
    lock = deepcopy(domain.inputs_lock)
    if kind == "outside_crop":
        rows = domain.array_rows.copy()
        rows[0] = 6
        domain = replace(domain, array_rows=rows)
    elif kind == "train_partition":
        lock["partition"] = "train"
        domain = replace(domain, inputs_lock=lock)
    elif kind == "unselected":
        del lock["benchmark_volume"]
        domain = replace(domain, inputs_lock=lock)
    else:
        domain = replace(domain, time_samples=(0, 3))
    reader = _ObservedRows(amplitudes)
    with pytest.raises(ValueError, match="volume|partition"):
        trace_graph_prediction_to_volume(
            prediction,
            domain,
            query_trace_ids=np.array([20, 10]),
            index_table=index,
            volume_metadata=metadata,
            amplitudes=reader,
        )
    assert reader.reads == []


def test_partial_queries_cannot_leave_dense_target_rows_unfilled(tmp_path: Path) -> None:
    domain, index, metadata, _, amplitudes, prediction = _volume_example(tmp_path)
    with pytest.raises(ValueError, match="every query"):
        trace_graph_prediction_to_volume(
            prediction[:1],
            domain,
            query_trace_ids=np.array([20]),
            index_table=index,
            volume_metadata=metadata,
            amplitudes=_ObservedRows(amplitudes),
        )


def test_prediction_store_does_not_overwrite_existing_artifacts(tmp_path: Path) -> None:
    domain, _, _, _, _, prediction = _volume_example(tmp_path)
    output = tmp_path / "native"
    output.mkdir()
    (output / "query_index.parquet").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        save_trace_graph_prediction(
            output,
            prediction,
            domain,
            query_trace_ids=np.array([20, 10]),
            has_observed_context=np.array([False, True]),
        )
    assert not (output / "prediction.npy").exists()
    assert (output / "query_index.parquet").read_bytes() == b"keep"
