"""Map physical trace predictions to query rows or an existing dense index."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp import run_records
from seis_interp.data.c3_volume_index_store import OUTPUT_FILE_NAMES as VOLUME_FILE_NAMES
from seis_interp.data.c3_volume_index_store import load_c3_volume_index, validate_c3_volume_index
from seis_interp.data.trace_graph_domain import TraceGraphDomain
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER


def trace_graph_query_positions(
    domain: TraceGraphDomain,
    query_trace_ids: np.ndarray,
    *,
    require_complete: bool = False,
) -> np.ndarray:
    """Resolve unique query IDs without treating storage order as prediction order."""
    ids = np.asarray(query_trace_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("query_trace_ids must be a unique integer vector")
    positions = {int(trace_id): index for index, trace_id in enumerate(domain.trace_ids)}
    try:
        selected = np.asarray([positions[int(trace_id)] for trace_id in ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("prediction queries must belong to the selected domain") from error
    if np.any(~domain.query_mask[selected]) or np.any(domain.observed_mask[selected]):
        raise ValueError("prediction queries must select only unobserved query rows")
    if require_complete and len(selected) != int(domain.query_mask.sum()):
        raise ValueError("prediction must cover every query in the selected domain")
    return selected


def trace_graph_query_index(
    domain: TraceGraphDomain,
    *,
    query_trace_ids: np.ndarray,
    has_observed_context: np.ndarray,
) -> pd.DataFrame:
    """Build runtime query correspondence, preserving the supplied output order."""
    positions = trace_graph_query_positions(domain, query_trace_ids)
    context = np.asarray(has_observed_context)
    if context.dtype != np.bool_ or context.shape != (len(positions),):
        raise ValueError("has_observed_context must be a boolean vector with one flag per query")
    records = {
        "prediction_row": np.arange(len(positions), dtype=np.int64),
        "trace_id": domain.trace_ids[positions],
        "source_x_m": domain.source_xy_m[positions, 0],
        "source_y_m": domain.source_xy_m[positions, 1],
        "receiver_x_m": domain.receiver_xy_m[positions, 0],
        "receiver_y_m": domain.receiver_xy_m[positions, 1],
        "has_observed_context": context,
    }
    if domain.array_rows is not None:
        rows = domain.array_rows[positions]
        records["array_row"] = pd.array(
            [int(row) if row >= 0 else None for row in rows], dtype="Int64"
        )
    return pd.DataFrame(records)


def trace_graph_prediction_to_volume(
    prediction: np.ndarray,
    domain: TraceGraphDomain,
    *,
    query_trace_ids: np.ndarray,
    index_table: pd.DataFrame,
    volume_metadata: Mapping[str, object],
    amplitudes: np.ndarray | None = None,
) -> np.ndarray:
    """Restore index order and copy original observed float32 amplitudes exactly.

    The domain must already have been limited to this volume during prediction.
    Dense shape is used only here, after the model has returned query traces.
    """
    validate_c3_volume_index(index_table, volume_metadata)
    positions = trace_graph_query_positions(domain, query_trace_ids, require_complete=True)
    prediction = _validated_prediction(prediction, len(positions), len(domain.time_s))
    _require_selected_volume(domain, index_table, volume_metadata)
    assert domain.array_rows is not None
    row_positions = {int(row): position for position, row in enumerate(domain.array_rows)}
    ordered = np.asarray(
        [row_positions[int(row)] for row in index_table["array_row"]], dtype=np.int64
    )
    traces = np.empty((len(domain.trace_ids), len(domain.time_s)), dtype=np.float32)
    observed_positions = np.flatnonzero(domain.observed_mask)
    if len(observed_positions):
        source = _observed_amplitudes(domain, amplitudes)
        start, stop = domain.time_samples
        observed = np.asarray(source[domain.array_rows[observed_positions], start:stop])
        if not np.all(np.isfinite(observed)):
            raise ValueError("original observed amplitudes must be finite")
        traces[observed_positions] = observed
    traces[positions] = prediction
    shape = tuple(int(value) for value in volume_metadata["shape"])
    return np.ascontiguousarray(traces[ordered].T.reshape(shape))


def save_trace_graph_prediction(
    output_dir: Path,
    prediction: np.ndarray,
    domain: TraceGraphDomain,
    *,
    query_trace_ids: np.ndarray,
    has_observed_context: np.ndarray,
    volume_dir: Path | None = None,
    amplitudes: np.ndarray | None = None,
) -> dict[str, object]:
    """Save run artifacts and return their layout for the caller's run metadata."""
    query_index = trace_graph_query_index(
        domain,
        query_trace_ids=query_trace_ids,
        has_observed_context=has_observed_context,
    )
    values = _validated_prediction(prediction, len(query_index), len(domain.time_s))
    axis_order = ["query", "time"]
    layout = "native_trace_list"
    if volume_dir is not None:
        index, metadata = load_c3_volume_index(volume_dir)
        volume_lock = domain.inputs_lock.get("benchmark_volume", {})
        if volume_lock.get("files") != run_records.file_hashes(volume_dir, VOLUME_FILE_NAMES):
            raise ValueError("output volume files must match the prediction domain binding")
        values = trace_graph_prediction_to_volume(
            values,
            domain,
            query_trace_ids=query_trace_ids,
            index_table=index,
            volume_metadata=metadata,
            amplitudes=amplitudes,
        )
        axis_order = list(VOLUME_AXIS_ORDER)
        layout = "dense_volume"
    directory = Path(output_dir)
    for name in ("prediction.npy", "query_index.parquet"):
        if (directory / name).exists():
            raise FileExistsError(f"prediction artifact already exists: {directory / name}")
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "prediction.npy", values, allow_pickle=False)
    query_index.to_parquet(directory / "query_index.parquet", index=False)
    return {
        "layout": layout,
        "axis_order": axis_order,
        "shape": list(values.shape),
        "query_count": len(query_index),
        "prediction_file": "prediction.npy",
        "query_index_file": "query_index.parquet",
    }


def _validated_prediction(values: np.ndarray, query_count: int, time_count: int) -> np.ndarray:
    prediction = np.asarray(values)
    if prediction.shape != (query_count, time_count) or prediction.dtype != np.float32:
        raise ValueError("prediction must have physical float32 shape [Q, T]")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction must contain finite physical amplitudes")
    return prediction


def _require_selected_volume(
    domain: TraceGraphDomain,
    index_table: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    lock = domain.inputs_lock.get("benchmark_volume")
    if not isinstance(lock, Mapping):
        raise ValueError("prediction domain must have the selected volume binding")
    rows = index_table["array_row"].to_numpy(dtype=np.int64)
    if (
        domain.array_rows is None
        or not np.array_equal(np.sort(domain.array_rows), np.sort(rows))
        or lock["array_rows"] != rows.tolist()
    ):
        raise ValueError("prediction context and queries must match the selected volume rows")
    if not np.all(domain.observed_mask | domain.query_mask):
        raise ValueError("observed and query roles must cover the selected volume")
    if (
        domain.inputs_lock.get("partition") != metadata["partition"]
        or domain.inputs_lock.get("benchmark_case", {}).get("sha256")
        != metadata["benchmark_case"]["sha256"]
        or lock["volume_id"] != metadata["volume_id"]
        or lock["selection"] != metadata["selection"]
    ):
        raise ValueError("prediction domain must match the volume case, partition and selection")
    if tuple(metadata["selection"]["time"]) != domain.time_samples:
        raise ValueError("prediction time selection must match the selected volume")
    expected_counts = {
        "observed": int(domain.observed_mask.sum()),
        "evaluation_target": int(domain.query_mask.sum()),
    }
    if expected_counts != metadata["role_counts"]:
        raise ValueError("prediction roles must match the selected volume role counts")


def _observed_amplitudes(domain: TraceGraphDomain, amplitudes: np.ndarray | None) -> np.ndarray:
    if amplitudes is None:
        if domain.amplitudes_path is None:
            raise ValueError("original observed amplitudes are required for dense reconstruction")
        amplitudes = np.load(domain.amplitudes_path, mmap_mode="r", allow_pickle=False)
    if amplitudes.ndim != 2 or amplitudes.dtype != np.float32:
        raise ValueError("original observed amplitudes must have float32 shape [rows, time]")
    assert domain.array_rows is not None
    if domain.time_samples[1] > amplitudes.shape[1] or np.any(
        domain.array_rows[domain.observed_mask] >= amplitudes.shape[0]
    ):
        raise ValueError("selected observed rows and time must be inside original amplitudes")
    return amplitudes
