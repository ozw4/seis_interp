"""Independent physical IDW over unique direct observed query neighbors."""

from __future__ import annotations

from numbers import Real

import numpy as np

from seis_interp.processing.trace_graph_subgraphs import TraceGraphPlan


def predict_trace_graph_idw(
    plan: TraceGraphPlan,
    *,
    observed_trace_ids: np.ndarray,
    observed_waveforms: np.ndarray,
    midpoint_scale_m: float,
    offset_scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical ``[Q,T]`` and context flags, without target amplitudes.

    Union the direct incoming senders across relations before weighting by
    ``1 / max(D0, 1e-6)**2``, where D0 uses midpoint and full-offset distances.
    No-context outputs are exact zero. The model's fixed RMS is not applied.
    """
    for name, value in (("midpoint_scale_m", midpoint_scale_m), ("offset_scale_m", offset_scale_m)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    ids = np.asarray(observed_trace_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("observed_trace_ids must be a unique integer vector")
    waveforms = np.asarray(observed_waveforms)
    if waveforms.ndim != 2 or waveforms.shape[0] != len(ids) or waveforms.dtype != np.float32:
        raise ValueError("observed_waveforms must have physical float32 shape [observed traces, T]")
    if np.any(np.isin(ids, plan.trace_ids[plan.query_indices])):
        raise ValueError("query IDs must not be supplied as observed traces")
    if np.any(~plan.observed_mask[plan.edge_index[0]]):
        raise ValueError("IDW senders must be observed")
    positions = {int(trace_id): index for index, trace_id in enumerate(ids)}
    output = np.zeros((len(plan.query_indices), waveforms.shape[1]), dtype=np.float32)
    context = np.zeros(len(plan.query_indices), dtype=bool)
    for row, query in enumerate(plan.query_indices):
        senders = np.unique(plan.edge_index[0, plan.edge_index[1] == query])
        if not len(senders):
            continue
        try:
            values = waveforms[[positions[int(trace_id)] for trace_id in plan.trace_ids[senders]]]
        except KeyError as error:
            raise ValueError("direct IDW support is missing from observed_trace_ids") from error
        if not np.all(np.isfinite(values)):
            raise ValueError("observed IDW support waveforms must be finite")
        midpoint = plan.geometry.midpoint_xy_m[senders] - plan.geometry.midpoint_xy_m[query]
        offset = plan.geometry.offset_xy_m[senders] - plan.geometry.offset_xy_m[query]
        distance = np.sqrt(
            np.sum((midpoint / midpoint_scale_m) ** 2, axis=1)
            + np.sum((offset / offset_scale_m) ** 2, axis=1)
        )
        weights = 1.0 / np.maximum(distance, 1e-6) ** 2
        weights /= weights.sum()
        output[row] = np.sum(weights[:, None] * values.astype(np.float64), axis=0)
        context[row] = True
    return output, context
