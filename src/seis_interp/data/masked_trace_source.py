"""Read only selected observed support traces and assemble graph inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_interp.data.masked_trace_inputs import (
    MaskedTraceGraphInputs,
    validate_masked_trace_graph_inputs,
)
from seis_interp.processing.trace_graph_geometry import (
    build_trace_graph_edge_features,
    build_trace_graph_node_features,
    compute_trace_graph_geometry,
)
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing

if TYPE_CHECKING:
    from seis_interp.data.trace_graph_domain import TraceGraphDomain
    from seis_interp.processing.trace_graph_subgraphs import TraceGraphPlan


class MaskedTraceSource:
    """Lazy original-row reader bound to one verified domain and fixed scales."""

    def __init__(
        self,
        domain: TraceGraphDomain,
        preprocessing: TraceGraphPreprocessing,
        amplitudes: np.ndarray | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        _require_time_grid(domain.time_s, preprocessing)
        if np.any(domain.observed_mask) and (
            domain.array_rows is None or np.any(domain.array_rows[domain.observed_mask] < 0)
        ):
            raise ValueError("observed domain traces must have amplitude array rows")
        if amplitudes is None and domain.amplitudes_path is not None:
            amplitudes = np.load(domain.amplitudes_path, mmap_mode="r", allow_pickle=False)
        if amplitudes is None and np.any(domain.observed_mask):
            raise ValueError("observed amplitudes or an amplitudes_path are required")
        if amplitudes is not None:
            if amplitudes.ndim != 2 or amplitudes.dtype != np.float32:
                raise ValueError("amplitudes must have float32 shape [rows, time]")
            if domain.time_samples[1] > amplitudes.shape[1]:
                raise ValueError("domain time selection is outside amplitudes")
            if domain.array_rows is not None and np.any(
                domain.array_rows[domain.observed_mask] >= amplitudes.shape[0]
            ):
                raise ValueError("observed array rows are outside amplitudes")
        self.domain = domain
        self.preprocessing = preprocessing
        self._amplitudes = amplitudes
        self._device = torch.device(device)
        self._positions = {int(trace_id): index for index, trace_id in enumerate(domain.trace_ids)}
        self._observed_pairs = {
            (*source, *receiver)
            for source, receiver in zip(
                domain.source_xy_m[domain.observed_mask],
                domain.receiver_xy_m[domain.observed_mask],
                strict=True,
            )
        }

    def read_observed_rows(self, trace_ids: np.ndarray) -> np.ndarray:
        """Read physical samples for these allowed IDs, preserving their order."""
        ids = np.asarray(trace_ids)
        if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
            raise ValueError("trace_ids must be a one-dimensional integer array")
        positions = []
        for trace_id in ids:
            position = self._positions.get(int(trace_id))
            if position is None or not self.domain.observed_mask[position]:
                raise ValueError("requested trace is outside the observed domain")
            positions.append(position)
        if not positions:
            return np.empty((0, len(self.domain.time_s)), dtype=np.float32)
        assert self.domain.array_rows is not None and self._amplitudes is not None
        rows = self.domain.array_rows[np.asarray(positions, dtype=np.int64)]
        start, stop = self.domain.time_samples
        values = np.array(self._amplitudes[rows, start:stop], dtype=np.float32, copy=True)
        if values.shape != (len(rows), len(self.domain.time_s)):
            raise ValueError("observed row reader returned an inconsistent shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("observed amplitudes must be finite")
        return values

    def inputs(self, plan: TraceGraphPlan) -> MaskedTraceGraphInputs:
        """Materialize only the observed support of this dependency plan."""
        for index, trace_id in enumerate(plan.trace_ids):
            position = self._positions.get(int(trace_id))
            if plan.observed_mask[index] and (
                position is None or not self.domain.observed_mask[position]
            ):
                raise ValueError("graph support is outside the observed domain")
            source = plan.geometry.source_xy_m[index]
            receiver = plan.geometry.receiver_xy_m[index]
            if position is not None:
                if not np.array_equal(
                    source, self.domain.source_xy_m[position]
                ) or not np.array_equal(receiver, self.domain.receiver_xy_m[position]):
                    raise ValueError("graph geometry does not match the domain trace ID")
            elif (*source, *receiver) in self._observed_pairs:
                raise ValueError("query duplicates an observed physical source/receiver pair")
        observed_ids = plan.trace_ids[plan.observed_mask]
        values = self.read_observed_rows(observed_ids)
        return assemble_masked_trace_graph_inputs(
            plan,
            observed_trace_ids=observed_ids,
            observed_waveforms=values,
            time_s=self.domain.time_s,
            preprocessing=self.preprocessing,
            device=self._device,
        )


def assemble_masked_trace_graph_inputs(
    plan: TraceGraphPlan,
    *,
    observed_trace_ids: np.ndarray,
    observed_waveforms: np.ndarray,
    time_s: np.ndarray,
    preprocessing: TraceGraphPreprocessing,
    device: torch.device | str = "cpu",
) -> MaskedTraceGraphInputs:
    """Assemble from physical observed traces without requiring file row IDs.

    The waveforms follow ``observed_trace_ids``. Queries are supplied by the
    geometry-only plan and have no waveform input. Only required support is
    selected from these explicitly observed traces.
    """
    _require_time_grid(time_s, preprocessing)
    ids = np.asarray(observed_trace_ids)
    if (
        ids.ndim != 1
        or (ids.size and ids.dtype.kind not in "iu")
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("observed_trace_ids must be a unique integer vector")
    if np.any(np.isin(plan.trace_ids[plan.query_indices], ids)):
        raise ValueError("query IDs must not be supplied as observed traces")
    pairs = np.column_stack((plan.geometry.source_xy_m, plan.geometry.receiver_xy_m))
    observed_pairs = {tuple(pair) for pair in pairs[plan.observed_mask]}
    if any(tuple(pair) in observed_pairs for pair in pairs[plan.query_indices]):
        raise ValueError("query duplicates an observed physical source/receiver pair")
    if observed_waveforms.shape != (len(ids), len(time_s)):
        raise ValueError("observed_waveforms must have shape [observed traces, T]")
    if observed_waveforms.dtype != np.float32:
        raise ValueError("observed_waveforms must have dtype float32")
    positions = {int(trace_id): index for index, trace_id in enumerate(ids)}
    try:
        support_positions = [
            positions[int(trace_id)] for trace_id in plan.trace_ids[plan.observed_mask]
        ]
    except KeyError as error:
        raise ValueError("graph support is missing from observed_trace_ids") from error
    values = observed_waveforms[np.asarray(support_positions, dtype=np.int64)]
    if not np.all(np.isfinite(values)):
        raise ValueError("observed support waveforms must be finite")
    waveforms = np.zeros((len(plan.trace_ids), len(time_s)), dtype=np.float32)
    waveforms[plan.observed_mask] = values.astype(np.float64) / preprocessing.amplitude_scale
    geometry = compute_trace_graph_geometry(
        plan.geometry.source_xy_m,
        plan.geometry.receiver_xy_m,
        azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
    )
    node_features = build_trace_graph_node_features(
        geometry,
        midpoint_origin_m=np.asarray(preprocessing.midpoint_origin_m),
        position_scale_m=preprocessing.position_scale_m,
        offset_scale_m=preprocessing.offset_scale_m,
        observed_mask=plan.observed_mask,
    )
    edge_features = build_trace_graph_edge_features(
        geometry,
        sender_indices=plan.edge_index[0],
        destination_indices=plan.edge_index[1],
        relation_distances=plan.edge_distances,
        position_scale_m=preprocessing.position_scale_m,
        offset_scale_m=preprocessing.offset_scale_m,
    )
    inputs = MaskedTraceGraphInputs(
        waveforms=torch.from_numpy(waveforms),
        node_features=torch.from_numpy(node_features),
        edge_features=torch.from_numpy(edge_features),
        edge_index=torch.tensor(plan.edge_index, dtype=torch.int64),
        edge_type=torch.tensor(plan.edge_type, dtype=torch.int64),
        observed_mask=torch.tensor(plan.observed_mask, dtype=torch.bool),
        query_indices=torch.tensor(plan.query_indices, dtype=torch.int64),
        coverage=torch.tensor(plan.coverage, dtype=torch.float32),
        dependency_rounds=plan.dependency_rounds,
        relation_names=plan.relation_names,
        common_edge_distances=(
            None
            if plan.common_edge_distances is None
            else torch.tensor(plan.common_edge_distances, dtype=torch.float32)
        ),
    )
    return validate_masked_trace_graph_inputs(inputs.to(device))


def _require_time_grid(time_s: np.ndarray, preprocessing: TraceGraphPreprocessing) -> None:
    if not np.array_equal(np.asarray(time_s, dtype=np.float64), preprocessing.time_s):
        raise ValueError("observed time_s must match the fixed preprocessing time grid")
