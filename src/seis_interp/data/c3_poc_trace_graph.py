"""Compact O-only graph training data derived from verified PoC volume inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.data.trace_graph_domain import TraceGraphDomain, build_trace_graph_domain
from seis_interp.processing.trace_graph_preprocessing import (
    TraceGraphPreprocessing,
    build_poc_trace_graph_preprocessing,
)


@dataclass(frozen=True)
class C3PocTraceGraphTrainingData:
    """O geometry and physical waveforms, with no disk reader or T amplitudes.

    Trace IDs retain original artifact row IDs; array_rows index the compact
    observed array and time_samples index its already cropped time grid.
    Construct with build_c3_poc_trace_graph_training_data.
    """

    domain: TraceGraphDomain
    observed_amplitudes: np.ndarray
    preprocessing: TraceGraphPreprocessing

    def source(self, visible_mask: np.ndarray, *, device: str = "cpu") -> MaskedTraceSource:
        """Own only visible physical rows; hidden IDs cannot be read as context."""
        visible = np.asarray(visible_mask)
        if visible.dtype != np.bool_ or visible.shape != self.domain.observed_mask.shape:
            raise ValueError("visible_mask must be a boolean vector over O")
        rows = np.full(len(visible), -1, dtype=np.int64)
        rows[visible] = np.arange(np.count_nonzero(visible))
        domain = replace(
            self.domain,
            array_rows=rows,
            observed_mask=visible.copy(),
            query_mask=~visible,
        )
        return MaskedTraceSource(
            domain, self.preprocessing, self.observed_amplitudes[visible], device=device
        )


def build_c3_poc_trace_graph_training_data(
    inputs: C3VolumeRunInputs,
    *,
    amplitude_scale: float,
    position_scale_m: float,
    offset_scale_m: float,
    azimuth_min_offset_m: float,
) -> C3PocTraceGraphTrainingData:
    """Select exactly O without opening files, fitting RMS, or requiring a partition."""
    volume = inputs.observed_volume
    analysis = build_c3_poc_trace_graph_domain(inputs)
    ids, observed = analysis.trace_ids, analysis.observed_mask
    if np.count_nonzero(observed) < 2:
        raise ValueError("O must contain at least two traces for query and context")
    preprocessing = build_poc_trace_graph_preprocessing(
        analysis,
        amplitude_scale=amplitude_scale,
        position_scale_m=position_scale_m,
        offset_scale_m=offset_scale_m,
        azimuth_min_offset_m=azimuth_min_offset_m,
    )
    domain = build_trace_graph_domain(
        trace_ids=ids[observed],
        source_xy_m=analysis.source_xy_m[observed],
        receiver_xy_m=analysis.receiver_xy_m[observed],
        ffids=analysis.ffids[observed],
        observed_mask=np.ones(np.count_nonzero(observed), dtype=bool),
        array_rows=np.arange(np.count_nonzero(observed), dtype=np.int64),
        time_s=volume.time_s,
        inputs_lock=inputs.inputs_lock,
    )
    amplitudes = np.array(
        volume.values.reshape(len(volume.time_s), -1)[:, observed].T,
        dtype=np.float32,
        copy=True,
    )
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("observed amplitudes must be finite float32 values")
    return C3PocTraceGraphTrainingData(domain, amplitudes, preprocessing)


def build_c3_poc_trace_graph_domain(inputs: C3VolumeRunInputs) -> TraceGraphDomain:
    """Build D geometry and outer visibility without any amplitude path or reader."""
    volume = inputs.observed_volume
    table = inputs.index_table
    ids = table["array_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, volume.array_rows.reshape(-1)):
        raise ValueError("volume index row order must match observed volume")
    observed = volume.observed_trace_mask.reshape(-1)
    source = table[["source_x_m", "source_y_m"]].to_numpy(dtype=np.float64)
    relative = table[["relative_receiver_x_m", "relative_receiver_y_m"]].to_numpy(dtype=np.float64)
    return build_trace_graph_domain(
        trace_ids=ids,
        source_xy_m=source,
        receiver_xy_m=source + relative,
        ffids=table["ffid"].to_numpy(),
        observed_mask=observed,
        time_s=volume.time_s,
        inputs_lock=inputs.inputs_lock,
    )
