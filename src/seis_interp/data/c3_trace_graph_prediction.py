"""Complete physical graph predictions scattered into the common C3 volume."""

import numpy as np

from seis_interp.data.c3_poc_trace_graph import build_c3_poc_trace_graph_domain
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.data.trace_graph_prediction_store import trace_graph_query_positions


def scatter_c3_trace_graph_prediction(
    inputs: C3VolumeRunInputs, query_trace_ids: np.ndarray, physical_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Require exactly T once; reinsert O without any additional amplitude scaling."""
    domain = build_c3_poc_trace_graph_domain(inputs)
    positions = trace_graph_query_positions(domain, query_trace_ids, require_complete=True)
    values = np.asarray(physical_prediction)
    if values.shape != (len(positions), len(domain.time_s)) or not np.all(np.isfinite(values)):
        raise ValueError("physical prediction must have finite shape [all target traces, time]")
    volume = inputs.observed_volume
    dense = np.array(volume.values, dtype=np.float32, copy=True)
    dense.reshape(len(domain.time_s), -1)[:, positions] = values.T
    if not np.all(np.isfinite(dense)):
        raise ValueError("dense physical predictions must be finite float32 amplitudes")
    coverage = np.zeros(volume.array_rows.size, dtype=bool)
    coverage[positions] = True
    return dense, coverage.reshape(volume.array_rows.shape)
