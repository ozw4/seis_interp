"""Small train/validation trace pools with an analytic time signal."""

from __future__ import annotations

import numpy as np

from seis_interp.data.trace_graph_domain import TraceGraphDomain, build_trace_graph_domain


def make_trace_graph_training_domains(
    *, pool: str = "all_train_traces", time_count: int = 7
) -> tuple[TraceGraphDomain, TraceGraphDomain, np.ndarray]:
    time_s = np.arange(time_count, dtype=np.float64) * 0.004
    phase = np.linspace(0.0, 2.4, time_count)
    values = np.tile((0.2 + np.sin(phase)).astype(np.float32), (16, 1))
    domains = []
    for partition, rows in [("train", np.arange(8)), ("validation", np.arange(8, 16))]:
        x = np.arange(8, dtype=np.float64) * 0.2
        source = np.column_stack((x, 0.04 * np.sin(x)))
        receiver = source - [1.0, 0.3]
        observed = np.ones(8, dtype=bool) if partition == "train" else rows % 2 == 0
        domains.append(
            build_trace_graph_domain(
                trace_ids=rows + 100,
                source_xy_m=source,
                receiver_xy_m=receiver,
                ffids=rows // 2,
                array_rows=rows,
                observed_mask=observed,
                time_s=time_s,
                pool=pool if partition == "train" else None,
                inputs_lock={"partition": partition},
            )
        )
    return domains[0], domains[1], values
