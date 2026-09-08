"""Continuous analytic wavelets at physical coordinates, not propagation or field data."""

from __future__ import annotations

import numpy as np

from seis_interp.data.trace_graph_domain import TraceGraphDomain, build_trace_graph_domain


def analytic_trace_waveforms(
    source_xy_m: np.ndarray, receiver_xy_m: np.ndarray, time_s: np.ndarray
) -> np.ndarray:
    """Evaluate two Ricker arrivals and amplitudes at the supplied coordinates."""
    source = np.asarray(source_xy_m, dtype=np.float64)
    receiver = np.asarray(receiver_xy_m, dtype=np.float64)
    midpoint = (source + receiver) / 2.0
    offset = source - receiver
    length = np.linalg.norm(offset, axis=1)
    arrival_one = 0.038 + midpoint[:, 0] / 6000 + midpoint[:, 1] / 8000 + length / 12000
    arrival_two = 0.094 - midpoint[:, 0] / 10000 + offset[:, 1] / 20000
    amplitude_one = 1.0 + 0.2 * np.sin(midpoint[:, 0] / 30) + 0.1 * np.cos(offset[:, 1] / 50)
    amplitude_two = -0.6 + 0.1 * np.sin(midpoint[:, 1] / 20)
    traces = np.zeros((len(source), len(time_s)), dtype=np.float64)
    for arrival, amplitude, frequency in (
        (arrival_one, amplitude_one, 22.0),
        (arrival_two, amplitude_two, 28.0),
    ):
        phase_squared = (np.pi * frequency * (time_s[None, :] - arrival[:, None])) ** 2
        traces += amplitude[:, None] * (1 - 2 * phase_squared) * np.exp(-phase_squared)
    return traces.astype(np.float32)


def make_analytic_trace_domains(
    mask_kind: str,
) -> tuple[TraceGraphDomain, TraceGraphDomain, np.ndarray]:
    """Return irregular complete training and fixed masked validation trace lists."""
    time_s = np.arange(33, dtype=np.float64) * 0.004
    sources, receivers, ffids = [], [], []
    for shot, count in enumerate((3, 2, 4, 2, 3, 3, 2, 3)):
        source = np.array([7.1 * shot, 3.4 * np.sin(shot)])
        for receiver in range(count):
            offset = np.array([11.0 + 4.3 * receiver + 0.2 * shot, 18.0 - 2.1 * receiver])
            sources.append(source)
            receivers.append(source - offset)
            ffids.append(1000 + shot)
    source_xy = np.asarray(sources)
    receiver_xy = np.asarray(receivers)
    ffids = np.asarray(ffids, dtype=np.int64)
    values = analytic_trace_waveforms(source_xy, receiver_xy, time_s)
    domains = []
    for partition, rows in (("train", np.arange(14)), ("validation", np.arange(14, 22))):
        observed = np.ones(len(rows), dtype=bool)
        if partition == "validation":
            if mask_kind == "random_trace":
                observed[[0, 3, 5, 7]] = False
            elif mask_kind == "random_whole_ffid":
                observed[ffids[rows] == 1006] = False
            else:
                raise ValueError("mask_kind must be random_trace or random_whole_ffid")
        domains.append(
            build_trace_graph_domain(
                trace_ids=rows + 100,
                source_xy_m=source_xy[rows],
                receiver_xy_m=receiver_xy[rows],
                ffids=ffids[rows],
                array_rows=rows,
                observed_mask=observed,
                time_s=time_s,
                pool="all_train_traces" if partition == "train" else None,
                inputs_lock={"partition": partition, "fixture": "analytic_ricker_arrivals"},
            )
        )
    return domains[0], domains[1], values


def analytic_offgrid_query_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Give independent arbitrary coordinates, including one query without context."""
    source = np.array([[3.17, 1.31], [17.63, -0.41], [29.28, 4.13], [1000.0, 7.4]])
    receiver = source - [[12.7, 17.2], [19.3, 13.6], [16.1, 14.9], [13.8, 15.1]]
    return source, receiver
