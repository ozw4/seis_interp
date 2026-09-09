"""Fit fixed trace-graph scales using only the authorized training pool."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry

if TYPE_CHECKING:
    from seis_interp.data.trace_graph_domain import TraceGraphDomain


@dataclass(frozen=True)
class TraceGraphPreprocessing:
    """Pure fixed values; fitting and applying these values are separate steps."""

    amplitude_scale: float
    midpoint_origin_m: tuple[float, float]
    position_scale_m: float
    offset_scale_m: float
    azimuth_min_offset_m: float
    time_s: tuple[float, ...]
    fit_domain: dict[str, object]

    def __post_init__(self) -> None:
        for name in ("amplitude_scale", "position_scale_m", "offset_scale_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be positive and finite")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        threshold = self.azimuth_min_offset_m
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not np.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError("azimuth_min_offset_m must be nonnegative and finite")
        origin = np.asarray(self.midpoint_origin_m, dtype=np.float64)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("midpoint_origin_m must have two finite coordinates")
        time = np.asarray(self.time_s, dtype=np.float64)
        if time.ndim != 1 or not len(time) or not np.all(np.isfinite(time)):
            raise ValueError("time_s must be a nonempty finite vector")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time_s must be strictly increasing")


def fit_trace_graph_preprocessing(
    domain: TraceGraphDomain,
    amplitudes: np.ndarray | None = None,
    *,
    position_scale_m: float,
    offset_scale_m: float,
    azimuth_min_offset_m: float,
    row_chunk_size: int = 4096,
    max_abs_amplitude: float | None = None,
) -> TraceGraphPreprocessing:
    """Stream float64 global RMS over O0 and fix its arithmetic mean CMP.

    ``domain`` must be an unmasked training pool produced by the training
    domain loader. The supplied amplitude array, if any, uses original file
    rows and time samples. Only those rows and the selected time slice are
    read. Artificial episode masks must be applied after this fit. An explicit
    amplitude bound rejects excessive physical samples before their energy is
    accumulated; it never clips samples or removes traces.
    """
    if domain.pool not in ("all_train_traces", "mask_observed"):
        raise ValueError("preprocessing requires an explicit authorized training pool")
    if not len(domain.trace_ids) or not np.all(domain.observed_mask) or np.any(domain.query_mask):
        raise ValueError("preprocessing must be fitted to the complete unmasked training pool")
    if domain.array_rows is None or np.any(domain.array_rows < 0):
        raise ValueError("training pool must have amplitude array rows")
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, Integral)
        or row_chunk_size <= 0
    ):
        raise ValueError("row_chunk_size must be a positive integer")
    if max_abs_amplitude is not None and (
        isinstance(max_abs_amplitude, bool)
        or not isinstance(max_abs_amplitude, Real)
        or not np.isfinite(max_abs_amplitude)
        or max_abs_amplitude <= 0
    ):
        raise ValueError("max_abs_amplitude must be positive and finite")
    if amplitudes is None:
        if domain.amplitudes_path is None:
            raise ValueError("training amplitudes or an amplitudes_path are required")
        amplitudes = np.load(domain.amplitudes_path, mmap_mode="r", allow_pickle=False)
    start, stop = domain.time_samples
    if amplitudes.ndim != 2 or amplitudes.dtype != np.float32:
        raise ValueError("amplitudes must have float32 shape [rows, time]")
    if np.any(domain.array_rows >= amplitudes.shape[0]) or stop > amplitudes.shape[1]:
        raise ValueError("training rows or time selection are outside amplitudes")

    order = np.argsort(domain.trace_ids, kind="stable")
    rows = domain.array_rows[order]
    sum_squares = 0.0
    sample_count = 0
    for offset in range(0, len(rows), row_chunk_size):
        chunk = np.asarray(amplitudes[rows[offset : offset + row_chunk_size], start:stop])
        if chunk.shape != (len(rows[offset : offset + row_chunk_size]), stop - start):
            raise ValueError("training row reader returned an inconsistent shape")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("training amplitudes must be finite")
        if max_abs_amplitude is not None:
            excessive = (chunk > max_abs_amplitude) | (chunk < -max_abs_amplitude)
            if np.any(excessive):
                row_index, sample_index = np.unravel_index(np.argmax(excessive), chunk.shape)
                position = offset + row_index
                raise ValueError(
                    f"training amplitude exceeds max_abs_amplitude={max_abs_amplitude} "
                    f"at trace_id {domain.trace_ids[order[position]]}, "
                    f"array_row {rows[position]}, sample {start + sample_index}: "
                    f"{chunk[row_index, sample_index]}"
                )
        values = chunk.astype(np.float64)
        sum_squares += float(np.sum(values * values, dtype=np.float64))
        sample_count += values.size
    amplitude_scale = float(np.sqrt(sum_squares / sample_count))
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m,
        domain.receiver_xy_m,
        azimuth_min_offset_m=azimuth_min_offset_m,
    )
    origin = np.mean(geometry.midpoint_xy_m[order], axis=0, dtype=np.float64)
    return TraceGraphPreprocessing(
        amplitude_scale=amplitude_scale,
        midpoint_origin_m=(float(origin[0]), float(origin[1])),
        position_scale_m=position_scale_m,
        offset_scale_m=offset_scale_m,
        azimuth_min_offset_m=float(azimuth_min_offset_m),
        time_s=tuple(float(value) for value in domain.time_s),
        fit_domain={
            "pool": domain.pool,
            "trace_count": len(rows),
            "trace_ids_sha256": sha256(domain.trace_ids[order].astype("<i8").tobytes()).hexdigest(),
            "time_samples": [start, stop],
            "inputs_lock": deepcopy(domain.inputs_lock),
        },
    )
