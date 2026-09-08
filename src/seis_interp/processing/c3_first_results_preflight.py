"""Measure one predetermined classical-method window without reading target truth."""

from __future__ import annotations

import math
import os
import resource
import time

import numpy as np

from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.processing.drr import (
    interpolate_drr_block,
    select_drr_frequencies,
    validate_drr_parameters,
)
from seis_interp.processing.level_four_hankel import (
    hankelize_level_four,
    level_four_hankel_matrix_shape,
)
from seis_interp.processing.pocs import interpolate_pocs_block


def _window_counts(shape: tuple, window: list, overlap: list) -> list[int]:
    # Native windows anchor their final start at length-window, so edge windows
    # have the same extent as the first and middle windows.
    return [
        1 + math.ceil(max(length - width, 0) / (width - shared))
        for length, width, shared in zip(shape, window, overlap, strict=True)
    ]


def preflight_classical_c3(
    *,
    method: str,
    inputs: C3VolumeRunInputs,
    config: dict,
    action_timeout_seconds: float,
) -> dict:
    """Run the geometry-first window and report a linear reconstruction estimate."""
    if method not in {"pocs", "drr"}:
        raise ValueError("classical preflight requires pocs or drr")
    observed = inputs.observed_volume
    settings = config[method]
    if method == "pocs":
        window, overlap = settings["window_shape"], settings["overlap"]
        counts = _window_counts(observed.values.shape, window, overlap)
        shape = tuple(min(n, w) for n, w in zip(observed.values.shape, window, strict=True))
    else:
        window, overlap = settings["spatial_window_shape"], settings["spatial_overlap"]
        counts = _window_counts(observed.values.shape[1:], window, overlap)
        spatial = tuple(min(n, w) for n, w in zip(observed.values.shape[1:], window, strict=True))
        shape = (observed.values.shape[0], *spatial)
        validate_drr_parameters(settings["rank"], settings["damping_power"], 1, spatial)
    slices = tuple(slice(0, n) for n in shape)
    values, mask = observed.values[slices], observed.observed_trace_mask[slices[1:]]
    extra = {}
    if method == "drr":
        selection = select_drr_frequencies(
            observed.time_s,
            frequency_min_hz=settings["frequency_min_hz"],
            frequency_max_hz=settings["frequency_max_hz"],
        )
        transformed = np.fft.rfft(
            values.astype(np.float64), n=selection.fft_length, axis=0, norm="ortho"
        )
        matrix = hankelize_level_four(transformed[int(selection.bin_indices[0])])
        extra = {
            "hankel_matrix_shape": list(level_four_hankel_matrix_shape(spatial)),
            "svd_input_dtype": str(matrix.dtype),
            "hankel_matrix_bytes": matrix.nbytes,
            "svd_workspace_bytes": None,
            "svd_workspace_reason": "NumPy LAPACK workspace not measured separately",
            "fft_length": selection.fft_length,
            "selected_bin_count": len(selection.bin_indices),
            "frequency_endpoints_hz": [
                float(selection.frequencies_hz[selection.bin_indices[0]]),
                float(selection.frequencies_hz[selection.bin_indices[-1]]),
            ],
        }
    start = time.perf_counter()
    if method == "pocs":
        predicted = interpolate_pocs_block(
            values,
            mask,
            n_iterations=2,
            threshold_start=settings["threshold_start"],
            threshold_end=settings["threshold_end"],
        )
        iterations = 2
    else:
        predicted = interpolate_drr_block(
            values,
            mask,
            observed.time_s,
            rank=settings["rank"],
            damping_power=settings["damping_power"],
            n_iterations=1,
            frequency_min_hz=settings["frequency_min_hz"],
            frequency_max_hz=settings["frequency_max_hz"],
        )
        iterations = 1
    seconds = time.perf_counter() - start
    informative = bool(np.any(mask) and np.any(values[:, mask]))
    estimate = (
        seconds / iterations * settings["n_iterations"] * math.prod(counts) if informative else None
    )
    return {
        "status": "measured",
        "scope": "single_window_resource_preflight",
        "method": method,
        "case_id": inputs.case["case_id"],
        "volume_shape": list(observed.values.shape),
        "window_selection_rule": "first_native_window_by_geometry_no_target_scoring",
        "window_start": [0] * 5,
        "window_shape": list(shape),
        "first_middle_last_window_shapes": [list(shape)] * 3,
        "windows_per_axis": counts,
        "window_count": math.prod(counts),
        "observed_trace_fraction": float(np.mean(mask)),
        "empty_window": not bool(np.any(mask)),
        "measured_iterations": iterations,
        "window_seconds": seconds,
        "prediction_finite": bool(np.isfinite(predicted).all()),
        "estimated_reconstruction_seconds": estimate,
        "estimate_scope": "linear_window_iteration_extrapolation_excludes_loading_scoring_saving",
        "estimate_informative": informative,
        "within_reconstruction_budget": estimate is not None and estimate < action_timeout_seconds,
        "estimated_svd_calls_upper_bound": (
            math.prod(counts) * settings["n_iterations"] * extra["selected_bin_count"]
            if method == "drr"
            else None
        ),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **extra,
    }
