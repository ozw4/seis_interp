"""Frozen CCNet-5D inference with exact core coverage and observed reinsertion."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.processing.ccnet5d_tiles import iter_ccnet5d_tiles, validate_ccnet5d_shape
from seis_interp.training.amplitude_scaling import (
    normalize_by_global_rms,
    restore_physical_amplitude,
)


@dataclass(frozen=True)
class CCNet5DVolumePrediction:
    values: np.ndarray
    coverage_counts: np.ndarray
    observed_model_rmse_before_reinsertion: float
    observed_model_max_abs_error_before_reinsertion: float
    tile_count: int
    maximum_input_shape: tuple[int, ...]


def predict_ccnet5d_volume(
    model: CCNet5D,
    observed_volume: ObservedC3Volume,
    *,
    amplitude_rms: float,
    core_shape: tuple[int, ...],
    device: torch.device | str,
) -> CCNet5DVolumePrediction:
    """Predict real, halo-expanded slices using only observed amplitudes.

    No external padding, blending, empty-tile skipping, or scale fitting is
    performed. Each Conv handles the true volume boundary at every layer.
    """
    values = observed_volume.values
    if values.ndim != 5 or not values.size or values.dtype not in (np.float32, np.float64):
        raise ValueError(
            "observed volume must be a nonempty five-dimensional float32/float64 array"
        )
    mask = observed_volume.observed_trace_mask
    if mask.shape != values.shape[1:] or mask.dtype != np.bool_:
        raise ValueError("observed trace mask must be boolean and match the spatial shape")
    if not np.all(np.isfinite(values[:, mask])):
        raise ValueError("observed amplitudes must be finite")
    if (
        isinstance(amplitude_rms, bool)
        or not isinstance(amplitude_rms, Real)
        or not np.isfinite(amplitude_rms)
        or amplitude_rms <= 0
    ):
        raise ValueError("amplitude_rms must be positive and finite")
    core = validate_ccnet5d_shape(core_shape, "core_shape")
    # np.where, not multiplication: target placeholders may contain NaN.
    source = normalize_by_global_rms(np.where(mask[None, ...], values, 0), amplitude_rms)
    predicted = np.empty_like(values)
    sample_coverage = np.zeros(values.shape, dtype=np.uint16)
    parameter = next(model.parameters())
    tile_count = 0
    maximum_input_shape = (0,) * 5
    original_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for tile in iter_ccnet5d_tiles(values.shape, core, halo_radius=model.halo_radius):
                block = np.ascontiguousarray(source[tile.input_slices])
                inputs = torch.as_tensor(block, dtype=parameter.dtype, device=device)[None, None]
                output = model(inputs)[0, 0][tile.local_core_slices]
                physical = restore_physical_amplitude(output.cpu().numpy(), amplitude_rms)
                predicted[tile.core_slices] = physical
                sample_coverage[tile.core_slices] += 1
                tile_count += 1
                maximum_input_shape = tuple(
                    max(previous, actual)
                    for previous, actual in zip(maximum_input_shape, block.shape, strict=True)
                )
    finally:
        model.train(original_mode)
    if np.any(sample_coverage == 0):
        raise ValueError("CCNet-5D tiles must cover every output sample")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("physical prediction must contain only finite values")
    error = predicted[:, mask].astype(np.float64) - values[:, mask].astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0
    maximum_error = float(np.max(np.abs(error))) if error.size else 0.0
    predicted[:, mask] = values[:, mask]
    return CCNet5DVolumePrediction(
        values=np.ascontiguousarray(predicted),
        coverage_counts=np.ascontiguousarray(sample_coverage.min(axis=0)),
        observed_model_rmse_before_reinsertion=rmse,
        observed_model_max_abs_error_before_reinsertion=maximum_error,
        tile_count=tile_count,
        maximum_input_shape=maximum_input_shape,
    )
