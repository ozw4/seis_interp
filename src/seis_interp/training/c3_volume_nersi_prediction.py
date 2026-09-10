"""Predict every NeRSI profile and restore a physical C3 volume."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.nersi import Nersi
from seis_interp.training.c3_volume_nersi_data import (
    C3VolumeNersiData,
    nersi_profiles_to_volume,
)


@dataclass(frozen=True)
class C3VolumeNersiPrediction:
    """Physical prediction plus observed fit before exact reinsertion."""

    values: np.ndarray
    predicted_profile_count: int
    observed_model_rmse_before_reinsertion: float
    observed_model_max_abs_error_before_reinsertion: float


def predict_c3_volume_nersi(
    model: Nersi,
    data: C3VolumeNersiData,
    observed: ObservedC3Volume,
    *,
    batch_size: int,
    device: torch.device | str,
) -> C3VolumeNersiPrediction:
    """Predict all stable profile keys, restore scale once, and reinsert observations."""
    count = _validate_prediction_inputs(model, data, observed, batch_size)
    profile_shape = tuple(data.profile_shape)
    normalized_profiles = np.empty((count, 1, *profile_shape), dtype=np.float32)

    was_training = model.training
    model.to(device)
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, count, int(batch_size)):
                stop = min(start + int(batch_size), count)
                coordinates = torch.as_tensor(
                    np.ascontiguousarray(data.normalized_coordinates[start:stop]),
                    dtype=torch.float32,
                    device=device,
                )
                prediction = model(coordinates)
                expected_shape = (stop - start, 1, *profile_shape)
                if tuple(prediction.shape) != expected_shape:
                    raise ValueError(
                        "model output must have shape "
                        f"{expected_shape}, got {tuple(prediction.shape)}"
                    )
                if not prediction.is_floating_point() or not bool(torch.isfinite(prediction).all()):
                    raise ValueError("normalized model prediction must contain finite values")
                normalized_profiles[start:stop] = (
                    prediction.detach().to(device="cpu", dtype=torch.float32).numpy()
                )
    finally:
        model.train(was_training)

    physical_profiles = np.ascontiguousarray(
        normalized_profiles * np.float32(data.amplitude_scale), dtype=np.float32
    )
    if not np.all(np.isfinite(physical_profiles)):
        raise ValueError("physical model prediction must contain finite values")
    predicted_values = np.ascontiguousarray(
        nersi_profiles_to_volume(physical_profiles, tuple(data.spatial_shape)),
        dtype=np.float32,
    )

    observed_values = observed.values[:, observed.observed_trace_mask]
    predicted_observed = predicted_values[:, observed.observed_trace_mask]
    error = predicted_observed.astype(np.float64) - observed_values.astype(np.float64)
    error_energy = float(np.sum(np.square(error), dtype=np.float64))
    observed_rmse = float(np.sqrt(error_energy / error.size))
    observed_maximum = float(np.max(np.abs(error)))

    predicted_values[:, observed.observed_trace_mask] = observed_values.astype(
        np.float32, copy=False
    )
    return C3VolumeNersiPrediction(
        values=np.ascontiguousarray(predicted_values, dtype=np.float32),
        predicted_profile_count=count,
        observed_model_rmse_before_reinsertion=observed_rmse,
        observed_model_max_abs_error_before_reinsertion=observed_maximum,
    )


def _validate_prediction_inputs(
    model: Nersi,
    data: C3VolumeNersiData,
    observed: ObservedC3Volume,
    batch_size: int,
) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(data, C3VolumeNersiData):
        raise TypeError("data must be C3VolumeNersiData")
    if not isinstance(observed, ObservedC3Volume):
        raise TypeError("observed must be an ObservedC3Volume")
    values = observed.values
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 5
        or not values.size
        or values.dtype.kind not in "fiu"
        or values.dtype.kind == "b"
    ):
        raise ValueError("observed values must be a nonempty real five-dimensional array")
    spatial_shape = tuple(int(value) for value in values.shape[1:])
    if tuple(data.spatial_shape) != spatial_shape:
        raise ValueError("data spatial_shape must match the observed volume spatial shape")
    profile_shape = (values.shape[0], values.shape[-1])
    if tuple(data.profile_shape) != profile_shape:
        raise ValueError("data profile_shape must match the observed volume time and receiver-y")
    model_profile_shape = getattr(model, "profile_shape", None)
    if model_profile_shape is None or tuple(model_profile_shape) != profile_shape:
        raise ValueError("model profile_shape must match data profile_shape")
    profile_count = math.prod(spatial_shape[:-1])
    coordinates = data.normalized_coordinates
    if (
        not isinstance(coordinates, np.ndarray)
        or coordinates.shape != (profile_count, 3)
        or coordinates.dtype.kind != "f"
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("normalized_coordinates must be finite with shape (profile_count, 3)")
    mask = observed.observed_trace_mask
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != spatial_shape:
        raise ValueError("observed trace mask must be boolean and match the spatial shape")
    profile_mask = data.observed_trace_mask
    expected_profile_mask = mask.reshape(profile_count, spatial_shape[-1])
    if (
        not isinstance(profile_mask, np.ndarray)
        or profile_mask.dtype != np.bool_
        or profile_mask.shape != expected_profile_mask.shape
        or not np.array_equal(profile_mask, expected_profile_mask)
    ):
        raise ValueError("data observed_trace_mask must match the observed volume in profile order")
    if not np.any(mask):
        raise ValueError("observed volume must contain at least one observed trace")
    observed_values = values[:, mask]
    if not np.all(np.isfinite(observed_values)):
        raise ValueError("observed amplitudes must contain finite values")
    scale = data.amplitude_scale
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float, np.integer, np.floating))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ValueError("amplitude_scale must be positive and finite")
    return profile_count
