"""Query a C3 volume in bounded coordinate chunks and restore observed traces."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import denormalize_amplitudes
from seis_interp.processing.training_coordinates import (
    model_coordinate_parameters,
    normalize_training_time_coordinate,
    normalize_training_trace_time_offsets,
)
from seis_interp.training.amplitude_scaling import PER_TRACE_RMS_SCALING
from seis_interp.training.c3_volume_siren_data import (
    C3VolumeSirenData,
    validate_c3_volume_siren_scaling,
)
from seis_interp.training.point_sampler import build_trace_coordinate_points
from seis_interp.training.prediction import predict_points


@dataclass(frozen=True)
class C3VolumeSirenPrediction:
    """Physical volume prediction and observed fit before exact reinsertion."""

    values: np.ndarray
    observed_model_rmse_before_reinsertion: float
    observed_model_max_abs_error_before_reinsertion: float
    predicted_point_count: int


def predict_c3_volume_siren(
    model: Siren,
    data: C3VolumeSirenData,
    observed_volume: ObservedC3Volume,
    *,
    batch_size: int,
    device: torch.device | str,
) -> C3VolumeSirenPrediction:
    """Predict trace chunks, collect observed fit, and reinsert observed values.

    Coordinates cover at most ``max(1, batch_size // time_count)`` traces at
    once; ``predict_points`` bounds each model forward by ``batch_size``.
    Target truth is never read. Return a C-contiguous volume in the input
    float32/float64 dtype and preserve the model's original training mode.
    """
    _validate_prediction_inputs(model, data, observed_volume, batch_size)
    time_count = len(data.normalized_time)
    trace_count = len(data.normalized_spatial)
    traces_per_chunk = max(1, int(batch_size) // time_count)
    trace_predictions = np.empty((trace_count, time_count), dtype=observed_volume.values.dtype)
    observed_mask = observed_volume.observed_trace_mask.reshape(-1)
    observed_values = observed_volume.values.reshape(time_count, trace_count)
    observed_error_energy = 0.0
    observed_maximum_error = 0.0
    time_offsets = {}
    if data.normalized_time_offsets is not None:
        time_offsets["normalized_time_offsets"] = data.normalized_time_offsets
    for start in range(0, trace_count, traces_per_chunk):
        stop = min(start + traces_per_chunk, trace_count)
        rows = np.arange(start, stop, dtype=np.int64)
        coordinates = build_trace_coordinate_points(
            data.normalized_time, data.normalized_spatial, rows, **time_offsets
        )
        normalized = predict_points(model, coordinates, batch_size=batch_size, device=device)
        if data.amplitude_scaling == PER_TRACE_RMS_SCALING:
            assert data.trace_amplitude_scales is not None
            physical = (
                normalized.reshape(len(rows), time_count).astype(np.float64)
                * data.trace_amplitude_scales[rows, None]
            )
        else:
            physical = denormalize_amplitudes(normalized, data.normalization)
        chunk = np.asarray(physical.reshape(len(rows), time_count), dtype=trace_predictions.dtype)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("physical prediction must contain only finite values")
        trace_predictions[start:stop] = chunk
        selected = observed_mask[start:stop]
        if np.any(selected):
            error = chunk[selected].astype(np.float64) - observed_values[:, rows[selected]].T
            observed_error_energy += float(np.sum(np.square(error), dtype=np.float64))
            observed_maximum_error = max(observed_maximum_error, float(np.max(np.abs(error))))

    values = np.ascontiguousarray(trace_predictions.T.reshape(observed_volume.values.shape))
    values[:, observed_volume.observed_trace_mask] = observed_volume.values[
        :, observed_volume.observed_trace_mask
    ]
    observed_sample_count = len(data.observed_flat_indices) * time_count
    return C3VolumeSirenPrediction(
        values=values,
        observed_model_rmse_before_reinsertion=float(
            np.sqrt(observed_error_energy / observed_sample_count)
        ),
        observed_model_max_abs_error_before_reinsertion=observed_maximum_error,
        predicted_point_count=values.size,
    )


def _validate_prediction_inputs(
    model: Siren,
    data: C3VolumeSirenData,
    observed_volume: ObservedC3Volume,
    batch_size: int,
) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    values = observed_volume.values
    if values.ndim != 5 or not values.size:
        raise ValueError("observed volume must be a nonempty five-dimensional array")
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("observed volume dtype must be float32 or float64")
    if data.normalized_time.shape != (values.shape[0],):
        raise ValueError("normalized time count must match the observed volume")
    trace_count = int(np.prod(values.shape[1:]))
    if data.normalized_spatial.shape != (trace_count, data.model_coordinates.input_features - 1):
        raise ValueError("normalized spatial shape must match the volume trace count and features")
    mask = observed_volume.observed_trace_mask
    if mask.dtype != np.bool_ or mask.shape != values.shape[1:]:
        raise ValueError("observed trace mask must be boolean and match the volume spatial shape")
    if not len(data.observed_flat_indices) or not np.array_equal(
        data.observed_flat_indices, np.flatnonzero(mask.reshape(-1))
    ):
        raise ValueError("observed flat indices must match the nonempty observed trace mask")
    if model.input_features != data.normalized_spatial.shape[1] + 1:
        raise ValueError("model input width must match the normalized coordinate width")
    expected_coordinates = model_coordinate_parameters(
        data.model_coordinates.coordinate_features,
        data.normalization,
        time_coordinate_scale=data.model_coordinates.time_coordinate_scale,
        relative_receiver_y_time_shear_s_per_m=(
            data.model_coordinates.relative_receiver_y_time_shear_s_per_m
        ),
    )
    expected_time = normalize_training_time_coordinate(
        observed_volume.time_s, data.normalization, expected_coordinates
    )
    if data.model_coordinates != expected_coordinates or not np.array_equal(
        data.normalized_time, expected_time
    ):
        raise ValueError("normalization and model coordinate parameters must be consistent")
    expected_offsets = normalize_training_trace_time_offsets(
        data.normalized_spatial, expected_coordinates
    )
    offsets = data.normalized_time_offsets
    if expected_offsets is None:
        if offsets is not None:
            raise ValueError("normalized_time_offsets must be absent when coordinate shear is zero")
    elif (
        not isinstance(offsets, np.ndarray)
        or offsets.dtype != np.float64
        or offsets.shape != (trace_count,)
        or not np.all(np.isfinite(offsets))
        or not np.array_equal(offsets, expected_offsets)
    ):
        raise ValueError(
            "normalized_time_offsets must match model coordinate shear and spatial rows"
        )
    mode, _ = validate_c3_volume_siren_scaling(data.amplitude_scaling, data.scale_interpolation)
    if mode == PER_TRACE_RMS_SCALING:
        scales = data.trace_amplitude_scales
        if (
            not isinstance(scales, np.ndarray)
            or scales.shape != (trace_count,)
            or scales.dtype != np.float64
            or not np.all(np.isfinite(scales))
            or np.any(scales < 0)
        ):
            raise ValueError(
                "per_trace_rms trace_amplitude_scales must be a finite nonnegative "
                "float64 vector matching volume traces"
            )
        rows = data.trace_array_rows
        if (
            not isinstance(rows, np.ndarray)
            or rows.shape != (trace_count,)
            or rows.dtype != np.int64
            or len(np.unique(rows)) != trace_count
            or np.any(rows < 0)
            or not np.array_equal(rows, observed_volume.array_rows.reshape(-1))
        ):
            raise ValueError(
                "per_trace_rms trace_array_rows must be unique nonnegative int64 rows "
                "matching the volume order"
            )
    elif data.trace_amplitude_scales is not None:
        raise ValueError("train_global_rms must not contain trace_amplitude_scales")
    elif data.trace_array_rows is not None:
        raise ValueError("train_global_rms must not contain trace_array_rows")
