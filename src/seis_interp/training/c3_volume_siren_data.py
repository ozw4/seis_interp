"""Build volume-local SIREN coordinates and observed-only training samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.geometry import compute_trace_geometry
from seis_interp.processing.model_coordinates import build_spatial_model_coordinates
from seis_interp.processing.normalization import NormalizationParameters, normalize_amplitudes
from seis_interp.processing.trace_rms_interpolation import interpolate_trace_rms
from seis_interp.processing.training_coordinates import (
    CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
    ModelCoordinateParameters,
    model_coordinate_parameters,
    normalize_training_spatial_coordinates,
    normalize_training_time_coordinate,
    normalize_training_trace_time_offsets,
)
from seis_interp.training.amplitude_scaling import (
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
    validated_amplitude_scaling,
)
from seis_interp.training.point_sampler import RandomPointSampler

VOLUME_COORDINATE_SCOPE = "selected_volume_geometry"
VOLUME_AMPLITUDE_SCALE_SOURCE = "observed_trace_samples_only"

_GEOMETRY_COLUMNS = (
    "source_x_m",
    "source_y_m",
    "relative_receiver_x_m",
    "relative_receiver_y_m",
)


@dataclass(frozen=True)
class C3VolumeSirenData:
    """All known volume coordinates with amplitudes compacted to observed traces.

    Spatial coordinates and observed indices follow C-order volume flattening.
    Only observed amplitudes are retained; the coordinate ranges use all known
    geometry while the amplitude RMS uses observed trace samples exclusively.
    """

    normalized_time: np.ndarray
    normalized_spatial: np.ndarray
    observed_flat_indices: np.ndarray
    normalized_observed_amplitudes: np.ndarray
    normalization: NormalizationParameters
    model_coordinates: ModelCoordinateParameters
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING
    trace_amplitude_scales: np.ndarray | None = None
    trace_array_rows: np.ndarray | None = None
    scale_interpolation: dict[str, object] | None = None
    normalized_time_offsets: np.ndarray | None = None


def validate_c3_volume_siren_scaling(
    amplitude_scaling: object,
    scale_interpolation: object,
) -> tuple[str, dict[str, object] | None]:
    """Validate the explicit observed-only scale interpolation contract."""
    mode = validated_amplitude_scaling(amplitude_scaling)
    if mode == TRAIN_GLOBAL_RMS_SCALING:
        if scale_interpolation is not None:
            raise ValueError("scale_interpolation must be absent for train_global_rms")
        return mode, None
    if not isinstance(scale_interpolation, Mapping) or set(scale_interpolation) != {
        "neighbors",
        "power",
        "distance_scales_m",
    }:
        raise ValueError(
            "per_trace_rms scale_interpolation requires exactly neighbors, power, distance_scales_m"
        )
    neighbors = scale_interpolation["neighbors"]
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors <= 0:
        raise ValueError("scale_interpolation.neighbors must be a positive integer")
    power = _positive_scale(scale_interpolation["power"], "scale_interpolation.power")
    distances = scale_interpolation["distance_scales_m"]
    if (
        isinstance(distances, (str, bytes))
        or not isinstance(distances, Sequence)
        or len(distances) != 4
    ):
        raise ValueError(
            "scale_interpolation.distance_scales_m must contain four positive finite numbers"
        )
    return mode, {
        "neighbors": int(neighbors),
        "power": power,
        "distance_scales_m": [
            _positive_scale(value, "scale_interpolation.distance_scales_m") for value in distances
        ],
    }


def _positive_scale(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def build_c3_volume_siren_data(
    observed_volume: ObservedC3Volume,
    index_table: pd.DataFrame,
    *,
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    scale_interpolation: Mapping | None = None,
    coordinate_features: str = CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
    time_coordinate_scale: float = 1.0,
    relative_receiver_y_time_shear_s_per_m: float = 0.0,
) -> C3VolumeSirenData:
    """Normalize known geometry and observed amplitudes without reading targets."""
    mode, interpolation = validate_c3_volume_siren_scaling(amplitude_scaling, scale_interpolation)
    _validate_volume_alignment(observed_volume, index_table)
    indices = np.flatnonzero(observed_volume.observed_trace_mask.reshape(-1))
    if not len(indices):
        raise ValueError("volume must contain at least one observed trace")
    observed_amplitudes = observed_volume.values[:, observed_volume.observed_trace_mask].T
    if not np.all(np.isfinite(observed_amplitudes)):
        raise ValueError("observed amplitudes must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        energy = np.sum(np.square(observed_amplitudes, dtype=np.float64), dtype=np.float64)
        amplitude_rms = float(np.sqrt(energy / observed_amplitudes.size))
    if not np.isfinite(amplitude_rms) or amplitude_rms <= 0:
        raise ValueError("observed amplitude RMS must be positive and finite")

    geometry = _geometry_table(index_table)
    spatial = build_spatial_model_coordinates(geometry)
    normalization = NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(
            float(np.min(observed_volume.time_s)),
            *(float(value) for value in spatial[:, :3].min(axis=0)),
            -1.0,
            -1.0,
        ),
        coordinate_max=(
            float(np.max(observed_volume.time_s)),
            *(float(value) for value in spatial[:, :3].max(axis=0)),
            1.0,
            1.0,
        ),
        amplitude_rms=amplitude_rms,
    )
    coordinates = model_coordinate_parameters(
        coordinate_features,
        normalization,
        time_coordinate_scale=time_coordinate_scale,
        relative_receiver_y_time_shear_s_per_m=relative_receiver_y_time_shear_s_per_m,
    )
    trace_scales = None
    trace_rows = None
    if mode == PER_TRACE_RMS_SCALING:
        assert interpolation is not None
        trace_rows = np.array(validated_array_rows(index_table), dtype=np.int64, copy=True)
        if np.any(trace_rows < 0):
            raise ValueError("per_trace_rms array rows must be nonnegative")
        normalized_amplitudes, trace_scales = _per_trace_amplitudes(
            observed_amplitudes, indices, index_table, interpolation
        )
    else:
        normalized_amplitudes = normalize_amplitudes(observed_amplitudes, normalization)
    normalized_time = np.ascontiguousarray(
        normalize_training_time_coordinate(observed_volume.time_s, normalization, coordinates),
        dtype=np.float64,
    )
    normalized_spatial = np.ascontiguousarray(
        normalize_training_spatial_coordinates(geometry, normalization, coordinates),
        dtype=np.float64,
    )
    return C3VolumeSirenData(
        normalized_time=normalized_time,
        normalized_spatial=normalized_spatial,
        observed_flat_indices=np.ascontiguousarray(indices, dtype=np.int64),
        normalized_observed_amplitudes=np.ascontiguousarray(normalized_amplitudes),
        normalization=normalization,
        model_coordinates=coordinates,
        amplitude_scaling=mode,
        trace_amplitude_scales=trace_scales,
        trace_array_rows=trace_rows,
        scale_interpolation=interpolation,
        normalized_time_offsets=normalize_training_trace_time_offsets(
            normalized_spatial, coordinates
        ),
    )


def _per_trace_amplitudes(amplitudes, observed_indices, index_table, interpolation):
    observed_scales = np.sqrt(np.mean(np.square(amplitudes, dtype=np.float64), axis=1))
    divisor = np.where(observed_scales == 0, 1.0, observed_scales)
    normalized = (amplitudes / divisor[:, None]).astype(
        np.result_type(amplitudes.dtype, np.float32)
    )
    coordinates = index_table[list(_GEOMETRY_COLUMNS)].to_numpy(dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        coordinates = coordinates / np.asarray(interpolation["distance_scales_m"])
    targets = np.ones(len(index_table), dtype=bool)
    targets[observed_indices] = False
    scales = np.empty(len(index_table), dtype=np.float64)
    scales[observed_indices] = observed_scales
    scales[targets] = interpolate_trace_rms(
        coordinates[observed_indices],
        observed_scales,
        coordinates[targets],
        observed_array_rows=index_table["array_row"].to_numpy()[observed_indices],
        neighbors=interpolation["neighbors"],
        power=interpolation["power"],
    )
    return normalized, scales


def build_c3_volume_siren_sampler(
    data: C3VolumeSirenData,
    *,
    random_seed: int,
) -> RandomPointSampler:
    """Sample compact observed trace/time points using the existing sampler."""
    time_offsets = {}
    if data.normalized_time_offsets is not None:
        time_offsets["normalized_time_offsets"] = data.normalized_time_offsets[
            data.observed_flat_indices
        ]
    return RandomPointSampler(
        data.normalized_time,
        data.normalized_spatial[data.observed_flat_indices],
        data.normalized_observed_amplitudes,
        np.arange(len(data.observed_flat_indices), dtype=np.int64),
        random_seed=random_seed,
        amplitude_scaling=data.amplitude_scaling,
        **time_offsets,
    )


def _validate_volume_alignment(volume: ObservedC3Volume, index_table: pd.DataFrame) -> None:
    if not isinstance(volume, ObservedC3Volume):
        raise ValueError("observed_volume must be an ObservedC3Volume")
    values = volume.values
    if not isinstance(values, np.ndarray) or values.ndim != 5 or not values.size:
        raise ValueError("observed volume values must be a nonempty five-dimensional array")
    if values.dtype.kind not in "fiu":
        raise ValueError("observed volume values must contain real numeric amplitudes")
    times = volume.time_s
    if (
        not isinstance(times, np.ndarray)
        or times.shape != (values.shape[0],)
        or times.dtype.kind not in "fiu"
        or not np.all(np.isfinite(times))
    ):
        raise ValueError("time_s must be a finite real array matching the volume time axis")
    mask = volume.observed_trace_mask
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != values.shape[1:]:
        raise ValueError("observed_trace_mask must be boolean and match the volume spatial shape")
    if not isinstance(volume.array_rows, np.ndarray) or volume.array_rows.shape != values.shape[1:]:
        raise ValueError("array_rows must match the volume spatial shape")
    if not isinstance(index_table, pd.DataFrame) or len(index_table) != volume.array_rows.size:
        raise ValueError("index table row count must match the volume spatial trace count")
    if "array_row" not in index_table:
        raise ValueError("index table is missing required column: array_row")
    if not np.array_equal(index_table["array_row"].to_numpy(), volume.array_rows.reshape(-1)):
        raise ValueError("index table array_row order must match the volume flat array rows")


def _geometry_table(index_table: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _GEOMETRY_COLUMNS if column not in index_table]
    if missing:
        raise ValueError(f"index table is missing required geometry columns: {missing}")
    try:
        geometry = index_table[list(_GEOMETRY_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("index table geometry must contain finite real values") from error
    if not np.all(np.isfinite(geometry)):
        raise ValueError("index table geometry must contain finite real values")
    source_x, source_y, relative_x, relative_y = geometry.T
    cmp_x, cmp_y, offset, azimuth = compute_trace_geometry(
        source_x, source_y, source_x + relative_x, source_y + relative_y
    )
    return pd.DataFrame(
        {
            "cmp_x_m": cmp_x,
            "cmp_y_m": cmp_y,
            "offset_m": offset,
            "azimuth_deg": azimuth,
            "source_x_m": source_x,
            "source_y_m": source_y,
            "receiver_x_m": source_x + relative_x,
            "receiver_y_m": source_y + relative_y,
        }
    )
