"""Build volume-local SIREN coordinates and observed-only training samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.processing.geometry import compute_trace_geometry
from seis_interp.processing.model_coordinates import build_spatial_model_coordinates
from seis_interp.processing.normalization import NormalizationParameters, normalize_amplitudes
from seis_interp.processing.training_coordinates import (
    CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
    ModelCoordinateParameters,
    model_coordinate_parameters,
    normalize_training_spatial_coordinates,
    normalize_training_time_coordinate,
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


def build_c3_volume_siren_data(
    observed_volume: ObservedC3Volume,
    index_table: pd.DataFrame,
) -> C3VolumeSirenData:
    """Normalize known geometry and observed amplitudes without reading targets."""
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
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES, normalization, time_coordinate_scale=1.0
    )
    return C3VolumeSirenData(
        normalized_time=np.ascontiguousarray(
            normalize_training_time_coordinate(observed_volume.time_s, normalization, coordinates),
            dtype=np.float64,
        ),
        normalized_spatial=np.ascontiguousarray(
            normalize_training_spatial_coordinates(geometry, normalization, coordinates),
            dtype=np.float64,
        ),
        observed_flat_indices=np.ascontiguousarray(indices, dtype=np.int64),
        normalized_observed_amplitudes=np.ascontiguousarray(
            normalize_amplitudes(observed_amplitudes, normalization)
        ),
        normalization=normalization,
        model_coordinates=coordinates,
    )


def build_c3_volume_siren_sampler(
    data: C3VolumeSirenData,
    *,
    random_seed: int,
) -> RandomPointSampler:
    """Sample compact observed trace/time points using the existing sampler."""
    return RandomPointSampler(
        data.normalized_time,
        data.normalized_spatial[data.observed_flat_indices],
        data.normalized_observed_amplitudes,
        np.arange(len(data.observed_flat_indices), dtype=np.int64),
        random_seed=random_seed,
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
        {"cmp_x_m": cmp_x, "cmp_y_m": cmp_y, "offset_m": offset, "azimuth_deg": azimuth}
    )
