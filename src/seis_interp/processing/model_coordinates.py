"""Build numerical spatial features from stored physical trace coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from seis_interp.data.trace_schema import MODEL_SPATIAL_FEATURE_ORDER, PHYSICAL_COORDINATE_ORDER

__all__ = ["build_spatial_model_coordinates"]

_PHYSICAL_SPATIAL_COORDINATE_ORDER = PHYSICAL_COORDINATE_ORDER[1:]


def build_spatial_model_coordinates(trace_table: pd.DataFrame) -> np.ndarray:
    """Return ``[cmp_x_m, cmp_y_m, offset_m, azimuth_sin, azimuth_cos]``."""
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")

    missing = [
        column for column in _PHYSICAL_SPATIAL_COORDINATE_ORDER if column not in trace_table.columns
    ]
    if missing:
        raise ValueError(f"trace table is missing required coordinate columns: {missing}")

    non_numeric = [
        column
        for column in _PHYSICAL_SPATIAL_COORDINATE_ORDER
        if not (
            is_integer_dtype(trace_table[column].dtype) or is_float_dtype(trace_table[column].dtype)
        )
    ]
    if non_numeric:
        raise ValueError(f"spatial coordinate columns must be numeric: {non_numeric}")

    try:
        physical_coordinates = trace_table[list(_PHYSICAL_SPATIAL_COORDINATE_ORDER)].to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("spatial coordinates must be numeric") from error
    if not np.all(np.isfinite(physical_coordinates)):
        raise ValueError("spatial coordinates contain non-finite values")

    azimuth_rad = np.deg2rad(physical_coordinates[:, 3])
    feature_values = {
        "cmp_x_m": physical_coordinates[:, 0],
        "cmp_y_m": physical_coordinates[:, 1],
        "offset_m": physical_coordinates[:, 2],
        "azimuth_sin": np.sin(azimuth_rad),
        "azimuth_cos": np.cos(azimuth_rad),
    }
    return np.column_stack(tuple(feature_values[name] for name in MODEL_SPATIAL_FEATURE_ORDER))
