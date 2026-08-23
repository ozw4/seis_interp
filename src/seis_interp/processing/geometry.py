"""Pure NumPy primitives for SEG-Y coordinate scalars and trace geometry."""

from __future__ import annotations

import numpy as np

_GEOMETRY_INPUT_NAMES = ("source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m")


def apply_coordinate_scalar(
    values: np.ndarray,
    scalars: np.ndarray,
) -> np.ndarray:
    """Apply the SEG-Y coordinate scalar rule element-wise.

    A positive scalar multiplies, a negative scalar divides by its absolute
    value, and a zero scalar leaves the coordinate unchanged. Both inputs must
    be one-dimensional and share the same shape; the inputs are not modified.
    """
    value_array = np.asarray(values)
    scalar_array = np.asarray(scalars)

    if value_array.ndim != 1 or scalar_array.ndim != 1:
        raise ValueError(
            "values and scalars must be one-dimensional, got "
            f"{value_array.ndim} and {scalar_array.ndim} dimensions"
        )
    if value_array.shape != scalar_array.shape:
        raise ValueError(
            f"values and scalars must share a shape, got {value_array.shape} "
            f"and {scalar_array.shape}"
        )

    scaled = value_array.astype(np.float64)
    scalar_float = scalar_array.astype(np.float64)

    positive = scalar_float > 0.0
    negative = scalar_float < 0.0
    scaled[positive] = scaled[positive] * scalar_float[positive]
    scaled[negative] = scaled[negative] / np.abs(scalar_float[negative])
    return scaled


def compute_trace_geometry(
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CMP, offset and azimuth from source and receiver coordinates.

    Returns ``(cmp_x_m, cmp_y_m, offset_m, azimuth_deg)``. The azimuth follows
    the paper argument order ``atan2(dx, dy)`` with ``d = source - receiver``
    and is wrapped to ``[0, 360)`` degrees. Inputs must be one-dimensional,
    share the same shape, be finite, and are not modified.
    """
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (source_x_m, source_y_m, receiver_x_m, receiver_y_m)
    )
    for name, array in zip(_GEOMETRY_INPUT_NAMES, arrays, strict=True):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.ndim} dimensions")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(
            "source and receiver arrays must share a shape, got "
            f"{[array.shape for array in arrays]}"
        )

    source_x, source_y, receiver_x, receiver_y = arrays
    cmp_x_m = 0.5 * (source_x + receiver_x)
    cmp_y_m = 0.5 * (source_y + receiver_y)

    dx_m = source_x - receiver_x
    dy_m = source_y - receiver_y

    offset_m = np.hypot(dx_m, dy_m)
    azimuth_deg = np.mod(np.degrees(np.arctan2(dx_m, dy_m)), 360.0)
    return cmp_x_m, cmp_y_m, offset_m, azimuth_deg
