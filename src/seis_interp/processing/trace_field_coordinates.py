"""The continuous coordinates a trace field reads, including the log-offset axis.

The four local coordinates are normalized by the grid extent, so one unit is the
whole acquisition in that axis. The optional fifth is the decimal logarithm of the
offset over a declared span, so one unit is ``OFFSET_DECADES`` decades of offset
on every dataset rather than a span the data happens to cover. A logarithm is the
axis that bounds the *relative* offset difference between a query and its
neighbours, which is the quantity a seismic waveform is sensitive to.
"""

from numbers import Integral

import numpy as np

LOCAL_COLUMNS = (
    "source_line_continuous",
    "source_point_continuous",
    "receiver_line_continuous",
    "receiver_point_continuous",
)
OFFSET_COLUMN = "offset_m"
OFFSET_FLOOR_M = 1.0
OFFSET_DECADES = 4.0


def field_coordinates(geometry, spatial_shape, input_features):
    """``(trace, input_features)`` float64 coordinates in the geometry's row order.

    ``input_features`` is four local coordinates, or five with the log-offset axis
    appended. Nothing here reads a waveform.
    """
    if isinstance(input_features, bool) or not isinstance(input_features, Integral):
        raise ValueError("input_features must be an integer")
    if input_features not in (4, 5):
        raise ValueError("a trace field reads four or five coordinates")
    extent = np.maximum(np.asarray(spatial_shape, dtype=np.float64) - 1, 1)
    if extent.shape != (4,):
        raise ValueError("spatial_shape must hold the four grid axes")
    local = geometry[list(LOCAL_COLUMNS)].to_numpy(dtype=np.float64) / extent
    if input_features == 4:
        return local
    offset = geometry[OFFSET_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(offset).all() or np.any(offset < 0):
        raise ValueError("offsets must be finite and nonnegative")
    decades = np.log10(np.maximum(offset, OFFSET_FLOOR_M)) / OFFSET_DECADES
    return np.column_stack((local, np.clip(decades, 0.0, 1.0)))
