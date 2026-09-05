"""Fixed SEG C3 NA relative-receiver grid shape and offset extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seis_interp.processing.neighbor_geometry import (
    RECEIVER_SPACING_M,
    RELATIVE_RECEIVER_X_MAX_M,
    RELATIVE_RECEIVER_X_MIN_M,
    RELATIVE_RECEIVER_Y_MAX_M,
    RELATIVE_RECEIVER_Y_MIN_M,
)

if TYPE_CHECKING:
    import pandas as pd

RECEIVER_X_COUNT = 8
RECEIVER_Y_COUNT = 68
_LATTICE_ATOL_M = 1.0e-6


def receiver_grid_offsets(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the ascending source-relative receiver offsets of the fixed grid."""
    relative_x = (table["receiver_x_m"] - table["source_x_m"]).to_numpy(dtype=np.float64)
    relative_y = (table["receiver_y_m"] - table["source_y_m"]).to_numpy(dtype=np.float64)
    x_values = np.sort(np.unique(relative_x))
    y_values = np.sort(np.unique(relative_y))
    if len(x_values) != RECEIVER_X_COUNT or len(y_values) != RECEIVER_Y_COUNT:
        raise ValueError(
            "selected data must expose the fixed 8 x 68 relative-receiver grid; "
            f"got {len(x_values)} x {len(y_values)}"
        )
    _require_fixed_offsets(
        x_values,
        minimum=RELATIVE_RECEIVER_X_MIN_M,
        maximum=RELATIVE_RECEIVER_X_MAX_M,
        count=RECEIVER_X_COUNT,
        axis="x",
    )
    _require_fixed_offsets(
        y_values,
        minimum=RELATIVE_RECEIVER_Y_MIN_M,
        maximum=RELATIVE_RECEIVER_Y_MAX_M,
        count=RECEIVER_Y_COUNT,
        axis="y",
    )
    return x_values, y_values


def _require_fixed_offsets(
    values: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    count: int,
    axis: str,
) -> None:
    expected = minimum + np.arange(count, dtype=np.float64) * RECEIVER_SPACING_M
    if expected[-1] != maximum:
        raise AssertionError("SEG C3 receiver-grid constants are inconsistent")
    if not np.allclose(values, expected, rtol=0.0, atol=_LATTICE_ATOL_M):
        mismatch = int(
            np.flatnonzero(~np.isclose(values, expected, rtol=0.0, atol=_LATTICE_ATOL_M))[0]
        )
        raise ValueError(
            f"relative-receiver {axis} offsets must match the fixed SEG C3 grid "
            f"[{minimum:g}, {maximum:g}] with {RECEIVER_SPACING_M:g} m spacing; "
            f"got {values[mismatch]:g} at index {mismatch}"
        )
