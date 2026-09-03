"""Fixed SEG C3 NA relative-receiver grid shape and offset extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

RECEIVER_X_COUNT = 8
RECEIVER_Y_COUNT = 68


def receiver_grid_offsets(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the ascending source-relative receiver offsets of the fixed grid."""
    relative_x = (table["receiver_x_m"] - table["source_x_m"]).to_numpy(dtype=np.float64)
    relative_y = (table["receiver_y_m"] - table["source_y_m"]).to_numpy(dtype=np.float64)
    x_values = np.unique(relative_x)
    y_values = np.unique(relative_y)
    if len(x_values) != RECEIVER_X_COUNT or len(y_values) != RECEIVER_Y_COUNT:
        raise ValueError(
            "selected data must expose the fixed 8 x 68 relative-receiver grid; "
            f"got {len(x_values)} x {len(y_values)}"
        )
    return np.sort(x_values), np.sort(y_values)
