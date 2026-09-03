from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.c3_receiver_grid import (
    RECEIVER_X_COUNT,
    RECEIVER_Y_COUNT,
    receiver_grid_offsets,
)


def _grid_table(
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    *,
    source_x_m: float = 1000.0,
    source_y_m: float = 2000.0,
) -> pd.DataFrame:
    rows = [
        {
            "source_x_m": source_x_m,
            "source_y_m": source_y_m,
            "receiver_x_m": source_x_m + float(offset_x),
            "receiver_y_m": source_y_m + float(offset_y),
        }
        for offset_x in x_offsets
        for offset_y in y_offsets
    ]
    return pd.DataFrame(rows)


def _full_offsets() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(RECEIVER_X_COUNT, dtype=np.float64) * 12.5,
        np.arange(RECEIVER_Y_COUNT, dtype=np.float64) * 6.25,
    )


def test_grid_shape_constants_are_fixed() -> None:
    assert RECEIVER_X_COUNT == 8
    assert RECEIVER_Y_COUNT == 68


def test_returns_ascending_offsets() -> None:
    x_offsets, y_offsets = _full_offsets()
    result_x, result_y = receiver_grid_offsets(_grid_table(x_offsets, y_offsets))

    assert len(result_x) == RECEIVER_X_COUNT
    assert len(result_y) == RECEIVER_Y_COUNT
    np.testing.assert_array_equal(result_x, x_offsets)
    np.testing.assert_array_equal(result_y, y_offsets)


def test_result_does_not_depend_on_row_order() -> None:
    x_offsets, y_offsets = _full_offsets()
    table = _grid_table(x_offsets, y_offsets)
    shuffled = table.sample(frac=1.0, random_state=7).reset_index(drop=True)

    result_x, result_y = receiver_grid_offsets(shuffled)

    np.testing.assert_array_equal(result_x, x_offsets)
    np.testing.assert_array_equal(result_y, y_offsets)


def test_uses_source_relative_offsets_not_absolute_coordinates() -> None:
    x_offsets, y_offsets = _full_offsets()
    first = _grid_table(x_offsets, y_offsets, source_x_m=0.0, source_y_m=0.0)
    second = _grid_table(x_offsets, y_offsets, source_x_m=137.0, source_y_m=-59.0)
    table = pd.concat([first, second], ignore_index=True)

    result_x, result_y = receiver_grid_offsets(table)

    np.testing.assert_array_equal(result_x, x_offsets)
    np.testing.assert_array_equal(result_y, y_offsets)


def test_rejects_wrong_x_count() -> None:
    x_offsets, y_offsets = _full_offsets()
    table = _grid_table(x_offsets[:-1], y_offsets)

    with pytest.raises(ValueError, match=r"fixed 8 x 68 relative-receiver grid; got 7 x 68"):
        receiver_grid_offsets(table)


def test_rejects_wrong_y_count() -> None:
    x_offsets, y_offsets = _full_offsets()
    table = _grid_table(x_offsets, y_offsets[:-2])

    with pytest.raises(ValueError, match=r"fixed 8 x 68 relative-receiver grid; got 8 x 66"):
        receiver_grid_offsets(table)
