from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.c3_volume_index import (
    VOLUME_INDEX_COLUMNS,
    build_c3_volume_index,
    selected_spatial_shape,
    validated_index_range,
)
from tests.fixtures.c3_volume_artifacts import make_c3_trace_table


def _build(table: pd.DataFrame, candidates: np.ndarray | None = None) -> pd.DataFrame:
    return build_c3_volume_index(
        table,
        table["array_row"].to_numpy() if candidates is None else candidates,
        source_line_range=(0, 2),
        shot_in_line_range=(1, 3),
        relative_receiver_x_range=(2, 5),
        relative_receiver_y_range=(10, 14),
    )


def test_maps_staggered_lines_by_line_local_shot_and_sorts_cells() -> None:
    table = make_c3_trace_table().sample(frac=1.0, random_state=7)

    result = _build(table)

    assert result.columns.tolist() == list(VOLUME_INDEX_COLUMNS)
    assert len(result) == np.prod((2, 2, 3, 4))
    assert result.iloc[0]["source_y_m"] == 80.0
    second_line = result[result["source_line_index"] == 1]
    assert second_line.iloc[0]["source_y_m"] == 120.0
    indices = result.iloc[:, 2:6].to_numpy()
    assert np.array_equal(
        np.ravel_multi_index(indices.T, (2, 2, 3, 4)),
        np.arange(len(result)),
    )
    assert all(dtype == np.dtype("int64") for dtype in result.dtypes.iloc[:6])
    assert all(dtype == np.dtype("float64") for dtype in result.dtypes.iloc[6:])


def test_mapping_is_independent_of_input_index_order_and_ffid_arithmetic() -> None:
    original = make_c3_trace_table()
    shuffled = original.sample(frac=1.0, random_state=3).set_axis(
        np.arange(10000, 10000 + len(original))
    )

    first = _build(original).set_index("array_row").iloc[:, 1:5]
    second = _build(shuffled).set_index("array_row").iloc[:, 1:5]

    pd.testing.assert_frame_equal(first.sort_index(), second.sort_index())


def test_rejects_missing_and_duplicate_cells() -> None:
    table = make_c3_trace_table()
    complete = _build(table)
    candidates = table["array_row"].to_numpy()
    candidates = candidates[candidates != complete.iloc[0]["array_row"]]
    with pytest.raises(ValueError, match="not dense"):
        _build(table, candidates)

    duplicate = table.iloc[[0]].copy()
    duplicate["array_row"] = len(table)
    duplicated_table = pd.concat([table, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate spatial cells"):
        build_c3_volume_index(
            duplicated_table,
            duplicated_table["array_row"].to_numpy(),
            source_line_range=(0, 1),
            shot_in_line_range=(0, 1),
            relative_receiver_x_range=(0, 1),
            relative_receiver_y_range=(0, 1),
        )


@pytest.mark.parametrize("value", [(0, 0), (-1, 2), (2, 1), (False, 2), (0, 1, 2)])
def test_validated_index_range_rejects_invalid_ranges(value: object) -> None:
    with pytest.raises(ValueError):
        validated_index_range(value, name="selection")


def test_range_and_shape_accept_numpy_integers() -> None:
    assert validated_index_range([np.int64(2), np.int32(5)], name="selection") == (2, 5)
    assert selected_spatial_shape(
        source_line_range=(0, 2),
        shot_in_line_range=(1, 3),
        relative_receiver_x_range=(0, 8),
        relative_receiver_y_range=(18, 50),
    ) == (2, 2, 8, 32)
