from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.trace_table import validated_array_rows


def test_returns_int64_identifiers_in_table_order() -> None:
    trace_table = pd.DataFrame(
        {"array_row": np.asarray([7, 2, 11], dtype=np.uint16)},
        index=[30, 10, 20],
    )

    result = validated_array_rows(trace_table)

    np.testing.assert_array_equal(result, [7, 2, 11])
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "array_rows",
    [
        pd.Series([0, 1, 2], dtype=object),
        pd.Series([0.0, 1.0, 2.0]),
        pd.Series(["0", "1", "2"]),
        pd.Series([False, True, False]),
    ],
)
def test_rejects_non_integer_dtype(array_rows: pd.Series) -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        validated_array_rows(pd.DataFrame({"array_row": array_rows}))


def test_rejects_duplicate_identifiers() -> None:
    with pytest.raises(ValueError, match="duplicate array_row"):
        validated_array_rows(pd.DataFrame({"array_row": [2, 2, 1]}))


def test_contiguous_identifiers_may_be_in_any_table_order() -> None:
    result = validated_array_rows(
        pd.DataFrame({"array_row": [3, 0, 2, 1]}),
        require_contiguous=True,
    )

    np.testing.assert_array_equal(result, [3, 0, 2, 1])


@pytest.mark.parametrize("array_rows", [[0, 1, 3], [-1, 0, 1]])
def test_contiguous_identifiers_must_cover_zero_through_row_count_minus_one(
    array_rows: list[int],
) -> None:
    with pytest.raises(ValueError, match="every integer"):
        validated_array_rows(
            pd.DataFrame({"array_row": array_rows}),
            require_contiguous=True,
        )
