"""Validate identifiers that link trace tables to row-oriented arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype


def validated_array_rows(
    trace_table: pd.DataFrame,
    *,
    require_contiguous: bool = False,
) -> np.ndarray:
    """Return unique ``array_row`` identifiers as signed 64-bit integers."""
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    if trace_table.empty:
        raise ValueError("trace table is empty")
    if "array_row" not in trace_table.columns:
        raise ValueError("trace table is missing required column: array_row")

    values = trace_table["array_row"]
    if values.isna().any():
        raise ValueError("array_row contains missing values")
    if is_bool_dtype(values.dtype) or not is_integer_dtype(values.dtype):
        raise ValueError(f"array_row must have an integer dtype, got {values.dtype}")
    if values.duplicated().any():
        raise ValueError("trace table contains duplicate array_row values")

    int64_info = np.iinfo(np.int64)
    if int(values.min()) < int64_info.min or int(values.max()) > int64_info.max:
        raise ValueError("array_row values must fit in int64")
    array_rows = values.to_numpy(dtype=np.int64)

    if require_contiguous and not np.array_equal(
        np.sort(array_rows), np.arange(len(trace_table), dtype=np.int64)
    ):
        raise ValueError(
            "array_row must contain every integer from 0 through "
            f"{len(trace_table) - 1} exactly once"
        )
    return array_rows
