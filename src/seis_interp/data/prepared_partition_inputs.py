"""Validate prepared partition rows against the source trace identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seis_interp.data.prepared_partition import TRACE_SPLIT_FILE_NAME
from seis_interp.data.trace_store import TRACES_FILE_NAME
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


def validated_prepared_split_rows(
    split_table: pd.DataFrame,
    *,
    expected_array_rows: np.ndarray,
) -> np.ndarray:
    """Require complete row alignment and known prepared split labels."""
    stored_splits = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, EXCLUDED_SPLIT)
    if SPLIT_COLUMN not in split_table.columns:
        raise ValueError(f"{TRACE_SPLIT_FILE_NAME} is missing required column: {SPLIT_COLUMN}")
    split_rows = validated_array_rows(split_table)
    if not np.array_equal(np.sort(split_rows), np.sort(expected_array_rows)):
        raise ValueError(
            f"{TRACE_SPLIT_FILE_NAME} array_row values do not exactly match {TRACES_FILE_NAME}"
        )

    known = split_table[SPLIT_COLUMN].isin(stored_splits)
    if not bool(known.all()):
        unknown = sorted({repr(value) for value in split_table.loc[~known, SPLIT_COLUMN].tolist()})
        raise ValueError(f"{TRACE_SPLIT_FILE_NAME} contains unknown split values: {unknown}")
    return split_rows
