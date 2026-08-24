"""Assign deterministic train, validation, and test splits to complete traces."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from numbers import Integral, Real

import numpy as np
import pandas as pd

SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"


def assign_random_trace_splits(
    trace_table: pd.DataFrame,
    *,
    holdout_fraction: float,
    validation_fraction_of_holdout: float,
    random_seed: int,
) -> pd.DataFrame:
    """Return a copy with deterministic whole-trace split assignments.

    Split membership is assigned to ``array_row`` identifiers, not to the
    DataFrame index. Identifiers are sorted before permutation so membership
    remains stable when the same table is stored in a different row order.
    """
    array_rows = _validated_array_rows(trace_table)
    holdout = _validated_fraction("holdout_fraction", holdout_fraction)
    validation_of_holdout = _validated_fraction(
        "validation_fraction_of_holdout", validation_fraction_of_holdout
    )
    seed = _validated_random_seed(random_seed)

    trace_count = len(array_rows)
    holdout_count = int(round(trace_count * holdout))
    validation_count = int(round(holdout_count * validation_of_holdout))
    test_count = holdout_count - validation_count
    train_count = trace_count - holdout_count
    split_counts = {
        TRAIN_SPLIT: train_count,
        VALIDATION_SPLIT: validation_count,
        TEST_SPLIT: test_count,
    }
    empty_splits = [name for name, count in split_counts.items() if count == 0]
    if empty_splits:
        raise ValueError(
            "split fractions produce an empty split: "
            f"{', '.join(empty_splits)}; counts={split_counts}"
        )

    permutation = np.random.default_rng(seed).permutation(np.sort(array_rows))
    assignments = {int(array_row): VALIDATION_SPLIT for array_row in permutation[:validation_count]}
    assignments.update(
        {
            int(array_row): TEST_SPLIT
            for array_row in permutation[validation_count : validation_count + test_count]
        }
    )
    assignments.update(
        {int(array_row): TRAIN_SPLIT for array_row in permutation[validation_count + test_count :]}
    )

    result = trace_table.copy()
    result[SPLIT_COLUMN] = [assignments[int(array_row)] for array_row in array_rows]
    return result


def _validated_array_rows(trace_table: pd.DataFrame) -> np.ndarray:
    """Return unique ``array_row`` values represented as signed integers."""
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    if trace_table.empty:
        raise ValueError("trace table is empty")
    if "array_row" not in trace_table.columns:
        raise ValueError("trace table is missing required column: array_row")

    values = trace_table["array_row"]
    if values.isna().any():
        raise ValueError("array_row must contain integers")
    try:
        converted = [_integer_array_row(value) for value in values]
    except (TypeError, ValueError, OverflowError, InvalidOperation) as error:
        raise ValueError("array_row must contain integers") from error
    array_rows = np.asarray(converted, dtype=np.int64)
    if len(np.unique(array_rows)) != len(array_rows):
        raise ValueError("trace table contains duplicate array_row values")
    return array_rows


def _integer_array_row(value: object) -> int:
    """Convert one integer-like identifier without losing large-value precision."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean array_row is not an integer identifier")
    if isinstance(value, Integral):
        converted = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("array_row must be a finite integer value")
        converted = int(numeric)
    elif isinstance(value, (str, Decimal)):
        numeric_decimal = Decimal(value)
        if (
            not numeric_decimal.is_finite()
            or numeric_decimal != numeric_decimal.to_integral_value()
        ):
            raise ValueError("array_row must be a finite integer value")
        converted = int(numeric_decimal)
    else:
        raise TypeError(f"unsupported array_row type: {type(value).__name__}")

    if not -(2**63) <= converted < 2**63:
        raise ValueError("array_row values must fit in int64")
    return converted


def _validated_fraction(name: str, value: float) -> float:
    """Return a finite real fraction strictly between zero and one."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number strictly between 0 and 1")
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return fraction


def _validated_random_seed(value: int) -> int:
    """Return an integer seed accepted by NumPy's default generator."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"random_seed must be an integer, got {value!r}")
    seed = int(value)
    if seed < 0:
        raise ValueError(f"random_seed must be non-negative, got {seed}")
    return seed
