"""Assign deterministic train, validation, and test splits to complete traces."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

from seis_interp.data.trace_table import validated_array_rows

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
    array_rows = validated_array_rows(trace_table)
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
