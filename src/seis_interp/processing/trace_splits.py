"""Assign deterministic train, validation, and test splits to complete traces."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

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

    split_counts = _split_counts(
        len(array_rows),
        holdout_fraction=holdout,
        validation_fraction_of_holdout=validation_of_holdout,
    )
    permutation = np.random.default_rng(seed).permutation(np.sort(array_rows))
    assignments = _split_assignments(
        permutation,
        validation_count=split_counts[VALIDATION_SPLIT],
        test_count=split_counts[TEST_SPLIT],
    )

    result = trace_table.copy()
    result[SPLIT_COLUMN] = [assignments[int(array_row)] for array_row in array_rows]
    return result


def assign_random_trace_splits_by_ffid(
    trace_table: pd.DataFrame,
    *,
    holdout_fraction: float,
    validation_fraction_of_holdout: float,
    random_seed: int,
) -> pd.DataFrame:
    """Assign deterministic whole-trace splits independently within every FFID.

    Each FFID has a generator derived from ``(random_seed, ffid)``. Adding or
    removing a different FFID therefore does not change existing membership.
    ``array_row`` values are sorted before permutation, so the result is also
    independent of DataFrame row order and index values.
    """
    array_rows = validated_array_rows(trace_table)
    ffids = _validated_ffids(trace_table)
    holdout = _validated_fraction("holdout_fraction", holdout_fraction)
    validation_of_holdout = _validated_fraction(
        "validation_fraction_of_holdout", validation_fraction_of_holdout
    )
    seed = _validated_random_seed(random_seed)

    assignments: dict[int, str] = {}
    for ffid in np.unique(ffids):
        ffid_value = int(ffid)
        ffid_rows = np.sort(array_rows[ffids == ffid])
        split_counts = _split_counts(
            len(ffid_rows),
            holdout_fraction=holdout,
            validation_fraction_of_holdout=validation_of_holdout,
            context=f"FFID {ffid_value}",
        )
        ffid_entropy = ffid_value if ffid_value >= 0 else (1 << 64) + ffid_value
        ffid_seed = np.random.SeedSequence([seed, ffid_entropy])
        permutation = np.random.default_rng(ffid_seed).permutation(ffid_rows)
        assignments.update(
            _split_assignments(
                permutation,
                validation_count=split_counts[VALIDATION_SPLIT],
                test_count=split_counts[TEST_SPLIT],
            )
        )

    result = trace_table.copy()
    result[SPLIT_COLUMN] = [assignments[int(array_row)] for array_row in array_rows]
    return result


def _split_counts(
    trace_count: int,
    *,
    holdout_fraction: float,
    validation_fraction_of_holdout: float,
    context: str | None = None,
) -> dict[str, int]:
    """Return split counts using the established round-based allocation rule."""
    holdout_count = int(round(trace_count * holdout_fraction))
    validation_count = int(round(holdout_count * validation_fraction_of_holdout))
    test_count = holdout_count - validation_count
    split_counts = {
        TRAIN_SPLIT: trace_count - holdout_count,
        VALIDATION_SPLIT: validation_count,
        TEST_SPLIT: test_count,
    }
    empty_splits = [name for name, count in split_counts.items() if count == 0]
    if empty_splits:
        context_text = f" for {context}" if context is not None else ""
        raise ValueError(
            f"split fractions produce an empty split{context_text}: "
            f"{', '.join(empty_splits)}; counts={split_counts}"
        )
    return split_counts


def _split_assignments(
    permutation: np.ndarray,
    *,
    validation_count: int,
    test_count: int,
) -> dict[int, str]:
    """Map one canonical row permutation onto validation, test, then train."""
    test_stop = validation_count + test_count
    assignments = {int(array_row): VALIDATION_SPLIT for array_row in permutation[:validation_count]}
    assignments.update(
        {int(array_row): TEST_SPLIT for array_row in permutation[validation_count:test_stop]}
    )
    assignments.update({int(array_row): TRAIN_SPLIT for array_row in permutation[test_stop:]})
    return assignments


def _validated_ffids(trace_table: pd.DataFrame) -> np.ndarray:
    """Return finite integer FFIDs as signed 64-bit values."""
    if "ffid" not in trace_table.columns:
        raise ValueError("trace table is missing required column: ffid")
    values = trace_table["ffid"]
    if values.isna().any():
        raise ValueError("ffid contains missing values")
    if is_bool_dtype(values.dtype) or not is_integer_dtype(values.dtype):
        raise ValueError(f"ffid must have an integer dtype, got {values.dtype}")

    int64_info = np.iinfo(np.int64)
    if int(values.min()) < int64_info.min or int(values.max()) > int64_info.max:
        raise ValueError("ffid values must fit in int64")
    return values.to_numpy(dtype=np.int64)


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
