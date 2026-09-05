"""Assign deterministic train, validation, and test splits to complete traces."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.c3_volume_index import c3_source_indices, validated_index_range

SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
EXCLUDED_SPLIT = "excluded"

GLOBAL_SPLIT_SCOPE = "global"
PER_FFID_SPLIT_SCOPE = "per_ffid"
WHOLE_FFID_SPLIT_SCOPE = "whole_ffid"
C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE = "c3_source_line_blocks"

SPLIT_SCOPES = frozenset(
    (
        GLOBAL_SPLIT_SCOPE,
        PER_FFID_SPLIT_SCOPE,
        WHOLE_FFID_SPLIT_SCOPE,
        C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
    )
)

WHOLE_FFID_ASSIGNMENT_SPLIT_SCOPES = frozenset(
    (WHOLE_FFID_SPLIT_SCOPE, C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE)
)

_EFFECTIVE_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


def validate_prepared_split_assignments(
    joined_table: pd.DataFrame,
    preparation: Mapping[str, object],
) -> None:
    """Validate recorded split counts and whole-FFID membership when applicable."""
    actual_split_counts = {
        split: int(joined_table[SPLIT_COLUMN].eq(split).sum()) for split in _EFFECTIVE_SPLITS
    }
    recorded_split_counts = preparation.get("split_counts")
    if recorded_split_counts != actual_split_counts:
        raise ValueError(
            "preparation.json split_counts do not match trace_split.parquet: "
            f"recorded={recorded_split_counts!r}, actual={actual_split_counts!r}"
        )

    split_scope = preparation.get("split_scope")
    if split_scope not in WHOLE_FFID_ASSIGNMENT_SPLIT_SCOPES:
        return

    _validated_ffids(joined_table)
    eligible = joined_table.loc[joined_table[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)]
    actual_ffid_split_counts = {
        split: int(eligible.loc[eligible[SPLIT_COLUMN].eq(split), "ffid"].nunique())
        for split in _EFFECTIVE_SPLITS
    }
    recorded_ffid_split_counts = preparation.get("ffid_split_counts")
    if recorded_ffid_split_counts != actual_ffid_split_counts:
        raise ValueError(
            "preparation.json ffid_split_counts do not match trace_split.parquet: "
            f"recorded={recorded_ffid_split_counts!r}, actual={actual_ffid_split_counts!r}"
        )

    splits_per_ffid = eligible.groupby("ffid")[SPLIT_COLUMN].nunique()
    crossing_ffids = splits_per_ffid.index[splits_per_ffid.gt(1)].tolist()
    if crossing_ffids:
        raise ValueError(
            "trace_split.parquet does not contain disjoint whole-FFID splits; "
            "FFIDs in multiple non-excluded splits: "
            f"{crossing_ffids}"
        )

    if split_scope == C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE:
        _validate_recorded_c3_source_line_blocks(joined_table, preparation)


def validated_c3_source_line_ranges(value: object) -> dict[str, tuple[int, int]]:
    """Return exact, contiguous train/validation/test source-line ranges."""
    if not isinstance(value, Mapping) or set(value) != set(_EFFECTIVE_SPLITS):
        raise ValueError(f"source_line_ranges must contain exactly {list(_EFFECTIVE_SPLITS)}")

    ranges = {
        split: validated_index_range(
            value[split],
            name=f"source_line_ranges.{split}",
        )
        for split in _EFFECTIVE_SPLITS
    }
    ordered = sorted((start, stop, split) for split, (start, stop) in ranges.items())
    if ordered[0][0] != 0:
        raise ValueError("source_line_ranges must start at source line 0")
    for (_, previous_stop, _), (next_start, _, _) in zip(ordered, ordered[1:], strict=False):
        if next_start < previous_stop:
            raise ValueError("source_line_ranges must not overlap")
        if next_start > previous_stop:
            raise ValueError("source_line_ranges must not contain gaps")
    return ranges


def assign_c3_source_line_block_splits(
    trace_table: pd.DataFrame,
    *,
    source_line_ranges: Mapping[str, object],
) -> pd.DataFrame:
    """Assign contiguous C3 source-line blocks without splitting an FFID."""
    array_rows = validated_array_rows(trace_table)
    _validated_ffids(trace_table)
    ranges = validated_c3_source_line_ranges(source_line_ranges)
    source_line_indices, _ = c3_source_indices(trace_table)
    source_line_count = int(source_line_indices.max()) + 1
    final_stop = max(stop for _, stop in ranges.values())
    if final_stop != source_line_count:
        raise ValueError(
            "source_line_ranges must cover every source line exactly once: "
            f"configured stop={final_stop}, source_line_count={source_line_count}"
        )

    result = trace_table.copy()
    result[SPLIT_COLUMN] = _c3_source_line_split_labels(source_line_indices, ranges)
    if not np.array_equal(result["array_row"].to_numpy(dtype=np.int64), array_rows):
        raise AssertionError("C3 source-line block splitting changed array-row order")

    splits_per_ffid = result.groupby("ffid", sort=False)[SPLIT_COLUMN].nunique()
    crossing_ffids = splits_per_ffid.index[splits_per_ffid.gt(1)].tolist()
    if crossing_ffids:
        raise ValueError(
            "C3 source-line blocks do not assign each FFID wholly to one split; "
            f"crossing FFIDs: {crossing_ffids}"
        )
    return result


def _c3_source_line_split_labels(
    source_line_indices: np.ndarray,
    ranges: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    assignments = np.empty(len(source_line_indices), dtype=object)
    for split in _EFFECTIVE_SPLITS:
        start, stop = ranges[split]
        selected = (source_line_indices >= start) & (source_line_indices < stop)
        assignments[selected] = split
    return assignments


def _validate_recorded_c3_source_line_blocks(
    joined_table: pd.DataFrame,
    preparation: Mapping[str, object],
) -> None:
    ranges = preparation.get("source_line_ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("preparation.json source_line_ranges must be an object")
    expected = assign_c3_source_line_block_splits(
        joined_table.drop(columns=[SPLIT_COLUMN]),
        source_line_ranges=ranges,
    )[SPLIT_COLUMN].to_numpy()
    actual = joined_table[SPLIT_COLUMN].to_numpy()
    eligible = actual != EXCLUDED_SPLIT
    if not np.array_equal(actual[eligible], expected[eligible]):
        raise ValueError("trace_split.parquet does not match preparation.json source_line_ranges")


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
    seed = validated_random_seed(random_seed)

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
    seed = validated_random_seed(random_seed)

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


def assign_random_whole_ffid_splits(
    trace_table: pd.DataFrame,
    *,
    holdout_fraction: float,
    validation_fraction_of_holdout: float,
    random_seed: int,
) -> pd.DataFrame:
    """Assign each eligible FFID wholly to train, validation, or test.

    Split counts are computed from the number of distinct FFIDs. FFIDs are
    sorted before permutation, so membership is independent of trace row order
    and DataFrame index values. Every row belonging to one FFID receives the
    same split label.
    """
    array_rows = validated_array_rows(trace_table)
    ffids = _validated_ffids(trace_table)
    holdout = _validated_fraction("holdout_fraction", holdout_fraction)
    validation_of_holdout = _validated_fraction(
        "validation_fraction_of_holdout", validation_fraction_of_holdout
    )
    seed = validated_random_seed(random_seed)

    unique_ffids = np.unique(ffids)
    split_counts = _split_counts(
        len(unique_ffids),
        holdout_fraction=holdout,
        validation_fraction_of_holdout=validation_of_holdout,
        context="eligible FFIDs",
    )
    permutation = np.random.default_rng(seed).permutation(unique_ffids)
    assignments = _split_assignments(
        permutation,
        validation_count=split_counts[VALIDATION_SPLIT],
        test_count=split_counts[TEST_SPLIT],
    )

    result = trace_table.copy()
    result[SPLIT_COLUMN] = [assignments[int(ffid)] for ffid in ffids]
    if not np.array_equal(result["array_row"].to_numpy(dtype=np.int64), array_rows):
        raise AssertionError("whole-FFID splitting changed array-row order")
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


def validated_random_seed(value: int) -> int:
    """Return an integer seed accepted by NumPy's default generator."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"random_seed must be an integer, got {value!r}")
    seed = int(value)
    if seed < 0:
        raise ValueError(f"random_seed must be non-negative, got {seed}")
    return seed
