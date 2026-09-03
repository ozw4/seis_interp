from __future__ import annotations

import numpy as np
import pandas as pd

from seis_interp.processing.trace_canonicalization import (
    DUPLICATE_PHYSICAL_COORDINATE_POLICY,
    canonicalize_eligible_physical_coordinates,
)
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

_PHYSICAL_KEY = ["source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m"]


def _joined_table(rows: list[tuple[int, int, str, float]]) -> pd.DataFrame:
    """Build a joined table where ``cell_x`` alone distinguishes physical cells."""
    return pd.DataFrame(
        {
            "array_row": np.asarray([row[0] for row in rows], dtype=np.int64),
            "ffid": np.asarray([row[1] for row in rows], dtype=np.int64),
            SPLIT_COLUMN: [row[2] for row in rows],
            "source_x_m": [float(row[3]) for row in rows],
            "source_y_m": [0.0] * len(rows),
            "receiver_x_m": [100.0] * len(rows),
            "receiver_y_m": [200.0] * len(rows),
        }
    )


def test_no_duplicates_keeps_all_rows_and_reports_zero_counts() -> None:
    table = _joined_table(
        [
            (0, 10, TRAIN_SPLIT, 0.0),
            (1, 10, VALIDATION_SPLIT, 1.0),
            (2, 11, TEST_SPLIT, 2.0),
        ]
    )

    canonical, audit = canonicalize_eligible_physical_coordinates(table)

    pd.testing.assert_frame_equal(canonical, table)
    assert audit == {
        "policy": DUPLICATE_PHYSICAL_COORDINATE_POLICY,
        "physical_coordinate_key": _PHYSICAL_KEY,
        "scope": "all_amplitude_eligible_splits_before_ffid_selection",
        "winner_rule": "lowest_array_row",
        "winner_selection_uses_split": False,
        "winner_selection_uses_amplitude": False,
        "input_eligible_trace_count": 3,
        "duplicate_physical_cell_count": 0,
        "duplicate_physical_row_count": 0,
        "removed_trace_count": 0,
        "removed_counts_by_split": {TRAIN_SPLIT: 0, VALIDATION_SPLIT: 0, TEST_SPLIT: 0},
        "removed_counts_by_ffid": {},
        "removed_rows": [],
        "retained_eligible_trace_count": 3,
        "remaining_duplicate_physical_cell_count": 0,
        "remaining_duplicate_physical_row_count": 0,
    }


def test_duplicate_within_same_split_keeps_lowest_array_row() -> None:
    table = _joined_table(
        [
            (5, 10, TRAIN_SPLIT, 0.0),
            (2, 10, TRAIN_SPLIT, 0.0),
            (3, 10, VALIDATION_SPLIT, 1.0),
            (4, 10, TEST_SPLIT, 2.0),
        ]
    )

    canonical, audit = canonicalize_eligible_physical_coordinates(table)

    assert list(canonical["array_row"]) == [2, 3, 4]
    assert audit["removed_trace_count"] == 1
    assert audit["removed_rows"][0]["array_row"] == 5
    assert audit["removed_rows"][0]["kept_array_row"] == 2


def test_duplicate_across_splits_and_ffids_keeps_lowest_array_row() -> None:
    table = _joined_table(
        [
            (7, 10, TRAIN_SPLIT, 0.0),
            (1, 11, VALIDATION_SPLIT, 0.0),
            (2, 12, TEST_SPLIT, 1.0),
        ]
    )

    canonical, audit = canonicalize_eligible_physical_coordinates(table)

    assert list(canonical["array_row"]) == [1, 2]
    assert audit["removed_rows"] == [
        {
            "array_row": 7,
            "ffid": 10,
            "split": TRAIN_SPLIT,
            "kept_array_row": 1,
            "kept_ffid": 11,
            "kept_split": VALIDATION_SPLIT,
        }
    ]


def test_winner_selection_does_not_prefer_train_split() -> None:
    table = _joined_table(
        [
            (3, 10, TRAIN_SPLIT, 0.0),
            (1, 10, TEST_SPLIT, 0.0),
            (2, 10, VALIDATION_SPLIT, 1.0),
        ]
    )

    canonical, audit = canonicalize_eligible_physical_coordinates(table)

    assert list(canonical[SPLIT_COLUMN]) == [TEST_SPLIT, VALIDATION_SPLIT]
    assert audit["winner_selection_uses_split"] is False
    assert audit["winner_selection_uses_amplitude"] is False
    assert audit["removed_rows"][0]["kept_split"] == TEST_SPLIT


def test_excluded_rows_never_win_and_are_never_removed() -> None:
    table = _joined_table(
        [
            (0, 10, EXCLUDED_SPLIT, 0.0),
            (4, 10, TRAIN_SPLIT, 0.0),
            (6, 11, VALIDATION_SPLIT, 0.0),
            (1, 10, TEST_SPLIT, 1.0),
        ]
    )

    canonical, audit = canonicalize_eligible_physical_coordinates(table)

    assert list(canonical["array_row"]) == [0, 4, 1]
    assert list(canonical[SPLIT_COLUMN]) == [EXCLUDED_SPLIT, TRAIN_SPLIT, TEST_SPLIT]
    assert audit["removed_rows"] == [
        {
            "array_row": 6,
            "ffid": 11,
            "split": VALIDATION_SPLIT,
            "kept_array_row": 4,
            "kept_ffid": 10,
            "kept_split": TRAIN_SPLIT,
        }
    ]


def test_canonical_table_preserves_row_order_and_resets_index() -> None:
    table = _joined_table(
        [
            (3, 10, TRAIN_SPLIT, 0.0),
            (1, 11, VALIDATION_SPLIT, 0.0),
            (0, 10, EXCLUDED_SPLIT, 0.0),
            (2, 10, TEST_SPLIT, 1.0),
            (5, 11, TRAIN_SPLIT, 1.0),
            (4, 12, VALIDATION_SPLIT, 2.0),
        ]
    )

    canonical, _audit = canonicalize_eligible_physical_coordinates(table)

    assert list(canonical["array_row"]) == [1, 0, 2, 4]
    assert list(canonical.index) == [0, 1, 2, 3]


def test_removed_rows_are_sorted_by_array_row_with_exact_records() -> None:
    table = _joined_table(
        [
            (3, 10, TRAIN_SPLIT, 0.0),
            (1, 11, VALIDATION_SPLIT, 0.0),
            (0, 10, EXCLUDED_SPLIT, 0.0),
            (2, 10, TEST_SPLIT, 1.0),
            (5, 11, TRAIN_SPLIT, 1.0),
            (4, 12, VALIDATION_SPLIT, 2.0),
        ]
    )

    _canonical, audit = canonicalize_eligible_physical_coordinates(table)

    assert audit["removed_rows"] == [
        {
            "array_row": 3,
            "ffid": 10,
            "split": TRAIN_SPLIT,
            "kept_array_row": 1,
            "kept_ffid": 11,
            "kept_split": VALIDATION_SPLIT,
        },
        {
            "array_row": 5,
            "ffid": 11,
            "split": TRAIN_SPLIT,
            "kept_array_row": 2,
            "kept_ffid": 10,
            "kept_split": TEST_SPLIT,
        },
    ]


def test_removed_counts_by_split_and_ffid() -> None:
    table = _joined_table(
        [
            (3, 10, TRAIN_SPLIT, 0.0),
            (1, 11, VALIDATION_SPLIT, 0.0),
            (0, 10, EXCLUDED_SPLIT, 0.0),
            (2, 10, TEST_SPLIT, 1.0),
            (5, 11, TRAIN_SPLIT, 1.0),
            (4, 12, VALIDATION_SPLIT, 2.0),
        ]
    )

    _canonical, audit = canonicalize_eligible_physical_coordinates(table)

    assert audit["input_eligible_trace_count"] == 5
    assert audit["duplicate_physical_cell_count"] == 2
    assert audit["duplicate_physical_row_count"] == 4
    assert audit["removed_trace_count"] == 2
    assert audit["removed_counts_by_split"] == {
        TRAIN_SPLIT: 2,
        VALIDATION_SPLIT: 0,
        TEST_SPLIT: 0,
    }
    assert audit["removed_counts_by_ffid"] == {"10": 1, "11": 1}
    assert audit["retained_eligible_trace_count"] == 3


def test_remaining_duplicate_counts_are_zero_after_canonicalization() -> None:
    table = _joined_table(
        [
            (3, 10, TRAIN_SPLIT, 0.0),
            (1, 11, VALIDATION_SPLIT, 0.0),
            (2, 10, TEST_SPLIT, 0.0),
        ]
    )

    _canonical, audit = canonicalize_eligible_physical_coordinates(table)

    assert audit["remaining_duplicate_physical_cell_count"] == 0
    assert audit["remaining_duplicate_physical_row_count"] == 0


def test_input_table_is_not_modified() -> None:
    table = _joined_table(
        [
            (3, 10, TRAIN_SPLIT, 0.0),
            (1, 11, VALIDATION_SPLIT, 0.0),
            (0, 10, EXCLUDED_SPLIT, 0.0),
        ]
    )
    original = table.copy(deep=True)

    canonicalize_eligible_physical_coordinates(table)

    pd.testing.assert_frame_equal(table, original)
