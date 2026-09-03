from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.trace_selection import (
    build_trace_selection_contract,
    join_trace_splits,
    select_eligible_traces,
    validate_selected_split_coverage,
)
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


def _trace_table(array_rows: list[int], ffids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": np.asarray(array_rows, dtype=np.int64),
            "ffid": np.asarray(ffids, dtype=np.int64),
        }
    )


def _joined_table(ffids: list[int], splits: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": np.arange(len(ffids), dtype=np.int64),
            "ffid": np.asarray(ffids, dtype=np.int64),
            SPLIT_COLUMN: splits,
        }
    )


def test_join_trace_splits_places_split_values_by_array_row() -> None:
    trace_table = _trace_table([0, 1, 2, 3], [10, 10, 11, 11])
    split_table = pd.DataFrame({SPLIT_COLUMN: [TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT]})
    split_rows = np.asarray([2, 0, 3], dtype=np.int64)

    joined = join_trace_splits(trace_table, split_table, split_rows)

    assert list(joined[SPLIT_COLUMN][[0, 2, 3]]) == [TRAIN_SPLIT, TEST_SPLIT, VALIDATION_SPLIT]


def test_join_trace_splits_preserves_row_order_and_does_not_modify_inputs() -> None:
    trace_table = _trace_table([2, 0, 1], [11, 10, 10])
    original_trace = trace_table.copy()
    split_table = pd.DataFrame({SPLIT_COLUMN: [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]})
    original_split = split_table.copy()
    split_rows = np.asarray([0, 1, 2], dtype=np.int64)

    joined = join_trace_splits(trace_table, split_table, split_rows)

    assert list(joined.columns) == ["array_row", "ffid", SPLIT_COLUMN]
    np.testing.assert_array_equal(
        joined["array_row"].to_numpy(), original_trace["array_row"].to_numpy()
    )
    np.testing.assert_array_equal(joined["ffid"].to_numpy(), original_trace["ffid"].to_numpy())
    assert list(joined[SPLIT_COLUMN]) == [TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT]
    pd.testing.assert_frame_equal(trace_table, original_trace)
    pd.testing.assert_frame_equal(split_table, original_split)


def test_select_eligible_traces_drops_excluded_and_resets_index() -> None:
    table = _joined_table(
        [10, 10, 11, 11],
        [TRAIN_SPLIT, EXCLUDED_SPLIT, VALIDATION_SPLIT, TEST_SPLIT],
    )

    selected = select_eligible_traces(table, ffid_range=None)

    assert list(selected.index) == [0, 1, 2]
    assert list(selected["array_row"]) == [0, 2, 3]
    assert EXCLUDED_SPLIT not in set(selected[SPLIT_COLUMN])


def test_select_eligible_traces_ffid_range_is_inclusive_and_preserves_order() -> None:
    table = _joined_table(
        [9, 10, 11, 12, 13],
        [TRAIN_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, TRAIN_SPLIT],
    )

    selected = select_eligible_traces(table, ffid_range=(10, 12))

    assert list(selected["ffid"]) == [10, 11, 12]
    assert list(selected.index) == [0, 1, 2]


def test_select_eligible_traces_raises_on_empty_selection() -> None:
    table = _joined_table([10, 11], [TRAIN_SPLIT, EXCLUDED_SPLIT])

    with pytest.raises(ValueError, match="configured FFID selection contains no eligible traces"):
        select_eligible_traces(table, ffid_range=(11, 11))


def test_validate_selected_split_coverage_accepts_complete_per_ffid_splits() -> None:
    table = _joined_table(
        [10, 10, 10, 11, 11, 11],
        [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT] * 2,
    )

    validate_selected_split_coverage(table, split_scope="per_ffid")


def test_validate_selected_split_coverage_rejects_incomplete_per_ffid_splits() -> None:
    table = _joined_table(
        [10, 10, 10, 11, 11],
        [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT],
    )

    with pytest.raises(
        ValueError, match=r"selected eligible FFIDs do not contain every split: \[11\]"
    ):
        validate_selected_split_coverage(table, split_scope="per_ffid")


def test_validate_selected_split_coverage_accepts_single_split_whole_ffids() -> None:
    table = _joined_table(
        [10, 10, 11, 12],
        [TRAIN_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT],
    )

    validate_selected_split_coverage(table, split_scope="whole_ffid")


def test_validate_selected_split_coverage_rejects_mixed_whole_ffids() -> None:
    table = _joined_table(
        [10, 10, 11, 12],
        [TRAIN_SPLIT, VALIDATION_SPLIT, VALIDATION_SPLIT, TEST_SPLIT],
    )

    with pytest.raises(
        ValueError, match=r"whole-FFID split assigns FFIDs to multiple splits: \[10\]"
    ):
        validate_selected_split_coverage(table, split_scope="whole_ffid")


def test_validate_selected_split_coverage_rejects_missing_global_split() -> None:
    table = _joined_table([10, 10, 11], [TRAIN_SPLIT, VALIDATION_SPLIT, TRAIN_SPLIT])

    with pytest.raises(
        ValueError, match=r"selected eligible traces contain no rows for: \['test'\]"
    ):
        validate_selected_split_coverage(table, split_scope="per_ffid")


def test_validate_selected_split_coverage_rejects_unsupported_scope() -> None:
    table = _joined_table(
        [10, 10, 10],
        [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT],
    )

    with pytest.raises(ValueError, match="does not support split_scope 'global'"):
        validate_selected_split_coverage(table, split_scope="global")


def test_build_trace_selection_contract_per_ffid_without_configured_range() -> None:
    canonical = _joined_table(
        [10, 10, 10, 11, 11, 11, 12],
        [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT] * 2 + [EXCLUDED_SPLIT],
    )
    selected = select_eligible_traces(canonical, ffid_range=None)

    contract = build_trace_selection_contract(
        canonical,
        selected,
        sample_count=64,
        configured_ffid_range=None,
    )

    assert contract == {
        "configured_ffid_range": None,
        "selected_ffid_count": 2,
        "selected_ffid_range": [10, 11],
        "selected_ffids": [10, 11],
        "ffids_by_split": {
            TRAIN_SPLIT: [10, 11],
            VALIDATION_SPLIT: [10, 11],
            TEST_SPLIT: [10, 11],
        },
        "ffid_split_counts": {TRAIN_SPLIT: 2, VALIDATION_SPLIT: 2, TEST_SPLIT: 2},
        "ffid_split_overlap_count": 2,
        "maximum_splits_per_ffid": 3,
        "sample_count": 64,
        "effective_eligible_trace_count": 6,
        "split_counts": {
            TRAIN_SPLIT: 2,
            VALIDATION_SPLIT: 2,
            TEST_SPLIT: 2,
            EXCLUDED_SPLIT: 1,
        },
    }


def test_build_trace_selection_contract_whole_ffid_with_configured_range() -> None:
    canonical = _joined_table(
        [9, 10, 10, 11, 12, 12, 13],
        [
            EXCLUDED_SPLIT,
            TRAIN_SPLIT,
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
            TEST_SPLIT,
            EXCLUDED_SPLIT,
            EXCLUDED_SPLIT,
        ],
    )
    selected = select_eligible_traces(canonical, ffid_range=(10, 12))

    contract = build_trace_selection_contract(
        canonical,
        selected,
        sample_count=32,
        configured_ffid_range=(10, 12),
    )

    assert contract == {
        "configured_ffid_range": [10, 12],
        "selected_ffid_count": 3,
        "selected_ffid_range": [10, 12],
        "selected_ffids": [10, 11, 12],
        "ffids_by_split": {
            TRAIN_SPLIT: [10],
            VALIDATION_SPLIT: [11],
            TEST_SPLIT: [12],
        },
        "ffid_split_counts": {TRAIN_SPLIT: 1, VALIDATION_SPLIT: 1, TEST_SPLIT: 1},
        "ffid_split_overlap_count": 0,
        "maximum_splits_per_ffid": 1,
        "sample_count": 32,
        "effective_eligible_trace_count": 4,
        "split_counts": {
            TRAIN_SPLIT: 2,
            VALIDATION_SPLIT: 1,
            TEST_SPLIT: 1,
            EXCLUDED_SPLIT: 1,
        },
    }


def test_build_trace_selection_contract_counts_only_excluded_rows_in_range() -> None:
    canonical = _joined_table(
        [9, 10, 11, 12, 13],
        [EXCLUDED_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, EXCLUDED_SPLIT],
    )
    selected = select_eligible_traces(canonical, ffid_range=(10, 13))

    contract = build_trace_selection_contract(
        canonical,
        selected,
        sample_count=16,
        configured_ffid_range=(10, 13),
    )

    assert contract["split_counts"][EXCLUDED_SPLIT] == 1
