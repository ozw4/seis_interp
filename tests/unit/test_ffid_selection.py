from __future__ import annotations

import pandas as pd
import pytest

from seis_interp.processing.ffid_selection import annotate_ffid_quality, select_ffid


def make_trace_table() -> pd.DataFrame:
    """FFID 10 has two traces, FFID 20 has three, FFID 30 has three."""
    return pd.DataFrame(
        {
            "trace_index": [5, 0, 1, 6, 2, 3, 4, 7],
            "ffid": [20, 10, 10, 30, 20, 20, 30, 30],
        }
    )


def test_annotate_counts_traces_per_ffid() -> None:
    annotated = annotate_ffid_quality(make_trace_table(), expected_trace_count=3)

    counts = dict(zip(annotated["ffid"], annotated["trace_count_in_ffid"], strict=True))
    assert counts == {10: 2, 20: 3, 30: 3}


def test_annotate_marks_only_exact_matches_complete() -> None:
    annotated = annotate_ffid_quality(make_trace_table(), expected_trace_count=3)

    complete = dict(zip(annotated["ffid"], annotated["is_complete_ffid"], strict=True))
    assert complete == {10: False, 20: True, 30: True}


def test_annotate_does_not_modify_input() -> None:
    trace_table = make_trace_table()
    original = trace_table.copy()

    annotate_ffid_quality(trace_table, expected_trace_count=3)

    pd.testing.assert_frame_equal(trace_table, original)


def test_annotate_rejects_duplicate_trace_index() -> None:
    trace_table = pd.DataFrame({"trace_index": [0, 0], "ffid": [10, 10]})

    with pytest.raises(ValueError, match="duplicate trace_index"):
        annotate_ffid_quality(trace_table, expected_trace_count=2)


def test_annotate_rejects_empty_table() -> None:
    with pytest.raises(ValueError, match="empty"):
        annotate_ffid_quality(pd.DataFrame({"trace_index": [], "ffid": []}), expected_trace_count=3)


def test_annotate_rejects_non_positive_expected_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        annotate_ffid_quality(make_trace_table(), expected_trace_count=0)


def test_select_explicit_ffid() -> None:
    selection = select_ffid(make_trace_table(), ffid=30, expected_trace_count=3)

    assert selection["ffid"].unique().tolist() == [30]
    assert selection["trace_index"].tolist() == [4, 6, 7]


def test_select_without_ffid_takes_smallest_complete_ffid() -> None:
    selection = select_ffid(make_trace_table(), expected_trace_count=3)

    assert selection["ffid"].unique().tolist() == [20]
    assert selection["trace_index"].tolist() == [2, 3, 5]


def test_selection_is_sorted_regardless_of_input_row_order() -> None:
    shuffled = make_trace_table().iloc[::-1].reset_index(drop=True)

    selection = select_ffid(shuffled, ffid=30, expected_trace_count=3)

    assert selection["trace_index"].tolist() == [4, 6, 7]


def test_selection_carries_quality_columns() -> None:
    selection = select_ffid(make_trace_table(), ffid=20, expected_trace_count=3)

    assert selection["trace_count_in_ffid"].tolist() == [3, 3, 3]
    assert selection["is_complete_ffid"].tolist() == [True, True, True]


def test_select_does_not_modify_input() -> None:
    trace_table = make_trace_table()
    original = trace_table.copy()

    select_ffid(trace_table, ffid=20, expected_trace_count=3)

    pd.testing.assert_frame_equal(trace_table, original)


def test_select_incomplete_ffid_is_an_error() -> None:
    with pytest.raises(ValueError, match="FFID 10 is incomplete"):
        select_ffid(make_trace_table(), ffid=10, expected_trace_count=3)


def test_select_incomplete_ffid_is_allowed_when_not_required() -> None:
    selection = select_ffid(
        make_trace_table(), ffid=10, expected_trace_count=3, require_complete=False
    )

    assert selection["trace_index"].tolist() == [0, 1]
    assert selection["is_complete_ffid"].tolist() == [False, False]


def test_select_missing_ffid_is_an_error() -> None:
    with pytest.raises(ValueError, match="FFID 99 is not present"):
        select_ffid(make_trace_table(), ffid=99, expected_trace_count=3)


def test_select_without_any_complete_ffid_is_an_error() -> None:
    with pytest.raises(ValueError, match="no FFID has the expected trace count"):
        select_ffid(make_trace_table(), expected_trace_count=4)


def test_expected_trace_count_has_no_survey_specific_default() -> None:
    with pytest.raises(TypeError, match="expected_trace_count"):
        select_ffid(make_trace_table())  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="expected_trace_count"):
        annotate_ffid_quality(make_trace_table())  # type: ignore[call-arg]
