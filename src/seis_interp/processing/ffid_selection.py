"""Annotate FFID completeness on a trace table and select one FFID."""

from __future__ import annotations

import pandas as pd

DEFAULT_EXPECTED_TRACE_COUNT = 544

_REQUIRED_COLUMNS = ("ffid", "trace_index")


def annotate_ffid_quality(
    trace_table: pd.DataFrame,
    expected_trace_count: int = DEFAULT_EXPECTED_TRACE_COUNT,
) -> pd.DataFrame:
    """Add per-FFID trace counts and a completeness flag to a trace table.

    Adds ``trace_count_in_ffid`` and ``is_complete_ffid``. An FFID is complete
    only when its trace count equals ``expected_trace_count``. The input frame
    is not modified; a copy is returned.
    """
    if expected_trace_count < 1:
        raise ValueError(f"expected_trace_count must be at least 1, got {expected_trace_count}")

    missing = [column for column in _REQUIRED_COLUMNS if column not in trace_table.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")
    if trace_table.empty:
        raise ValueError("trace table is empty")
    if trace_table["trace_index"].duplicated().any():
        raise ValueError("trace table contains duplicate trace_index values")

    annotated = trace_table.copy()
    annotated["trace_count_in_ffid"] = (
        annotated.groupby("ffid")["trace_index"].transform("size").astype("int64")
    )
    annotated["is_complete_ffid"] = annotated["trace_count_in_ffid"] == expected_trace_count
    return annotated


def select_ffid(
    trace_table: pd.DataFrame,
    ffid: int | None = None,
    expected_trace_count: int = DEFAULT_EXPECTED_TRACE_COUNT,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Select the traces of one FFID, sorted by ``trace_index``.

    With ``ffid=None`` the numerically smallest complete FFID is selected. With
    an explicit ``ffid`` and ``require_complete=True`` an incomplete FFID is an
    error. The input frame is not modified.
    """
    annotated = annotate_ffid_quality(trace_table, expected_trace_count=expected_trace_count)

    if ffid is None:
        complete_ffids = annotated.loc[annotated["is_complete_ffid"], "ffid"]
        if complete_ffids.empty:
            raise ValueError(
                f"no FFID has the expected trace count of {expected_trace_count} traces"
            )
        selected_ffid = int(complete_ffids.min())
    else:
        selected_ffid = int(ffid)
        rows = annotated.loc[annotated["ffid"] == selected_ffid]
        if rows.empty:
            raise ValueError(f"FFID {selected_ffid} is not present in the trace table")
        trace_count = int(rows["trace_count_in_ffid"].iloc[0])
        if require_complete and trace_count != expected_trace_count:
            raise ValueError(
                f"FFID {selected_ffid} is incomplete: {trace_count} traces, "
                f"expected {expected_trace_count}"
            )

    selection = annotated.loc[annotated["ffid"] == selected_ffid]
    return selection.sort_values("trace_index").reset_index(drop=True)
