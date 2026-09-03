"""Join split labels onto trace tables and select eligible traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

_EFFECTIVE_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


def join_trace_splits(
    trace_table: pd.DataFrame,
    split_table: pd.DataFrame,
    split_rows: np.ndarray,
) -> pd.DataFrame:
    """Return a trace-table copy with split labels placed by ``array_row``."""
    split_by_array_row = np.empty(len(trace_table), dtype=object)
    split_by_array_row[split_rows] = split_table[SPLIT_COLUMN].to_numpy()
    trace_rows = trace_table["array_row"].to_numpy(dtype=np.int64)
    joined = trace_table.copy()
    joined[SPLIT_COLUMN] = split_by_array_row[trace_rows]
    return joined


def select_eligible_traces(
    canonical_table: pd.DataFrame,
    *,
    ffid_range: tuple[int, int] | None,
) -> pd.DataFrame:
    """Return non-excluded rows within the inclusive FFID range, reindexed."""
    selected = canonical_table[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)
    if ffid_range is not None:
        selected &= canonical_table["ffid"].between(*ffid_range)
    result = canonical_table.loc[selected].reset_index(drop=True)
    if result.empty:
        raise ValueError("configured FFID selection contains no eligible traces")
    return result


def validate_selected_split_coverage(table: pd.DataFrame, *, split_scope: str) -> None:
    """Require the selected traces to satisfy the configured split scope."""
    present_splits = set(str(value) for value in table[SPLIT_COLUMN].unique())
    missing_splits = set(_EFFECTIVE_SPLITS) - present_splits
    if missing_splits:
        raise ValueError(f"selected eligible traces contain no rows for: {sorted(missing_splits)}")

    split_counts_by_ffid = table.groupby("ffid")[SPLIT_COLUMN].nunique()
    if split_scope == "per_ffid":
        incomplete = sorted(
            int(ffid)
            for ffid, count in split_counts_by_ffid.items()
            if count != len(_EFFECTIVE_SPLITS)
        )
        if incomplete:
            raise ValueError(f"selected eligible FFIDs do not contain every split: {incomplete}")
        return
    if split_scope == "whole_ffid":
        mixed = sorted(int(ffid) for ffid, count in split_counts_by_ffid.items() if count != 1)
        if mixed:
            raise ValueError(f"whole-FFID split assigns FFIDs to multiple splits: {mixed}")
        return
    raise ValueError(f"neighbor inpainter does not support split_scope {split_scope!r}")


def build_trace_selection_contract(
    canonical_table: pd.DataFrame,
    selected_table: pd.DataFrame,
    *,
    sample_count: int,
    configured_ffid_range: tuple[int, int] | None,
) -> dict[str, object]:
    """Summarize the selected traces, FFIDs, and splits for run provenance."""
    split_counts = {
        split: int(selected_table[SPLIT_COLUMN].eq(split).sum())
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }
    in_range = np.ones(len(canonical_table), dtype=bool)
    if configured_ffid_range is not None:
        in_range = canonical_table["ffid"].between(*configured_ffid_range).to_numpy()
    full_split = canonical_table[SPLIT_COLUMN].to_numpy()
    excluded_count = int(np.count_nonzero(in_range & (full_split == EXCLUDED_SPLIT)))
    ffids = sorted(int(value) for value in selected_table["ffid"].unique())
    ffids_by_split = {
        split: sorted(
            int(value)
            for value in selected_table.loc[selected_table[SPLIT_COLUMN].eq(split), "ffid"].unique()
        )
        for split in _EFFECTIVE_SPLITS
    }
    split_memberships_per_ffid = selected_table.groupby("ffid")[SPLIT_COLUMN].nunique()
    contract: dict[str, object] = {
        "configured_ffid_range": (
            list(configured_ffid_range) if configured_ffid_range is not None else None
        ),
        "selected_ffid_count": len(ffids),
        "selected_ffid_range": [ffids[0], ffids[-1]],
        "selected_ffids": ffids,
        "ffids_by_split": ffids_by_split,
        "ffid_split_counts": {
            split: len(split_ffids) for split, split_ffids in ffids_by_split.items()
        },
        "ffid_split_overlap_count": int(split_memberships_per_ffid.gt(1).sum()),
        "maximum_splits_per_ffid": int(split_memberships_per_ffid.max()),
        "sample_count": sample_count,
        "effective_eligible_trace_count": sum(split_counts.values()),
        "split_counts": {**split_counts, EXCLUDED_SPLIT: excluded_count},
    }
    return contract
