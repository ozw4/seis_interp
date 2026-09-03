"""Canonicalize duplicate physical coordinate cells deterministically."""

from __future__ import annotations

import pandas as pd

from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

DUPLICATE_PHYSICAL_COORDINATE_POLICY = "keep_lowest_array_row"

_PHYSICAL_COORDINATE_COLUMNS = (
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
)
_EFFECTIVE_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


def canonicalize_eligible_physical_coordinates(
    joined_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep the lowest array row per eligible physical cell, before selection.

    Winner selection deliberately uses only the exact physical key and
    ``array_row``. Split labels are retained solely for the removal audit and
    never influence which row wins.
    """
    keys = list(_PHYSICAL_COORDINATE_COLUMNS)
    eligible = joined_table.loc[joined_table[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)].copy()
    group_sizes = eligible.groupby(keys, sort=False, dropna=False)["array_row"].transform("size")
    duplicate_rows = eligible.loc[group_sizes.gt(1)]
    winners = (
        eligible.sort_values("array_row", kind="stable")
        .drop_duplicates(keys, keep="first")
        .sort_index()
    )
    removed = eligible.loc[~eligible.index.isin(winners.index)].copy()
    winner_lookup = winners[keys + ["array_row", "ffid", SPLIT_COLUMN]].rename(
        columns={
            "array_row": "kept_array_row",
            "ffid": "kept_ffid",
            SPLIT_COLUMN: "kept_split",
        }
    )
    removed_details = removed.merge(
        winner_lookup,
        how="left",
        on=keys,
        sort=False,
        validate="many_to_one",
    ).sort_values("array_row")
    canonical = joined_table.drop(index=removed.index).reset_index(drop=True)
    canonical_eligible = canonical.loc[canonical[SPLIT_COLUMN].ne(EXCLUDED_SPLIT)]
    remaining_duplicate_mask = canonical_eligible.duplicated(keys, keep=False)
    remaining_duplicate_rows = canonical_eligible.loc[remaining_duplicate_mask]

    removed_counts_by_split = {
        split: int(removed[SPLIT_COLUMN].eq(split).sum()) for split in _EFFECTIVE_SPLITS
    }
    removed_ffid_counts = removed["ffid"].value_counts().sort_index()
    removed_records = [
        {
            "array_row": int(row.array_row),
            "ffid": int(row.ffid),
            "split": str(row.split),
            "kept_array_row": int(row.kept_array_row),
            "kept_ffid": int(row.kept_ffid),
            "kept_split": str(row.kept_split),
        }
        for row in removed_details[
            ["array_row", "ffid", SPLIT_COLUMN, "kept_array_row", "kept_ffid", "kept_split"]
        ].itertuples(index=False)
    ]
    audit: dict[str, object] = {
        "policy": DUPLICATE_PHYSICAL_COORDINATE_POLICY,
        "physical_coordinate_key": keys,
        "scope": "all_amplitude_eligible_splits_before_ffid_selection",
        "winner_rule": "lowest_array_row",
        "winner_selection_uses_split": False,
        "winner_selection_uses_amplitude": False,
        "input_eligible_trace_count": len(eligible),
        "duplicate_physical_cell_count": int(duplicate_rows.drop_duplicates(keys).shape[0]),
        "duplicate_physical_row_count": len(duplicate_rows),
        "removed_trace_count": len(removed),
        "removed_counts_by_split": removed_counts_by_split,
        "removed_counts_by_ffid": {
            str(int(ffid)): int(count) for ffid, count in removed_ffid_counts.items()
        },
        "removed_rows": removed_records,
        "retained_eligible_trace_count": len(canonical_eligible),
        "remaining_duplicate_physical_cell_count": int(
            remaining_duplicate_rows.drop_duplicates(keys).shape[0]
        ),
        "remaining_duplicate_physical_row_count": len(remaining_duplicate_rows),
    }
    return canonical, audit
