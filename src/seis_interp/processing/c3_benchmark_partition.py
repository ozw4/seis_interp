"""Check one new C3 partition and its authorized canonical training pool."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from seis_interp.data.prepared_partition_inputs import validated_prepared_split_rows
from seis_interp.processing.c3_volume_index import validated_index_range
from seis_interp.processing.trace_canonicalization import (
    PHYSICAL_COORDINATE_COLUMNS,
    canonicalize_eligible_physical_coordinates,
)
from seis_interp.processing.trace_splits import (
    assign_c3_source_line_block_splits,
    validate_prepared_split_assignments,
    validated_c3_source_line_ranges,
)


def c3_benchmark_partition_ranges(
    source_line_range: tuple[int, int], source_line_count: int
) -> dict:
    """Place train before the fixed test block and validation after it."""
    start, stop = validated_index_range(source_line_range, name="test source_line_range")
    if start == 0 or stop >= source_line_count:
        raise ValueError(
            f"fixed test [{start},{stop}) leaves an empty train or validation block "
            f"among {source_line_count} lines"
        )
    return validated_c3_source_line_ranges(
        {"train": [0, start], "test": [start, stop], "validation": [stop, source_line_count]}
    )


def audit_c3_benchmark_partition(
    table: pd.DataFrame,
    split: pd.DataFrame,
    preparation: Mapping,
    crops: Mapping[str, pd.DataFrame],
    *,
    time_range: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Require row, FFID, physical-pair, block, and crop-role consistency."""
    validated_prepared_split_rows(
        split, expected_array_rows=table["array_row"].to_numpy(dtype=np.int64)
    )
    joined = table.merge(
        split[["array_row", "split"]], on="array_row", validate="one_to_one", sort=False
    )
    validate_prepared_split_assignments(joined, preparation)
    expected = assign_c3_source_line_block_splits(
        table, source_line_ranges=preparation["source_line_ranges"]
    )
    expected_roles = expected.set_index("array_row")["split"]
    if not np.array_equal(
        joined["split"].to_numpy(), expected_roles.loc[joined["array_row"]].to_numpy()
    ):
        raise ValueError("benchmark partition must contain the unfiltered source-line assignments")
    if joined.groupby(list(PHYSICAL_COORDINATE_COLUMNS))["split"].nunique().gt(1).any():
        raise ValueError("canonical physical pair crosses partition roles")
    if joined.groupby("ffid")["split"].nunique().gt(1).any():
        raise ValueError("FFID crosses partition roles")
    canonical, audit = canonicalize_eligible_physical_coordinates(joined)
    for partition, index in crops.items():
        allowed = canonical.loc[canonical["split"].eq(partition), "array_row"].to_numpy(
            dtype=np.int64
        )
        if not np.isin(index["array_row"], allowed).all():
            raise ValueError(f"{partition} crop contains rows outside its canonical partition")
    rows = np.sort(
        canonical.loc[canonical["split"].eq("train"), "array_row"].to_numpy(dtype=np.int64)
    )
    if not len(rows):
        raise ValueError("canonical training pool is empty")
    return rows, {
        "source_line_ranges": {
            key: list(value) for key, value in preparation["source_line_ranges"].items()
        },
        "trace_counts": {
            key: int(joined["split"].eq(key).sum()) for key in ("train", "test", "validation")
        },
        "canonical_trace_counts": {
            key: int(canonical["split"].eq(key).sum()) for key in ("train", "test", "validation")
        },
        "canonical_policy": audit["policy"],
        "removed_duplicate_trace_count": audit["removed_trace_count"],
        "training_time_samples": list(validated_index_range(time_range, name="training time")),
        "crop_trace_counts": {key: len(value) for key, value in crops.items()},
        "normalization_scope": "generic_preparation_separate_from_model_train_scales",
    }
