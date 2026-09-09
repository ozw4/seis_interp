"""Check one new C3 partition and its authorized canonical training pool."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from seis_interp.data.prepared_partition_inputs import validated_prepared_split_rows
from seis_interp.processing.c3_volume_index import validated_index_range
from seis_interp.processing.trace_amplitude_filter import (
    TraceAmplitudeFilterConfig,
    filter_trace_amplitudes,
    validated_trace_amplitude_filter_config,
)
from seis_interp.processing.trace_canonicalization import (
    PHYSICAL_COORDINATE_COLUMNS,
    canonicalize_eligible_physical_coordinates,
)
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    assign_c3_source_line_block_splits,
    validate_prepared_split_assignments,
    validated_c3_source_line_ranges,
)


def validated_c3_benchmark_amplitude_filter(
    sampling: Mapping,
    preparation: Mapping | None = None,
) -> TraceAmplitudeFilterConfig | None:
    """Require an explicit policy and, when available, its exact prepared binding."""
    policy = (
        validated_trace_amplitude_filter_config(
            sampling["trace_amplitude_filter"], name="sampling.trace_amplitude_filter"
        )
        if "trace_amplitude_filter" in sampling
        else None
    )
    if preparation is not None:
        declared = preparation.get("trace_amplitude_filter")
        expected = None if policy is None else policy.to_dict()
        if declared != expected or (policy is None and "trace_amplitude_filter" in preparation):
            raise ValueError("benchmark amplitude filter differs between config and preparation")
    return policy


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
    sampling: Mapping | None = None,
    amplitudes: np.ndarray | None = None,
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
    policy = validated_c3_benchmark_amplitude_filter(
        {} if sampling is None else sampling, preparation
    )
    quality = None
    original_canonical = None
    if policy is not None:
        if amplitudes is None or amplitudes.shape != (len(table), preparation.get("sample_count")):
            raise ValueError(
                "explicit benchmark amplitude QC requires all raw rows and time samples"
            )
        original_canonical, _ = canonicalize_eligible_physical_coordinates(expected)
        quality = _audit_trace_amplitude_filter(table, amplitudes, policy, preparation)
        expected.loc[expected["array_row"].isin(quality["excluded_array_rows"]), "split"] = (
            EXCLUDED_SPLIT
        )
    expected_roles = expected.set_index("array_row")["split"]
    if not np.array_equal(
        joined["split"].to_numpy(), expected_roles.loc[joined["array_row"]].to_numpy()
    ):
        if policy is None:
            raise ValueError(
                "benchmark partition must contain the unfiltered source-line assignments"
            )
        raise ValueError(
            "benchmark partition differs from independently filtered source-line assignments"
        )
    eligible = joined.loc[joined["split"].ne(EXCLUDED_SPLIT)]
    if eligible.groupby(list(PHYSICAL_COORDINATE_COLUMNS))["split"].nunique().gt(1).any():
        raise ValueError("canonical physical pair crosses partition roles")
    if eligible.groupby("ffid")["split"].nunique().gt(1).any():
        raise ValueError("FFID crosses partition roles")
    canonical, audit = canonicalize_eligible_physical_coordinates(joined)
    if original_canonical is not None:
        remaining = canonical.loc[canonical["split"].ne(EXCLUDED_SPLIT), "array_row"]
        if not remaining.isin(original_canonical["array_row"]).all():
            raise ValueError("amplitude QC would promote a noncanonical physical-cell alias")
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
    summary = {
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
    if quality is not None:
        assert policy is not None and amplitudes is not None
        summary["amplitude_qc"] = {
            "policy": policy.to_dict(),
            "time_samples": [0, amplitudes.shape[1]],
            "scope": "all raw trace samples before model training-time selection",
            **quality,
        }
    return rows, summary


def _audit_trace_amplitude_filter(table, amplitudes, policy, preparation):
    """Recompute exclusions and every native quality count from raw samples."""
    result = filter_trace_amplitudes(amplitudes, policy)
    excluded = np.setdiff1d(table["array_row"], result.eligible_array_rows, assume_unique=True)
    affected_ffids = set(table.loc[table["array_row"].isin(excluded), "ffid"])
    retained_ffids = set(table.loc[table["array_row"].isin(result.eligible_array_rows), "ffid"])
    measured = {
        "input_trace_count": len(table),
        "eligible_trace_count": len(result.eligible_array_rows),
        "excluded_trace_count": len(excluded),
        "all_zero_trace_count": len(result.all_zero_array_rows),
        "excess_amplitude_trace_count": len(result.excess_amplitude_array_rows),
        "excluded_array_rows": excluded.tolist(),
        "affected_ffids": sorted(int(value) for value in affected_ffids),
        "fully_excluded_ffids": sorted(int(value) for value in affected_ffids - retained_ffids),
    }
    if preparation.get("trace_quality") != measured:
        raise ValueError("prepared amplitude QC exclusions/counts differ from raw samples")
    return measured
