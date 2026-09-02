"""Train-only source-axis bracketing for held-out-shot interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
)

_REQUIRED_COLUMNS = (
    "array_row",
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
)
_COORDINATE_COLUMNS = _REQUIRED_COLUMNS[1:]


@dataclass(frozen=True)
class SourceBracketingBatch:
    """Nearest strict source-y brackets and their interpolation weights."""

    positions: np.ndarray
    weights: np.ndarray


class SameLineReceiverBracketingLookup:
    """Find train-only source brackets at an exact relative receiver location.

    Candidate keys are ``(source_x, receiver_x - source_x,
    receiver_y - source_y)``. The nearest candidate strictly below and strictly
    above the target source-y are retained. A candidate from the target FFID is
    skipped even if an unusual input assigns one FFID to multiple source points.
    """

    def __init__(
        self,
        trace_table: pd.DataFrame,
        train_available: np.ndarray,
        *,
        ffids_by_position: np.ndarray,
    ) -> None:
        array_rows, coordinates = _validated_trace_table(trace_table)
        available = _validated_available_mask(train_available, len(trace_table))
        ffids = _validated_ffids(ffids_by_position, len(trace_table))
        if not np.any(available):
            raise ValueError("train_available must select at least one trace row")

        source_x = coordinates[:, 0]
        source_y = coordinates[:, 1]
        relative_receiver_x = coordinates[:, 2] - source_x
        relative_receiver_y = coordinates[:, 3] - source_y
        key_index = pd.MultiIndex.from_arrays(
            (source_x, relative_receiver_x, relative_receiver_y),
            names=("source_x_m", "relative_receiver_x_m", "relative_receiver_y_m"),
        )
        group_codes, unique_keys = pd.factorize(key_index, sort=True)
        if np.any(group_codes < 0):
            raise ValueError("source bracketing keys must not contain missing values")

        train_positions = np.flatnonzero(available).astype(np.int64)
        train_order = np.lexsort(
            (
                array_rows[train_positions],
                source_y[train_positions],
                group_codes[train_positions],
            )
        )
        sorted_train = train_positions[train_order]
        sorted_train_groups = group_codes[sorted_train]
        group_count = len(unique_keys)
        train_counts = np.bincount(sorted_train_groups, minlength=group_count)
        train_starts = np.concatenate(([0], np.cumsum(train_counts, dtype=np.int64)))

        all_order = np.argsort(group_codes, kind="stable")
        all_counts = np.bincount(group_codes, minlength=group_count)
        all_starts = np.concatenate(([0], np.cumsum(all_counts, dtype=np.int64)))
        lower = np.full(len(trace_table), -1, dtype=np.int64)
        upper = np.full(len(trace_table), -1, dtype=np.int64)

        populated_groups = np.flatnonzero(train_counts)
        for group in populated_groups:
            candidates = sorted_train[train_starts[group] : train_starts[group + 1]]
            targets = all_order[all_starts[group] : all_starts[group + 1]]
            candidate_source_y = source_y[candidates]
            target_source_y = source_y[targets]
            lower_indices = np.searchsorted(candidate_source_y, target_source_y, side="left") - 1
            upper_indices = np.searchsorted(candidate_source_y, target_source_y, side="right")
            lower[targets] = _distinct_ffid_candidates(
                candidates,
                lower_indices,
                step=-1,
                target_ffids=ffids[targets],
                candidate_ffids=ffids[candidates],
            )
            upper[targets] = _distinct_ffid_candidates(
                candidates,
                upper_indices,
                step=1,
                target_ffids=ffids[targets],
                candidate_ffids=ffids[candidates],
            )

        self._row_count = len(trace_table)
        self._source_y = source_y
        self._ffids = ffids
        self._train_available = available
        self._lower = lower
        self._upper = upper

    @property
    def row_count(self) -> int:
        return self._row_count

    def batch(self, target_positions: np.ndarray) -> SourceBracketingBatch:
        """Return lower/upper train positions and linear/nearest weights."""
        targets = _validated_target_positions(target_positions, self._row_count)
        positions = np.column_stack((self._lower[targets], self._upper[targets]))
        available = positions >= 0
        weights = np.zeros(positions.shape, dtype=np.float32)
        both = available[:, 0] & available[:, 1]
        if np.any(both):
            lower_y = self._source_y[positions[both, 0]]
            upper_y = self._source_y[positions[both, 1]]
            target_y = self._source_y[targets[both]]
            span = upper_y - lower_y
            if np.any(span <= 0.0):
                raise RuntimeError("source brackets must strictly surround their target")
            weights[both, 0] = ((upper_y - target_y) / span).astype(np.float32)
            weights[both, 1] = ((target_y - lower_y) / span).astype(np.float32)
        lower_only = available[:, 0] & ~available[:, 1]
        upper_only = ~available[:, 0] & available[:, 1]
        weights[lower_only, 0] = 1.0
        weights[upper_only, 1] = 1.0
        return SourceBracketingBatch(positions=positions, weights=weights)

    def audit(self, target_positions: np.ndarray) -> dict[str, object]:
        """Summarize coverage and prove that every reference source is train-only."""
        targets = _validated_target_positions(target_positions, self._row_count)
        batch = self.batch(targets)
        available = batch.positions >= 0
        reference_count = np.count_nonzero(available, axis=1)
        safe_positions = np.maximum(batch.positions, 0)
        target_ffid_entries = int(
            np.count_nonzero(
                available & (self._ffids[safe_positions] == self._ffids[targets, None])
            )
        )
        same_source_y_entries = int(
            np.count_nonzero(
                available & (self._source_y[safe_positions] == self._source_y[targets, None])
            )
        )
        non_train_entries = int(
            np.count_nonzero(available & ~self._train_available[safe_positions])
        )
        source_entry_count = int(np.count_nonzero(available))
        return {
            "row_count": len(targets),
            "bracketed_rows": int(np.count_nonzero(reference_count == 2)),
            "one_sided_rows": int(np.count_nonzero(reference_count == 1)),
            "unresolved_rows": int(np.count_nonzero(reference_count == 0)),
            "source_entry_count": source_entry_count,
            "source_split_counts": {
                "train": source_entry_count - non_train_entries,
                "non_train": non_train_entries,
            },
            "target_ffid_reference_entries": target_ffid_entries,
            "same_source_y_reference_entries": same_source_y_entries,
        }


def _distinct_ffid_candidates(
    candidates: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    step: int,
    target_ffids: np.ndarray,
    candidate_ffids: np.ndarray,
) -> np.ndarray:
    """Resolve candidate indices while skipping all target-FFID entries."""
    indices = candidate_indices.copy()
    valid = (indices >= 0) & (indices < len(candidates))
    while np.any(valid):
        rows = np.flatnonzero(valid)
        conflicts = candidate_ffids[indices[rows]] == target_ffids[rows]
        if not np.any(conflicts):
            break
        indices[rows[conflicts]] += step
        valid = (indices >= 0) & (indices < len(candidates))
    result = np.full(len(indices), -1, dtype=np.int64)
    valid = (indices >= 0) & (indices < len(candidates))
    result[valid] = candidates[indices[valid]]
    return result


def _validated_trace_table(trace_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError("trace_table must be a pandas DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in trace_table.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")
    if not trace_table.index.equals(pd.RangeIndex(len(trace_table))):
        raise ValueError("trace table index must be a zero-based contiguous RangeIndex")
    array_rows = trace_table["array_row"].to_numpy()
    if is_bool_dtype(array_rows.dtype) or not is_integer_dtype(array_rows.dtype):
        raise ValueError("array_row must have an integer dtype")
    if len(np.unique(array_rows)) != len(array_rows):
        raise ValueError("array_row values must be unique")
    non_numeric = [
        column
        for column in _COORDINATE_COLUMNS
        if is_bool_dtype(trace_table[column].dtype)
        or not (
            is_integer_dtype(trace_table[column].dtype) or is_float_dtype(trace_table[column].dtype)
        )
    ]
    if non_numeric:
        raise ValueError(f"physical coordinate columns must be numeric: {non_numeric}")
    coordinates = trace_table[list(_COORDINATE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("physical coordinates must be finite")
    return array_rows.astype(np.int64, copy=False), coordinates


def _validated_available_mask(value: np.ndarray, row_count: int) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (row_count,):
        raise ValueError(f"train_available must be boolean with shape ({row_count},)")
    return mask


def _validated_ffids(value: np.ndarray, row_count: int) -> np.ndarray:
    ffids = np.asarray(value)
    if is_bool_dtype(ffids.dtype) or not is_integer_dtype(ffids.dtype):
        raise ValueError("ffids_by_position must have an integer dtype")
    if ffids.shape != (row_count,):
        raise ValueError(f"ffids_by_position must have shape ({row_count},)")
    return ffids.astype(np.int64, copy=False)


def _validated_target_positions(value: np.ndarray, row_count: int) -> np.ndarray:
    positions = np.asarray(value)
    if (
        positions.ndim != 1
        or is_bool_dtype(positions.dtype)
        or not is_integer_dtype(positions.dtype)
    ):
        raise ValueError("target_positions must be a one-dimensional integer array")
    if positions.size and (np.any(positions < 0) or np.any(positions >= row_count)):
        raise ValueError(f"target_positions must be within [0, {row_count})")
    return positions.astype(np.int64, copy=False)
