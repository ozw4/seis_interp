"""Build masked target and context gathers from a verified C3 volume."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from seis_interp.data.c3_volume_adapter import (
    ObservedC3Volume,
    load_observed_c3_volume,
)
from seis_interp.data.c3_volume_index_store import (
    load_c3_volume_index,
    validate_c3_volume_index,
)
from seis_interp.data.masked_gather_inputs import (
    MaskedGatherInputs,
    validate_masked_gather_inputs,
)
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT
from seis_interp.processing.interpolation_masks import EVALUATION_TARGET_ROLE, OBSERVED_ROLE

CONTEXT_SELECTION = "nearest_observed_source_positions"
SOURCE_DISTANCE = "euclidean_source_xy_m"
TARGET_COORDINATE_SCALING = "volume_source_minmax"


class MaskedC3GatherSource:
    """Materialize requested masked C3 gathers without copying the full volume."""

    def __init__(
        self,
        observed_volume: ObservedC3Volume,
        index_table: pd.DataFrame,
        volume_metadata: Mapping[str, object],
        *,
        context_gather_count: int,
        device: torch.device | str,
    ) -> None:
        validate_c3_volume_index(index_table, volume_metadata)
        if isinstance(context_gather_count, bool) or not isinstance(context_gather_count, Integral):
            raise ValueError("context_gather_count must be a positive integer")
        if int(context_gather_count) <= 0:
            raise ValueError("context_gather_count must be a positive integer")
        self._context_gather_count = int(context_gather_count)
        self._device = torch.device(device)

        _validate_observed_volume(observed_volume, volume_metadata)
        spatial_shape = tuple(int(value) for value in volume_metadata["shape"][1:])  # type: ignore[index]
        line_count, shot_count, _, _ = spatial_shape
        self._source_count = line_count * shot_count

        expected_rows = index_table["array_row"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed_volume.array_rows.reshape(-1), expected_rows):
            raise ValueError("observed volume array_rows do not match volume index row order")

        observed_flat = observed_volume.observed_trace_mask.reshape(
            self._source_count,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
        )
        target_flat = observed_volume.evaluation_target_trace_mask.reshape(
            self._source_count,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
        )
        if np.any(observed_flat & target_flat) or not np.all(observed_flat | target_flat):
            raise ValueError(
                "observed and evaluation target masks must disjointly cover the volume"
            )
        actual_role_counts = {
            OBSERVED_ROLE: int(np.count_nonzero(observed_flat)),
            EVALUATION_TARGET_ROLE: int(np.count_nonzero(target_flat)),
        }
        if actual_role_counts != dict(volume_metadata["role_counts"]):  # type: ignore[arg-type]
            raise ValueError(
                f"observed volume role counts {actual_role_counts} do not match volume metadata"
            )

        self._source_indices = np.column_stack(
            np.unravel_index(
                np.arange(self._source_count, dtype=np.int64), (line_count, shot_count)
            )
        ).astype(np.int64, copy=False)
        receiver_cell_count = RECEIVER_X_COUNT * RECEIVER_Y_COUNT
        ffids_by_receiver = (
            index_table["ffid"]
            .to_numpy(dtype=np.int64)
            .reshape(self._source_count, receiver_cell_count)
        )
        source_coordinates_by_receiver = (
            index_table[["source_x_m", "source_y_m"]]
            .to_numpy(dtype=np.float64)
            .reshape(self._source_count, receiver_cell_count, 2)
        )
        if np.any(ffids_by_receiver != ffids_by_receiver[:, :1]):
            raise ValueError("ffid must be unique within each source cell")
        if np.any(source_coordinates_by_receiver != source_coordinates_by_receiver[:, :1, :]):
            raise ValueError("source coordinates must be unique within each source cell")
        self._ffids = ffids_by_receiver[:, 0].copy()
        self._source_coordinates_m = source_coordinates_by_receiver[:, 0, :].copy()
        self._array_rows = observed_volume.array_rows.reshape(
            self._source_count,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
        )
        self._observed_mask = observed_flat
        self._evaluation_target_mask = target_flat
        self._shot_major_values = observed_volume.values.transpose(1, 2, 3, 4, 0).reshape(
            self._source_count,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
            observed_volume.values.shape[0],
        )

        self._target_source_flat_indices = np.flatnonzero(
            target_flat.reshape(self._source_count, -1).any(axis=1)
        ).astype(np.int64, copy=False)
        observed_source_indices = np.flatnonzero(
            observed_flat.reshape(self._source_count, -1).any(axis=1)
        ).astype(np.int64, copy=False)
        self._context_source_indices = self._select_context_sources(observed_source_indices)

        coordinate_minimum = self._source_coordinates_m.min(axis=0)
        coordinate_maximum = self._source_coordinates_m.max(axis=0)
        coordinate_denominator = np.where(
            coordinate_maximum > coordinate_minimum,
            coordinate_maximum - coordinate_minimum,
            1.0,
        )
        self._normalized_source_coordinates = (
            self._source_coordinates_m - coordinate_minimum
        ) / coordinate_denominator

    @property
    def source_count(self) -> int:
        return self._source_count

    @property
    def target_count(self) -> int:
        return len(self._target_source_flat_indices)

    @property
    def context_gather_count(self) -> int:
        return self._context_gather_count

    @property
    def target_source_indices(self) -> np.ndarray:
        """Return target-local ``(source_line, shot_in_line)`` indices."""
        return self._source_indices[self._target_source_flat_indices].copy()

    @property
    def target_ffids(self) -> np.ndarray:
        return self._ffids[self._target_source_flat_indices].copy()

    @property
    def target_array_rows(self) -> np.ndarray:
        """Return source array rows on the fixed receiver grid."""
        return self._array_rows[self._target_source_flat_indices].copy()

    @property
    def target_evaluation_mask(self) -> np.ndarray:
        """Return target receiver masks without target amplitudes."""
        return self._evaluation_target_mask[self._target_source_flat_indices].copy()

    @property
    def context_source_indices(self) -> np.ndarray:
        """Return selected flat context-source indices for every target."""
        return self._context_source_indices.copy()

    def inputs(self, target_indices: np.ndarray) -> MaskedGatherInputs:
        """Materialize one ordered, potentially repeated target batch."""
        indices = _validated_target_indices(target_indices, target_count=self.target_count)
        target_source_flat = self._target_source_flat_indices[indices]
        context_source_flat = self._context_source_indices[indices]

        target_values = np.ascontiguousarray(self._shot_major_values[target_source_flat])
        target_mask = np.ascontiguousarray(self._observed_mask[target_source_flat])
        context_values = np.ascontiguousarray(self._shot_major_values[context_source_flat])
        context_mask = np.ascontiguousarray(self._observed_mask[context_source_flat])

        amplitude_dtype = self._shot_major_values.dtype
        source_deltas = np.ascontiguousarray(
            (
                self._source_coordinates_m[context_source_flat]
                - self._source_coordinates_m[target_source_flat, None, :]
            ).astype(amplitude_dtype, copy=False)
        )
        target_coordinates = np.ascontiguousarray(
            self._normalized_source_coordinates[target_source_flat].astype(
                amplitude_dtype, copy=False
            )
        )
        result = MaskedGatherInputs(
            target_observed=torch.as_tensor(target_values, device=self._device),
            target_observation_mask=torch.as_tensor(target_mask, device=self._device),
            context_gathers=torch.as_tensor(context_values, device=self._device),
            context_availability=torch.as_tensor(context_mask, device=self._device),
            source_deltas_m=torch.as_tensor(source_deltas, device=self._device),
            target_coordinates=torch.as_tensor(target_coordinates, device=self._device),
        )
        return validate_masked_gather_inputs(result)

    def _select_context_sources(self, observed_source_indices: np.ndarray) -> np.ndarray:
        for target_source_flat in self._target_source_flat_indices:
            available_count = len(observed_source_indices) - int(
                np.any(observed_source_indices == target_source_flat)
            )
            if available_count < self._context_gather_count:
                source_line, shot_in_line = self._source_indices[target_source_flat]
                raise ValueError(
                    f"target source indices ({int(source_line)}, {int(shot_in_line)}) has only "
                    f"{available_count} observed context sources; "
                    f"{self._context_gather_count} required"
                )

        selected = np.empty(
            (len(self._target_source_flat_indices), self._context_gather_count),
            dtype=np.int64,
        )
        for target_list_index, target_source_flat in enumerate(self._target_source_flat_indices):
            candidates = observed_source_indices[observed_source_indices != target_source_flat]
            deltas = (
                self._source_coordinates_m[candidates]
                - self._source_coordinates_m[target_source_flat]
            )
            squared_distances = np.sum(np.square(deltas), axis=1)
            order = np.lexsort((candidates, squared_distances))
            selected[target_list_index] = candidates[order[: self._context_gather_count]]
        return selected


def load_masked_c3_gather_source(
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    context_gather_count: int,
    device: torch.device | str,
) -> MaskedC3GatherSource:
    """Load verified artifacts through the leakage-safe C3 volume adapter."""
    observed_volume = load_observed_c3_volume(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
    )
    index_table, volume_metadata = load_c3_volume_index(volume_dir)
    return MaskedC3GatherSource(
        observed_volume,
        index_table,
        volume_metadata,
        context_gather_count=context_gather_count,
        device=device,
    )


def _validate_observed_volume(
    observed_volume: ObservedC3Volume,
    volume_metadata: Mapping[str, object],
) -> None:
    if not isinstance(observed_volume, ObservedC3Volume):
        raise TypeError("observed_volume must be an ObservedC3Volume")
    arrays = (
        ("values", observed_volume.values),
        ("time_s", observed_volume.time_s),
        ("array_rows", observed_volume.array_rows),
        ("observed_trace_mask", observed_volume.observed_trace_mask),
        ("evaluation_target_trace_mask", observed_volume.evaluation_target_trace_mask),
    )
    for name, value in arrays:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"observed_volume.{name} must be a NumPy array")

    expected_shape = tuple(int(value) for value in volume_metadata["shape"])  # type: ignore[arg-type]
    spatial_shape = expected_shape[1:]
    if expected_shape[0] < 2:
        raise ValueError("observed volume time dimension must contain at least two samples")
    if spatial_shape[2:] != (RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError("observed volume must use the fixed 8 x 68 receiver grid")
    if observed_volume.values.shape != expected_shape:
        raise ValueError("observed volume values shape must match volume metadata shape")
    if observed_volume.time_s.shape != (expected_shape[0],):
        raise ValueError("observed volume time_s shape must match the volume time dimension")
    for name, value in arrays[2:]:
        if value.shape != spatial_shape:
            raise ValueError(f"observed volume {name} shape must match volume spatial shape")

    if observed_volume.values.dtype.kind != "f":
        raise TypeError("observed volume values must have a floating-point dtype")
    if observed_volume.time_s.dtype.kind != "f":
        raise TypeError("observed volume time_s must have a floating-point dtype")
    if (
        observed_volume.array_rows.dtype.kind not in "iu"
        or observed_volume.array_rows.dtype.kind == "b"
    ):
        raise TypeError("observed volume array_rows must have an integer dtype")
    if observed_volume.observed_trace_mask.dtype != np.bool_:
        raise TypeError("observed volume observed_trace_mask must have dtype bool")
    if observed_volume.evaluation_target_trace_mask.dtype != np.bool_:
        raise TypeError("observed volume evaluation_target_trace_mask must have dtype bool")


def _validated_target_indices(target_indices: np.ndarray, *, target_count: int) -> np.ndarray:
    if not isinstance(target_indices, np.ndarray):
        raise TypeError("target_indices must be a NumPy array")
    if target_indices.ndim != 1:
        raise ValueError("target_indices must be one-dimensional")
    if target_indices.dtype.kind not in "iu" or target_indices.dtype.kind == "b":
        raise TypeError("target_indices must have an integer dtype")
    if len(target_indices) == 0:
        raise ValueError("target_indices must not be empty")
    if np.any(target_indices < 0) or np.any(target_indices >= target_count):
        raise IndexError(f"target_indices must be within [0, {target_count})")
    return target_indices.astype(np.int64, copy=False)
