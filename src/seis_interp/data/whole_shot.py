"""Whole-shot gather tensors, targets, and nearest-source input assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT

if TYPE_CHECKING:
    import pandas as pd

NEIGHBORHOOD_TYPE = "nearest_train_source_gathers"
SOURCE_DISTANCE = "euclidean_source_xy_m"
TARGET_COORDINATES = ("source_x_m", "source_y_m")
TARGET_COORDINATE_SCALING = "train_minmax"


@dataclass(frozen=True)
class WholeShotTargets:
    ffids: np.ndarray
    source_coordinates_m: np.ndarray
    gathers: torch.Tensor
    availability: torch.Tensor
    neighbor_train_indices: np.ndarray

    @property
    def ffid_count(self) -> int:
        return len(self.ffids)

    @property
    def trace_count(self) -> int:
        return int(torch.count_nonzero(self.availability).cpu())


def build_gather_tensors(
    table: pd.DataFrame,
    amplitudes: np.ndarray,
    *,
    receiver_x_offsets: np.ndarray,
    receiver_y_offsets: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Place per-trace amplitudes onto zero-filled fixed-grid gather tensors."""
    if len(table) != len(amplitudes):
        raise ValueError("gather rows and amplitudes must have equal length")
    ffids = np.sort(table["ffid"].unique().astype(np.int64))
    ffid_index = {int(ffid): index for index, ffid in enumerate(ffids)}
    x_index = {float(value): index for index, value in enumerate(receiver_x_offsets)}
    y_index = {float(value): index for index, value in enumerate(receiver_y_offsets)}
    source_coordinates = np.empty((len(ffids), 2), dtype=np.float64)
    source_seen = np.zeros(len(ffids), dtype=bool)
    gather_host = np.zeros(
        (len(ffids), RECEIVER_X_COUNT, RECEIVER_Y_COUNT, amplitudes.shape[1]),
        dtype=np.float32,
    )
    availability_host = np.zeros(
        (len(ffids), RECEIVER_X_COUNT, RECEIVER_Y_COUNT),
        dtype=bool,
    )
    for row_index, row in enumerate(table.itertuples(index=False)):
        target_index = ffid_index[int(row.ffid)]
        source = np.asarray((row.source_x_m, row.source_y_m), dtype=np.float64)
        if source_seen[target_index] and not np.array_equal(
            source_coordinates[target_index],
            source,
        ):
            raise ValueError(f"FFID {int(row.ffid)} contains multiple source coordinates")
        source_coordinates[target_index] = source
        source_seen[target_index] = True
        relative_x = float(row.receiver_x_m - row.source_x_m)
        relative_y = float(row.receiver_y_m - row.source_y_m)
        try:
            receiver_x_index = x_index[relative_x]
            receiver_y_index = y_index[relative_y]
        except KeyError as error:
            raise ValueError("trace is outside the validated receiver grid") from error
        if availability_host[target_index, receiver_x_index, receiver_y_index]:
            raise ValueError(f"FFID {int(row.ffid)} has a duplicate receiver cell")
        gather_host[target_index, receiver_x_index, receiver_y_index] = amplitudes[row_index]
        availability_host[target_index, receiver_x_index, receiver_y_index] = True
    if not np.all(source_seen):
        raise AssertionError("a selected FFID was not populated")
    return (
        ffids,
        source_coordinates,
        torch.from_numpy(gather_host).to(device),
        torch.from_numpy(availability_host).to(device),
    )


def nearest_train_source_indices(
    train_ffids: np.ndarray,
    train_source_coordinates_m: np.ndarray,
    target_ffids: np.ndarray,
    target_source_coordinates_m: np.ndarray,
    *,
    source_gather_count: int,
) -> np.ndarray:
    """Rank TRAIN sources by squared distance, tie-broken by ascending FFID."""
    result = np.empty((len(target_ffids), source_gather_count), dtype=np.int64)
    for target_index, (target_ffid, target_source) in enumerate(
        zip(target_ffids, target_source_coordinates_m, strict=True)
    ):
        squared_distance = np.sum(
            np.square(train_source_coordinates_m - target_source),
            axis=1,
        )
        eligible = (train_ffids != target_ffid) & (squared_distance > 0.0)
        candidates = np.flatnonzero(eligible)
        if len(candidates) < source_gather_count:
            raise ValueError(
                f"FFID {int(target_ffid)} has only {len(candidates)} non-colliding TRAIN sources; "
                f"{source_gather_count} required"
            )
        order = np.lexsort((train_ffids[candidates], squared_distance[candidates]))
        result[target_index] = candidates[order[:source_gather_count]]
    return result


class WholeShotTensorSource:
    """Gather nearest source shots exclusively from the compact TRAIN store."""

    def __init__(
        self,
        *,
        train_ffids: np.ndarray,
        train_source_coordinates_m: np.ndarray,
        train_gathers: torch.Tensor,
        train_availability: torch.Tensor,
        source_gather_count: int,
        device: torch.device,
    ) -> None:
        self.train_ffids = np.asarray(train_ffids, dtype=np.int64)
        self.train_source_coordinates_m = np.asarray(
            train_source_coordinates_m,
            dtype=np.float64,
        )
        self.train_gathers = train_gathers
        self.train_availability = train_availability
        self.source_gather_count = source_gather_count
        self.device = device
        if self.train_source_coordinates_m.shape != (len(self.train_ffids), 2):
            raise ValueError("TRAIN source coordinates must have shape (ffids, 2)")
        if len(self.train_ffids) <= source_gather_count:
            raise ValueError(
                "source_gather_count must be smaller than the selected TRAIN FFID count"
            )
        if train_gathers.shape[:3] != (
            len(self.train_ffids),
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
        ):
            raise ValueError("TRAIN gathers must use the fixed 8 x 68 receiver grid")
        if train_availability.shape != train_gathers.shape[:3]:
            raise ValueError("TRAIN availability must match the gather receiver grid")
        unique_source_count = len(np.unique(self.train_source_coordinates_m, axis=0))
        if unique_source_count != len(self.train_ffids):
            raise ValueError("selected TRAIN FFIDs must have unique source coordinates")
        self._train_source_coordinates = torch.as_tensor(
            self.train_source_coordinates_m,
            dtype=torch.float32,
            device=device,
        )
        coordinate_min = np.min(self.train_source_coordinates_m, axis=0)
        coordinate_max = np.max(self.train_source_coordinates_m, axis=0)
        self.coordinate_min = tuple(float(value) for value in coordinate_min)
        self.coordinate_max = tuple(float(value) for value in coordinate_max)
        coordinate_range = coordinate_max - coordinate_min
        self._coordinate_min = torch.as_tensor(
            coordinate_min,
            dtype=torch.float32,
            device=device,
        )
        self._coordinate_denominator = torch.as_tensor(
            np.where(coordinate_range > 0.0, coordinate_range, 1.0),
            dtype=torch.float32,
            device=device,
        )

    def build_targets(
        self,
        *,
        ffids: np.ndarray,
        source_coordinates_m: np.ndarray,
        gathers: torch.Tensor,
        availability: torch.Tensor,
    ) -> WholeShotTargets:
        target_ffids = np.asarray(ffids, dtype=np.int64)
        target_coordinates = np.asarray(source_coordinates_m, dtype=np.float64)
        neighbor_indices = nearest_train_source_indices(
            self.train_ffids,
            self.train_source_coordinates_m,
            target_ffids,
            target_coordinates,
            source_gather_count=self.source_gather_count,
        )
        return WholeShotTargets(
            ffids=target_ffids,
            source_coordinates_m=target_coordinates,
            gathers=gathers,
            availability=availability,
            neighbor_train_indices=neighbor_indices,
        )

    def inputs(
        self,
        targets: WholeShotTargets,
        target_indices: np.ndarray,
        *,
        generator: torch.Generator | None = None,
        neighbor_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.asarray(target_indices, dtype=np.int64)
        neighbor_indices = targets.neighbor_train_indices[indices]
        neighbor_index_tensor = torch.as_tensor(
            neighbor_indices,
            dtype=torch.long,
            device=self.device,
        )
        neighbors = self.train_gathers[neighbor_index_tensor]
        availability = self.train_availability[neighbor_index_tensor].clone()
        if neighbor_dropout > 0.0:
            if generator is None:
                raise ValueError("generator is required when neighbor_dropout is positive")
            source_keep = (
                torch.rand(
                    (len(indices), self.source_gather_count, 1, 1),
                    generator=generator,
                    device=self.device,
                )
                >= neighbor_dropout
            )
            availability &= source_keep
        neighbors = neighbors * availability[..., None]
        target_source_coordinates = torch.as_tensor(
            targets.source_coordinates_m[indices],
            dtype=torch.float32,
            device=self.device,
        )
        neighbor_source_coordinates = self._train_source_coordinates[neighbor_index_tensor]
        source_deltas = neighbor_source_coordinates - target_source_coordinates[:, None]
        target_coordinates = (
            target_source_coordinates - self._coordinate_min
        ) / self._coordinate_denominator
        return neighbors, availability, source_deltas, target_coordinates

    def audit(self, targets: WholeShotTargets) -> dict[str, object]:
        neighbor_ffids = self.train_ffids[targets.neighbor_train_indices]
        target_ffid_entries = int(np.count_nonzero(neighbor_ffids == targets.ffids[:, None]))
        neighbor_availability = self.train_availability[
            torch.as_tensor(
                targets.neighbor_train_indices,
                dtype=torch.long,
                device=self.device,
            )
        ]
        receiver_coverage = neighbor_availability.any(dim=1)
        target_mask = targets.availability
        uncovered_target_cells = target_mask & ~receiver_coverage
        covered_counts = torch.count_nonzero(receiver_coverage, dim=(1, 2)).cpu().numpy()
        return {
            "target_ffid_count": targets.ffid_count,
            "source_gather_count": self.source_gather_count,
            "neighbor_source_entries": int(neighbor_ffids.size),
            "target_ffid_neighbor_entries": target_ffid_entries,
            "non_train_neighbor_entries": 0,
            "target_trace_count": targets.trace_count,
            "uncovered_target_receiver_cells": int(
                torch.count_nonzero(uncovered_target_cells).cpu()
            ),
            "receiver_cells_with_any_neighbor": {
                "min": int(np.min(covered_counts)),
                "mean": float(np.mean(covered_counts, dtype=np.float64)),
                "max": int(np.max(covered_counts)),
            },
        }
