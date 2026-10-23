"""Profiles of the profile-wise NeRSI on a FORGE grid and their exact-count sampling.

A profile fixes one source cell and one receiver line; its columns run along the
receiver points. Its three coordinates are unrounded actual positions: the
source's continuous line and point, and the mean continuous line of the receiver
line's stations. Only the receiver point is a grid index (the profile column).
Coordinates are divided by ``max(n - 1, 1)`` per axis, as for ``nersi_real``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SOURCE_KEYS = ["source_grid_line", "source_grid_point"]
RECEIVER_KEYS = ["receiver_grid_line", "receiver_grid_point"]


@dataclass(frozen=True)
class NersiProfileLayout:
    """``coordinates[p]`` are the normalized inputs of profile ``p``.

    Cell IDs follow C order over ``spatial_shape``, so a cell's profile is
    ``cell // receiver_points`` and its column ``cell % receiver_points``.
    """

    coordinates: np.ndarray
    receiver_points: int

    def locate(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cells = np.asarray(cells, dtype=np.int64)
        return cells // self.receiver_points, cells % self.receiver_points


def nersi_profile_layout(geometry: pd.DataFrame, spatial_shape) -> NersiProfileLayout:
    """Profile coordinates from one geometry table; every source cell and receiver line exists."""
    lines, points, receiver_lines, receiver_points = (int(n) for n in spatial_shape)
    expected = (
        (geometry.source_grid_line * points + geometry.source_grid_point) * receiver_lines
        + geometry.receiver_grid_line
    ) * receiver_points + geometry.receiver_grid_point
    if not np.array_equal(expected.to_numpy(), geometry.cell_id.to_numpy()):
        raise ValueError("cell IDs must follow C order over the spatial grid")
    source = _unique_positions(
        geometry, SOURCE_KEYS, ["source_line_continuous", "source_point_continuous"]
    )
    receivers = _unique_positions(geometry, RECEIVER_KEYS, ["receiver_line_continuous"])
    receiver_lines_position = receivers.groupby(level=0).receiver_line_continuous.mean()
    source_cells = pd.MultiIndex.from_product([range(lines), range(points)])
    if not source.index.equals(source_cells) or not np.array_equal(
        receiver_lines_position.index, np.arange(receiver_lines)
    ):
        raise ValueError("every source cell and receiver line needs an actual position")
    coordinates = np.column_stack(
        (
            np.repeat(source.to_numpy(), receiver_lines, axis=0),
            np.tile(receiver_lines_position.to_numpy(), lines * points),
        )
    )
    scale = np.maximum(np.array([lines, points, receiver_lines]) - 1, 1)
    return NersiProfileLayout(coordinates / scale, receiver_points)


def _unique_positions(geometry, keys, columns):
    """One actual position per grid cell, sorted by the cell's grid indices."""
    grouped = geometry.groupby(keys)[columns]
    if (grouped.max() - grouped.min()).to_numpy().max() != 0:
        raise ValueError(f"cells of {keys} must each have one actual position")
    return grouped.first().sort_index()


class ExactTraceProfileSampler:
    """Draw whole profiles until they hold ``traces`` observed traces, then keep exactly that many.

    Each update visits the profiles holding an observed trace in a fresh random
    order without replacement, stops at the first prefix with at least ``traces``
    observed traces, and selects ``traces`` of the pooled traces uniformly
    without replacement (the CCNet5D exact-count rule applied to profiles).
    ``sample`` returns indices into ``profiles`` (one entry per observed trace).
    """

    def __init__(self, profiles: np.ndarray, traces: int, seed: int) -> None:
        profiles = np.asarray(profiles, dtype=np.int64)
        if traces < 1 or len(profiles) < traces:
            raise ValueError("fewer observed traces than the traces per update")
        order = np.argsort(profiles, kind="stable")
        self.eligible, starts, self.counts = np.unique(
            profiles[order], return_index=True, return_counts=True
        )
        self._members = np.split(order, starts[1:])
        self.traces = traces
        self.rng = np.random.default_rng(seed)
        self.profiles_drawn = 0

    def sample(self) -> np.ndarray:
        visit = self.rng.permutation(len(self.eligible))
        drawn = int(np.searchsorted(np.cumsum(self.counts[visit]), self.traces)) + 1
        pool = np.concatenate([self._members[i] for i in visit[:drawn]])
        self.profiles_drawn += drawn
        return pool[self.rng.choice(len(pool), size=self.traces, replace=False)]
