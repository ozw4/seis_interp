"""Parameter-free regularizers of profile-wise NeRSI training (no model change).

``normalized_nuclear_norm`` is the low-rank profile penalty of Gao et al. (2025,
eq. 11), divided by ``sqrt(time * receivers)`` so that a profile of unit-RMS traces
scores between 1 (rank one) and ``sqrt(min(time, receivers))``.
``ProfileCoordinateJitter`` perturbs the inputs of training profiles within a
fraction of one grid cell while their observed labels stay unchanged; it uses its
own random stream so that the trace sampling sequence does not change.
"""

import math
from numbers import Real

import numpy as np
import torch

JITTER_STREAM = 1


def normalized_nuclear_norm(profiles: torch.Tensor) -> torch.Tensor:
    """Mean of ``||P||_* / sqrt(rows * columns)`` over a ``(profiles, rows, columns)`` tensor."""
    if profiles.ndim != 3 or not torch.is_floating_point(profiles):
        raise ValueError("profiles must be a floating (profiles, rows, columns) tensor")
    scale = math.sqrt(profiles.shape[1] * profiles.shape[2])
    return torch.linalg.svdvals(profiles).sum(dim=1).mean() / scale


def validate_nuclear_norm_weight(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("training.nuclear_norm_weight must be a finite number >= 0")
    if value < 0:
        raise ValueError("training.nuclear_norm_weight must be a finite number >= 0")
    return float(value)


class ProfileCoordinateJitter:
    """Uniform offsets of at most ``cells`` grid cells on each normalized coordinate axis."""

    def __init__(self, cells: float, axis_lengths, seed: int) -> None:
        if (
            isinstance(cells, bool)
            or not isinstance(cells, Real)
            or not math.isfinite(cells)
            or not 0 < cells <= 0.5
        ):
            raise ValueError("training.coordinate_jitter_cells must be in (0, 0.5]")
        lengths = np.asarray(axis_lengths, dtype=np.float64)
        self.radius = float(cells) / np.maximum(lengths - 1, 1)
        self.rng = np.random.default_rng([seed, JITTER_STREAM])

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        offsets = self.rng.uniform(-1.0, 1.0, size=tuple(coordinates.shape)) * self.radius
        return coordinates + torch.from_numpy(offsets.astype(np.float32))
