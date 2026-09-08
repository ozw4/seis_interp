"""Fixed exact-neighbor settings shared by training and frozen prediction."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class TraceGraphSettings:
    """Relation scales in the fixed source, receiver, CMP, offset order.

    Dependency rounds are deliberately absent: callers derive them from the
    model so graph construction and message passing cannot diverge.
    """

    relation_scales_m: tuple[tuple[float, float], ...]
    neighbors_per_relation: int = 8
    radius: float = 1.0
    candidate_chunk_size: int = 4096

    def __post_init__(self) -> None:
        raw = np.asarray(self.relation_scales_m)
        if raw.shape != (4, 2) or raw.dtype.kind not in "iuf":
            raise ValueError("relation_scales_m must contain positive numbers with shape [4, 2]")
        scales = raw.astype(np.float64)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("relation_scales_m must be finite and positive")
        object.__setattr__(
            self, "relation_scales_m", tuple(tuple(map(float, row)) for row in scales)
        )
        for name in ("neighbors_per_relation", "candidate_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, Real)
            or not np.isfinite(self.radius)
            or self.radius <= 0
        ):
            raise ValueError("radius must be finite and positive")
        object.__setattr__(self, "radius", float(self.radius))

    def constructor_config(self) -> dict[str, object]:
        """Return an independent JSON-compatible description."""
        return {
            "relation_scales_m": [list(row) for row in self.relation_scales_m],
            "neighbors_per_relation": self.neighbors_per_relation,
            "radius": self.radius,
            "candidate_chunk_size": self.candidate_chunk_size,
        }

    def subgraph_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for exact dependency-graph construction."""
        return self.constructor_config()
