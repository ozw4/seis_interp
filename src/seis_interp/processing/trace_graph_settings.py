"""Fixed exact-neighbor settings shared by training and frozen prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from seis_interp.processing.trace_graph_geometry import RELATION_NAMES


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
    topology: str = "multi_relation"
    excluded_relation: str | None = None
    common_distance_scales_m: tuple[float, float] | None = None
    single_4d_neighbors: int = 32

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
        if self.topology not in ("multi_relation", "single_4d"):
            raise ValueError("topology must be multi_relation or single_4d")
        if self.excluded_relation is not None and self.excluded_relation not in RELATION_NAMES:
            raise ValueError("excluded_relation must name one of the four seismic relations")
        if (
            isinstance(self.single_4d_neighbors, bool)
            or not isinstance(self.single_4d_neighbors, Integral)
            or self.single_4d_neighbors < 1
        ):
            raise ValueError("single_4d_neighbors must be a positive integer")
        object.__setattr__(self, "single_4d_neighbors", int(self.single_4d_neighbors))
        if self.common_distance_scales_m is not None:
            common = np.asarray(self.common_distance_scales_m)
            if (
                common.shape != (2,)
                or common.dtype.kind not in "iuf"
                or not np.all(np.isfinite(common))
                or np.any(common <= 0)
            ):
                raise ValueError("common_distance_scales_m must have two positive finite scales")
            object.__setattr__(self, "common_distance_scales_m", tuple(map(float, common)))
        if self.topology == "single_4d":
            if self.excluded_relation is not None:
                raise ValueError("single_4d cannot exclude a seismic relation")
            if self.common_distance_scales_m is None:
                raise ValueError("single_4d requires common_distance_scales_m")
        elif self.single_4d_neighbors != 32:
            raise ValueError("single_4d_neighbors applies only to single_4d topology")

    def constructor_config(self) -> dict[str, object]:
        """Return an independent JSON-compatible description."""
        config = {
            "relation_scales_m": [list(row) for row in self.relation_scales_m],
            "neighbors_per_relation": self.neighbors_per_relation,
            "radius": self.radius,
            "candidate_chunk_size": self.candidate_chunk_size,
        }
        if self.topology != "multi_relation":
            config.update(topology=self.topology, single_4d_neighbors=self.single_4d_neighbors)
        if self.excluded_relation is not None:
            config["excluded_relation"] = self.excluded_relation
        if self.common_distance_scales_m is not None:
            config["common_distance_scales_m"] = list(self.common_distance_scales_m)
        return config

    def subgraph_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for exact dependency-graph construction."""
        return self.constructor_config()

    def validate_model_config(self, model_config: Mapping[str, object]) -> None:
        """Reject combinations that would silently ignore an ablation setting."""
        variant = model_config.get("method_variant", "relational")
        if self.topology == "single_4d" and variant != "untyped_edge_conditioned":
            raise ValueError("single_4d requires the untyped_edge_conditioned model")
        if variant == "untyped_edge_conditioned" and self.common_distance_scales_m is None:
            raise ValueError("untyped_edge_conditioned requires common_distance_scales_m for D0")
        if variant != "relational" and model_config.get("relation_fusion", "mean") != "mean":
            raise ValueError("comparison models require relation_fusion=mean")
