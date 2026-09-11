"""Pure summaries of recorded graph training costs for run comparisons."""

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from statistics import fmean

_MEAN_FIELDS = {
    "seconds": "mean_recorded_step_seconds",
    "batch_preparation_seconds": "mean_batch_preparation_seconds",
    "optimization_seconds": "mean_optimization_seconds",
    "subgraph_node_count": "mean_subgraph_node_count",
    "subgraph_support_node_count": "mean_subgraph_support_node_count",
    "subgraph_edge_count": "mean_subgraph_edge_count",
    "subgraph_max_depth": "mean_subgraph_max_depth",
}


def summarize_trace_graph_training(history: Sequence[Mapping]) -> dict[str, int | float]:
    """Summarize nonempty step history without mutating or rounding its values."""
    if not history:
        raise ValueError("training history must not be empty")
    profile = {"optimizer_updates": len(history)}
    for key, output_key in _MEAN_FIELDS.items():
        values = []
        for row in history:
            if key not in row:
                raise ValueError(f"training history is missing {key}")
            value = row[key]
            numeric_type = Real if key.endswith("seconds") else Integral
            if (
                isinstance(value, bool)
                or not isinstance(value, numeric_type)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"training history {key} must be finite and nonnegative")
            values.append(value)
        mean = fmean(values)
        if not math.isfinite(mean):
            raise ValueError(f"training history mean for {key} must be finite")
        profile[output_key] = mean
        if key in ("subgraph_node_count", "subgraph_edge_count"):
            profile[f"max_{key}"] = int(max(values))
    return profile


def optimizer_updates_per_second(updates: int, training_seconds: float) -> float:
    """Use the full training interval, including index/episode setup overhead."""
    if isinstance(updates, bool) or not isinstance(updates, Integral) or updates < 1:
        raise ValueError("optimizer updates must be a positive integer")
    if (
        isinstance(training_seconds, bool)
        or not isinstance(training_seconds, Real)
        or not math.isfinite(training_seconds)
        or training_seconds <= 0
    ):
        raise ValueError("training_seconds must be finite and positive")
    rate = float(updates / training_seconds)
    if not math.isfinite(rate):
        raise ValueError("optimizer_updates_per_second must be finite")
    return rate
