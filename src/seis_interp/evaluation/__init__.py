"""Evaluation helpers for interpolation predictions."""

from seis_interp.evaluation.baselines import (
    inverse_distance_weighted_predict,
    nearest_neighbor_predict,
)

__all__ = ["inverse_distance_weighted_predict", "nearest_neighbor_predict"]
