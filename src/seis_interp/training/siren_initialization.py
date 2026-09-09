"""Apply optional training-time initialization to an already constructed SIREN."""

from __future__ import annotations

import math
from numbers import Real

import torch

from seis_interp.models.siren import Siren


def apply_siren_time_weight_initialization(model: Siren, scale: float = 1.0) -> None:
    """Scale only the first sine layer's time-input weights in place.

    The caller's coordinate contract must put time in input column zero. Apply
    this once after model construction, before training. No random values are
    drawn, and a unit scale leaves even tensor mutation versions unchanged.
    This initialization does not change the forward or checkpoint contract.
    """
    if not isinstance(model, Siren):
        raise TypeError("model must be a Siren")
    error_message = "scale must be a positive finite number"
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise ValueError(error_message)
    try:
        multiplier = float(scale)
    except (OverflowError, ValueError) as error:
        raise ValueError(error_message) from error
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(error_message)
    if multiplier == 1.0:
        return
    with torch.no_grad():
        model.network[0].linear.weight[:, 0].mul_(multiplier)
