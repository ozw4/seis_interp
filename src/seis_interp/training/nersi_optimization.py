"""Explicit learning-rate and final-weight averaging options for NeRSI."""

import math
from collections.abc import Mapping

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError


def validate_nersi_optimization(value: object, learning_rate: float) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "learning_rate_schedule",
        "minimum_learning_rate",
        "ema_decay",
    }:
        raise ConfigurationError("optimization requires schedule, minimum rate, and EMA decay")
    schedule = value["learning_rate_schedule"]
    if schedule not in ("constant", "cosine"):
        raise ConfigurationError("learning_rate_schedule must be constant or cosine")
    minimum = config_values.positive_float(value["minimum_learning_rate"], "minimum_learning_rate")
    if minimum > learning_rate or (schedule == "constant" and minimum != learning_rate):
        raise ConfigurationError(
            "minimum_learning_rate must not exceed initial rate; constant requires equality"
        )
    decay = value["ema_decay"]
    if decay is not None:
        decay = config_values.probability(decay, "ema_decay")
        if decay == 0:
            raise ConfigurationError("ema_decay must be in (0, 1) or null")
    return dict(learning_rate_schedule=schedule, minimum_learning_rate=minimum, ema_decay=decay)


def nersi_step_learning_rate(step: int, steps: int, initial: float, options: dict) -> float:
    """Use the initial rate on update one and the minimum on the final update."""
    if options["learning_rate_schedule"] == "constant" or steps == 1:
        return initial
    minimum = options["minimum_learning_rate"]
    return minimum + (initial - minimum) * (1 + math.cos(math.pi * (step - 1) / (steps - 1))) / 2
