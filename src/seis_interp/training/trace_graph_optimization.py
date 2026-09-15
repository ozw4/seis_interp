"""Fixed-budget learning-rate schedules for observed-only graph training."""

import math
from collections.abc import Mapping

from seis_interp import config_values as values
from seis_interp.configuration import ConfigurationError


def validate_trace_graph_schedule(value: object, initial: float, steps: int) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "kind",
        "hold_steps",
        "minimum_learning_rate",
    }:
        raise ConfigurationError(
            "learning_rate_schedule requires kind, hold_steps and minimum_learning_rate"
        )
    if value["kind"] != "constant_then_cosine":
        raise ConfigurationError("learning_rate_schedule.kind must be constant_then_cosine")
    hold = values.positive_integer(value["hold_steps"], "hold_steps")
    minimum = values.positive_float(value["minimum_learning_rate"], "minimum_learning_rate")
    if hold >= steps:
        raise ConfigurationError("hold_steps must be less than max_steps")
    if minimum > initial:
        raise ConfigurationError("minimum_learning_rate must not exceed learning_rate")
    return dict(kind="constant_then_cosine", hold_steps=hold, minimum_learning_rate=minimum)


def trace_graph_step_learning_rate(step: int, steps: int, initial: float, schedule: dict) -> float:
    """Hold through hold_steps; reach the minimum on the last optimizer update."""
    hold = schedule["hold_steps"]
    if step <= hold:
        return initial
    minimum = schedule["minimum_learning_rate"]
    progress = (step - hold) / (steps - hold)
    return minimum + (initial - minimum) * (1 + math.cos(math.pi * progress)) / 2
