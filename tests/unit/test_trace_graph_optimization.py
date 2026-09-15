import pytest

from seis_interp.training.trace_graph_optimization import (
    trace_graph_step_learning_rate,
    validate_trace_graph_schedule,
)


def test_hold_cosine_endpoints_and_monotonicity():
    schedule = validate_trace_graph_schedule(
        dict(kind="constant_then_cosine", hold_steps=10000, minimum_learning_rate=0.00003),
        0.001,
        20000,
    )
    rates = [trace_graph_step_learning_rate(i, 20000, 0.001, schedule) for i in range(1, 20001)]
    assert rates[:10000] == [0.001] * 10000
    assert rates[14999] == pytest.approx(0.000515)
    assert rates[-1] == 0.00003
    assert all(a >= b for a, b in zip(rates, rates[1:], strict=False))


@pytest.mark.parametrize(
    "key,value",
    [
        ("kind", "unknown"),
        ("hold_steps", 0),
        ("hold_steps", True),
        ("hold_steps", 20),
        ("minimum_learning_rate", float("nan")),
        ("minimum_learning_rate", 0),
        ("minimum_learning_rate", 0.1),
    ],
)
def test_reject_invalid_schedule(key, value):
    schedule = dict(kind="constant_then_cosine", hold_steps=10, minimum_learning_rate=0.00003)
    schedule[key] = value
    with pytest.raises(ValueError):
        validate_trace_graph_schedule(schedule, 0.001, 20)
