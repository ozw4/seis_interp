import numpy as np
import pytest

from seis_interp.processing.linear_moveout import (
    align_moveout,
    moveout_delay_samples,
    unalign_moveout,
)


def test_delays_round_offset_over_velocity_to_integer_samples():
    delays = moveout_delay_samples(np.array([0.0, 99.0, 101.0, 250.0]), 100.0, 0.5)

    assert delays.dtype == np.int64
    np.testing.assert_array_equal(delays, [0, 2, 2, 5])


@pytest.mark.parametrize(
    ("offsets", "velocity", "interval"),
    [
        (np.array([[1.0]]), 100.0, 0.5),
        (np.array([-1.0]), 100.0, 0.5),
        (np.array([np.nan]), 100.0, 0.5),
        (np.array([1.0]), 0.0, 0.5),
        (np.array([1.0]), 100.0, -0.5),
        (np.array([1.0]), True, 0.5),
    ],
)
def test_delays_reject_invalid_offsets_velocity_or_interval(offsets, velocity, interval):
    with pytest.raises(ValueError):
        moveout_delay_samples(offsets, velocity, interval)


def test_alignment_flattens_a_linear_event_and_inverts_exactly():
    time_count, delays = 6, np.array([[0, 2], [3, 1]])
    values = np.zeros((time_count, 2, 2), dtype=np.float32)
    for cell, delay in np.ndenumerate(delays):
        values[(1 + delay, *cell)] = 1.0 + delay
        values[(0, *cell)] = -0.5
    canvas = align_moveout(values, delays)

    assert canvas.shape == (time_count + 3, 2, 2)
    assert canvas.dtype == values.dtype
    # Trace i is shifted later by max(delay) - delay[i]; the event lands at one canvas time.
    np.testing.assert_array_equal(canvas[4], delays.astype(np.float32) + 1.0)
    for cell, delay in np.ndenumerate(delays):
        shift = 3 - delay
        assert np.all(canvas[:shift][(slice(None), *cell)] == 0)
        np.testing.assert_array_equal(
            canvas[shift : shift + time_count][(slice(None), *cell)], values[(slice(None), *cell)]
        )
        assert np.all(canvas[shift + time_count :][(slice(None), *cell)] == 0)
    np.testing.assert_array_equal(unalign_moveout(canvas, delays, time_count), values)


def test_alignment_with_zero_delays_is_the_identity():
    values = np.random.default_rng(0).standard_normal((4, 2, 3)).astype(np.float32)
    delays = np.zeros((2, 3), dtype=np.int64)

    np.testing.assert_array_equal(align_moveout(values, delays), values)
    np.testing.assert_array_equal(unalign_moveout(values, delays, 4), values)


@pytest.mark.parametrize(
    ("values", "delays"),
    [
        (np.zeros((4, 2)), np.zeros((3,), dtype=np.int64)),
        (np.zeros((4, 2)), np.array([0.0, 1.0])),
        (np.zeros((4, 2)), np.array([0, -1])),
        (np.zeros((4,)), np.zeros((), dtype=np.int64)),
    ],
)
def test_alignment_rejects_mismatched_or_negative_delays(values, delays):
    with pytest.raises(ValueError):
        align_moveout(values, delays)


def test_unalign_requires_the_matching_canvas_length():
    with pytest.raises(ValueError):
        unalign_moveout(np.zeros((5, 2)), np.array([0, 1]), 5)


def test_explicit_max_delay_places_trace_subsets_on_one_shared_canvas():
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    delays = np.array([0, 2, 1])
    shared = align_moveout(values, delays, max_delay=5)

    assert shared.shape == (9, 3)
    for cell, delay in enumerate(delays):
        shift = 5 - delay
        np.testing.assert_array_equal(shared[shift : shift + 4, cell], values[:, cell])
        assert np.count_nonzero(shared[:, cell]) == np.count_nonzero(values[:, cell])
    # A subset aligned alone with the same maximum lands on identical canvas rows.
    np.testing.assert_array_equal(
        align_moveout(values[:, 1:], delays[1:], max_delay=5), shared[:, 1:]
    )
    np.testing.assert_array_equal(unalign_moveout(shared, delays, 4, max_delay=5), values)


@pytest.mark.parametrize("max_delay", [1, -1, 2.0, True])
def test_explicit_max_delay_must_cover_every_delay(max_delay):
    with pytest.raises(ValueError):
        align_moveout(np.zeros((4, 2)), np.array([0, 2]), max_delay=max_delay)
