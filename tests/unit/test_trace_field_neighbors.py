"""A neighbourhood follows the declared coordinates and never contains the query."""

import numpy as np
import pytest

from seis_interp.processing.trace_field_neighbors import nearest_observed_positions


def _line(count):
    """A one-dimensional lattice in the first coordinate makes the answer checkable."""
    coordinates = np.zeros((count, 4))
    coordinates[:, 0] = np.arange(count) / (count - 1)
    return coordinates


def test_the_nearest_observed_traces_are_returned_in_distance_order():
    coordinates = _line(10)
    observed = np.array([0, 2, 4, 6, 8])
    positions = nearest_observed_positions(coordinates, observed, np.array([1, 7]), 3)
    assert observed[positions[0]].tolist() == [0, 2, 4]
    assert observed[positions[1]].tolist() == [8, 6, 4]


def test_an_observed_query_never_reads_itself():
    coordinates = _line(10)
    observed = np.array([0, 2, 4, 6, 8])
    positions = nearest_observed_positions(coordinates, observed, observed, 4)
    for row, neighbours in zip(observed, observed[positions], strict=True):
        assert row not in neighbours


def test_equidistant_neighbours_break_on_the_observed_position():
    coordinates = np.zeros((5, 4))
    coordinates[:, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]
    observed = np.array([0, 2, 4])
    # The query at 1.0 is exactly one away from observed rows 0 and 2.
    positions = nearest_observed_positions(coordinates, observed, np.array([1]), 2)
    assert positions[0].tolist() == [0, 1]
    assert observed[positions[0]].tolist() == [0, 2]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64])
def test_chunking_never_changes_the_result(chunk_size):
    coordinates = np.random.default_rng(5).uniform(0, 1, size=(40, 4))
    observed = np.arange(0, 40, 3)
    expected = nearest_observed_positions(coordinates, observed, np.arange(40), 4)
    actual = nearest_observed_positions(
        coordinates, observed, np.arange(40), 4, chunk_size=chunk_size
    )
    np.testing.assert_array_equal(actual, expected)


def test_the_count_must_leave_one_observed_trace_after_excluding_the_query():
    coordinates = _line(6)
    observed = np.array([0, 2, 4])
    nearest_observed_positions(coordinates, observed, observed, 2)
    with pytest.raises(ValueError, match="excluding the query itself"):
        nearest_observed_positions(coordinates, observed, observed, 3)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"coordinates": np.zeros((4, 3))}, "finite"),
        ({"coordinates": np.full((4, 4), np.nan)}, "finite"),
        ({"observed_rows": np.array([0, 0])}, "distinct"),
        ({"observed_rows": np.array([0, 99])}, "observed_rows"),
        ({"query_rows": np.array([-1])}, "query_rows"),
        ({"count": 0}, "count"),
        ({"count": True}, "count"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    settings = {
        "coordinates": _line(6),
        "observed_rows": np.array([0, 2, 4]),
        "query_rows": np.array([1, 3]),
        "count": 2,
    }
    settings.update(kwargs)
    chunk = settings.pop("chunk_size", None)
    with pytest.raises(ValueError, match=message):
        nearest_observed_positions(
            **settings, **({"chunk_size": chunk} if chunk is not None else {})
        )
