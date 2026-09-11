"""Fanout counts, stable subsequences and caller-owned randomness."""

from copy import deepcopy
from dataclasses import fields, replace

import numpy as np
import pytest

from seis_interp.processing.trace_graph_edge_sampling import sample_trace_graph_relation_fanout
from seis_interp.processing.trace_graph_neighbors import TraceGraphNeighbors


def _neighbors():
    # Destination input order is deliberately not numeric order.
    destinations, relations, senders, distances = [], [], [], []
    for destination in (90, 10):
        for relation, count in enumerate((8, 1, 0, 5)):
            destinations.extend([destination] * count)
            relations.extend([relation] * count)
            senders.extend(range(count))
            distances.extend(np.arange(count) / 10)
    return TraceGraphNeighbors(
        np.array(senders, dtype=np.int64),
        np.array(destinations, dtype=np.int64),
        np.array(relations, dtype=np.int64),
        np.array(distances, dtype=np.float64),
    )


def _records(neighbors):
    return list(zip(*(getattr(neighbors, field.name) for field in fields(neighbors)), strict=True))


def test_fanout_is_stable_subsequence_with_exact_group_counts_and_no_duplicates():
    original = _neighbors()
    snapshot = deepcopy(original)
    sampled = sample_trace_graph_relation_fanout(
        original, fanout_per_relation=3, rng=np.random.default_rng(42)
    )
    records = _records(sampled)
    assert len(records) == len(set(records)) == 14 < len(_records(original))
    input_positions = [_records(original).index(record) for record in records]
    assert input_positions == sorted(input_positions)
    for destination in (90, 10):
        for relation, count in enumerate((8, 1, 0, 5)):
            group = (sampled.destination_ids == destination) & (sampled.edge_type == relation)
            assert group.sum() == min(count, 3)
    for field in fields(original):
        np.testing.assert_array_equal(getattr(original, field.name), getattr(snapshot, field.name))
        assert getattr(sampled, field.name).dtype == getattr(original, field.name).dtype
        assert not np.shares_memory(getattr(sampled, field.name), getattr(original, field.name))


def test_seed_reproducibility_and_state_advance():
    def sample(seed):
        return sample_trace_graph_relation_fanout(
            _neighbors(), fanout_per_relation=3, rng=np.random.default_rng(seed)
        )

    assert _records(sample(42)) == _records(sample(42))
    assert _records(sample(42)) != _records(sample(43))
    rng = np.random.default_rng(42)
    state = deepcopy(rng.bit_generator.state)
    sample_trace_graph_relation_fanout(_neighbors(), fanout_per_relation=3, rng=rng)
    assert state != rng.bit_generator.state


@pytest.mark.parametrize("empty", [False, True])
def test_no_sampling_consumes_no_rng_and_preserves_all_edges(empty):
    original = _neighbors()
    if empty:
        original = TraceGraphNeighbors(*(getattr(original, f.name)[:0] for f in fields(original)))
    rng = np.random.default_rng(42)
    state = deepcopy(rng.bit_generator.state)
    sampled = sample_trace_graph_relation_fanout(original, fanout_per_relation=8, rng=rng)
    assert _records(sampled) == _records(original)
    assert rng.bit_generator.state == state
    assert sampled.sender_ids.dtype == np.int64
    assert sampled.distances.dtype == np.float64


@pytest.mark.parametrize("fanout", [True, np.bool_(False), 0, -1, 1.5, {}, None])
def test_invalid_fanout_is_rejected(fanout):
    with pytest.raises(ValueError, match="fanout_per_relation"):
        sample_trace_graph_relation_fanout(
            _neighbors(), fanout_per_relation=fanout, rng=np.random.default_rng(42)
        )


@pytest.mark.parametrize("rng", [None, 42, np.random.RandomState(42)])
def test_generator_is_required(rng):
    with pytest.raises(ValueError, match="numpy.random.Generator"):
        sample_trace_graph_relation_fanout(_neighbors(), fanout_per_relation=3, rng=rng)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sender_ids", np.ones(28, dtype=np.int32)),
        ("destination_ids", np.ones((28, 1), dtype=np.int64)),
        ("edge_type", np.ones(27, dtype=np.int64)),
        ("distances", np.ones(28, dtype=np.float32)),
        ("sender_ids", [1] * 28),
        ("edge_type", np.full(28, 4, dtype=np.int64)),
        ("edge_type", np.full(28, -1, dtype=np.int64)),
    ],
)
def test_invalid_arrays_are_rejected(field, value):
    with pytest.raises(ValueError):
        sample_trace_graph_relation_fanout(
            replace(_neighbors(), **{field: value}),
            fanout_per_relation=3,
            rng=np.random.default_rng(42),
        )
