from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph
from seis_interp.training.trace_graph_episodes import (
    TraceGraphEpisodeGenerator,
    TraceGraphEpisodeLabelReader,
    read_trace_graph_training_labels,
)
from tests.fixtures.trace_graph_training import make_trace_graph_training_domains


def _preprocessing(domain, values):
    return fit_trace_graph_preprocessing(
        domain, values, position_scale_m=2.0, offset_scale_m=2.0, azimuth_min_offset_m=0.01
    )


@pytest.fixture(params=["function", "episode_reader"])
def read_labels(request):
    def read(domain, episode, ids, preprocessing, amplitudes):
        if request.param == "function":
            return read_trace_graph_training_labels(domain, episode, ids, preprocessing, amplitudes)
        return TraceGraphEpisodeLabelReader(domain, episode, preprocessing, amplitudes).read(ids)

    return read


def _generator(domain, *, seed=41, kind=None):
    probabilities = {kind: 1.0} if kind else {"random_trace": 0.5, "random_whole_ffid": 0.5}
    return TraceGraphEpisodeGenerator(
        domain, random_seed=seed, kind_probabilities=probabilities, missing_fractions=[0.5, 0.75]
    )


def _plan(domain, episode, query_ids):
    positions = np.flatnonzero(np.isin(domain.trace_ids, query_ids))
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m, domain.receiver_xy_m, azimuth_min_offset_m=0.01
    )
    queries = compute_trace_graph_geometry(
        domain.source_xy_m[positions], domain.receiver_xy_m[positions], azimuth_min_offset_m=0.01
    )
    return build_trace_graph_subgraph(
        queries,
        domain.trace_ids[positions],
        geometry,
        domain.trace_ids,
        episode.visible_mask,
        rounds=3,
        relation_scales_m=np.full((4, 2), 2.0),
        neighbors_per_relation=2,
    )


def test_episode_masks_and_query_order_ignore_global_rng_and_batch_size() -> None:
    domain, _, _ = make_trace_graph_training_domains()
    left, right = _generator(domain), _generator(domain)
    for expected_id in range(1, 8):
        one = left.next_episode()
        np.random.seed(expected_id)
        np.random.random(100)
        torch.manual_seed(expected_id)
        torch.rand(100)
        two = right.next_episode()
        assert one.episode_id == two.episode_id == expected_id
        assert (one.kind, one.missing_fraction) == (two.kind, two.missing_fraction)
        np.testing.assert_array_equal(one.visible_mask, two.visible_mask)
        np.testing.assert_array_equal(one.hidden_trace_ids, two.hidden_trace_ids)
        np.testing.assert_array_equal(
            np.concatenate(list(one.query_batches(1))), np.concatenate(list(two.query_batches(3)))
        )


def test_whole_ffid_hidden_members_never_enter_multihop_support() -> None:
    domain, _, _ = make_trace_graph_training_domains()
    episode = _generator(domain, kind="random_whole_ffid").next_episode()
    for ffid in np.unique(domain.ffids):
        flags = episode.visible_mask[domain.ffids == ffid]
        assert flags.all() or not flags.any()
    for batch in episode.query_batches(1):
        plan = _plan(domain, episode, batch)
        senders = plan.trace_ids[plan.edge_index[0]]
        assert len(senders) and not np.isin(senders, episode.hidden_trace_ids).any()
        assert not np.isin(plan.trace_ids[plan.observed_mask], episode.hidden_trace_ids).any()


def test_hidden_labels_outside_query_batch_do_not_change_inputs_or_graph(read_labels) -> None:
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain, kind="random_whole_ffid").next_episode()
    query_ids = next(episode.query_batches(1))
    plan = _plan(domain, episode, query_ids)
    before = MaskedTraceSource(domain, preprocessing, values).inputs(plan)
    changed_ids = np.setdiff1d(episode.hidden_trace_ids, query_ids)
    values[domain.array_rows[np.isin(domain.trace_ids, changed_ids)]] = np.nan
    after_plan = _plan(domain, episode, query_ids)
    after = MaskedTraceSource(domain, preprocessing, values).inputs(after_plan)
    for name in ("trace_ids", "edge_index", "edge_type", "edge_distances"):
        np.testing.assert_array_equal(getattr(plan, name), getattr(after_plan, name))
    for name in ("waveforms", "node_features", "edge_features", "coverage"):
        assert torch.equal(getattr(before, name), getattr(after, name))
    labels = read_labels(domain, episode, query_ids, preprocessing, values)
    assert torch.isfinite(labels).all()


@pytest.mark.parametrize("pool", ["all_train_traces", "mask_observed"])
def test_label_reader_reads_only_authorized_hidden_batch_and_fixed_time_scale(
    pool, read_labels
) -> None:
    domain, _, values = make_trace_graph_training_domains(pool=pool)
    domain = replace(domain, time_s=domain.time_s[1:6], time_samples=(1, 6))
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    batch = next(episode.query_batches(2))
    positions = np.array([np.flatnonzero(domain.trace_ids == value)[0] for value in batch])
    rows = domain.array_rows[positions]
    expected = values[rows, 1:6] / preprocessing.amplitude_scale
    safe_values = np.full_like(values, np.nan)
    safe_values[rows, 1:6] = values[rows, 1:6]
    labels = read_labels(domain, episode, batch, preprocessing, safe_values)
    np.testing.assert_allclose(labels.numpy(), expected, rtol=1e-6)
    with pytest.raises(ValueError, match="hidden set"):
        read_labels(domain, episode, np.array([999]), preprocessing, values)
    visible_id = domain.trace_ids[episode.visible_mask][:1]
    with pytest.raises(ValueError, match="hidden set"):
        read_labels(domain, episode, visible_id, preprocessing, values)


def test_each_hidden_trace_is_visited_once_and_final_small_batch_remains() -> None:
    domain, _, _ = make_trace_graph_training_domains()
    episode = _generator(domain, kind="random_trace").next_episode()
    batches = list(episode.query_batches(len(episode.hidden_trace_ids) - 1))
    assert [len(batch) for batch in batches] == [len(episode.hidden_trace_ids) - 1, 1]
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), episode.hidden_trace_ids)


@pytest.mark.parametrize("kind", ["random_trace", "random_whole_ffid"])
def test_too_small_training_pool_has_clear_error(kind) -> None:
    domain, _, _ = make_trace_graph_training_domains()
    if kind == "random_trace":
        domain = replace(
            domain,
            trace_ids=domain.trace_ids[:1],
            observed_mask=np.ones(1, bool),
            query_mask=np.zeros(1, bool),
            array_rows=domain.array_rows[:1],
        )
    else:
        domain = replace(domain, ffids=np.zeros_like(domain.ffids))
    with pytest.raises(ValueError, match="both visible and hidden"):
        _generator(domain, kind=kind)


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_nontraining_domain_cannot_be_used_for_episodes_or_labels(partition, read_labels) -> None:
    domain, validation, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    with pytest.raises(ValueError, match="training pool"):
        _generator(validation)
    invalid = replace(domain, inputs_lock={"partition": partition})
    with pytest.raises(ValueError, match="train partition"):
        _generator(invalid)
    with pytest.raises(ValueError, match="train partition"):
        read_labels(invalid, episode, episode.query_trace_ids, preprocessing, values)


@pytest.mark.parametrize(
    "probabilities", [{"bad": 1.0}, {"random_trace": 0.9}, {"random_trace": -1.0}]
)
def test_invalid_episode_probabilities_are_rejected(probabilities) -> None:
    domain, _, _ = make_trace_graph_training_domains()
    with pytest.raises(ValueError, match="kind_probabilities"):
        TraceGraphEpisodeGenerator(
            domain, random_seed=1, kind_probabilities=probabilities, missing_fractions=[0.5]
        )


@pytest.mark.parametrize("probability", [1.0 - 5e-9, 1.0 + 5e-9])
def test_episode_sampling_accepts_the_config_probability_sum_tolerance(probability) -> None:
    domain, _, _ = make_trace_graph_training_domains()
    generator = TraceGraphEpisodeGenerator(
        domain,
        random_seed=1,
        kind_probabilities={"random_trace": probability},
        missing_fractions=[0.5],
    )
    assert generator.next_episode().kind == "random_trace"


@pytest.mark.parametrize("kind", ["random_trace", "random_whole_ffid"])
def test_episode_reader_is_bitwise_equal_for_reversed_empty_and_final_small_batches(kind):
    domain, _, values = make_trace_graph_training_domains()
    values *= np.arange(1, len(values) + 1, dtype=np.float32)[:, None]
    domain = replace(domain, time_s=domain.time_s[1:6], time_samples=(1, 6))
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain, kind=kind).next_episode()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    reader = TraceGraphEpisodeLabelReader(domain, episode, preprocessing, values)
    batches = list(episode.query_batches(max(1, len(episode.hidden_trace_ids) - 1)))
    batches += [episode.query_trace_ids[::-1], np.empty(0, dtype=np.int64)]
    for batch in batches:
        expected = read_trace_graph_training_labels(domain, episode, batch, preprocessing, values)
        assert torch.equal(reader.read(batch), expected)
    after = np.random.get_state()
    assert after[0] == numpy_before[0] and after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(after[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_episode_reader_owns_row_time_and_episode_metadata():
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    batch = episode.query_trace_ids.copy()
    reader = TraceGraphEpisodeLabelReader(domain, episode, preprocessing, values)
    expected = reader.read(batch)
    domain.trace_ids[:] += 10000
    domain.array_rows[:] = len(values) + 1
    domain.time_s[:] += 1
    domain.observed_mask[:] = False
    domain.query_mask[:] = True
    domain.inputs_lock["partition"] = "test"
    episode.trace_ids[:] += 20000
    episode.hidden_trace_ids[:] += 30000
    episode.visible_mask[:] = True
    assert torch.equal(reader.read(batch), expected)


def test_episode_reader_replaces_authorization_for_a_new_episode():
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    previous_hidden = episode.query_trace_ids[:1].copy()
    next_hidden = domain.trace_ids[episode.visible_mask].copy()
    next_episode = replace(
        episode,
        episode_id=episode.episode_id + 1,
        visible_mask=~episode.visible_mask,
        hidden_trace_ids=next_hidden,
        query_trace_ids=next_hidden[::-1],
    )
    previous = TraceGraphEpisodeLabelReader(domain, episode, preprocessing, values)
    current = TraceGraphEpisodeLabelReader(domain, next_episode, preprocessing, values)
    assert torch.isfinite(previous.read(previous_hidden)).all()
    with pytest.raises(ValueError, match="hidden set"):
        current.read(previous_hidden)
    assert torch.equal(
        current.read(next_hidden),
        read_trace_graph_training_labels(domain, next_episode, next_hidden, preprocessing, values),
    )


def test_episode_reader_does_not_repeat_domain_membership_work_for_valid_batches(monkeypatch):
    from seis_interp.training import trace_graph_episodes as module

    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    reader = TraceGraphEpisodeLabelReader(domain, episode, preprocessing, values)

    def fail(*args, **kwargs):
        raise AssertionError("full-domain validation/membership must run only at construction")

    monkeypatch.setattr(module, "validate_trace_graph_training_domain", fail)
    monkeypatch.setattr(module.np, "isin", fail)
    for batch in episode.query_batches(1):
        assert torch.isfinite(reader.read(batch)).all()


@pytest.mark.parametrize(
    "query", [np.array([1.0]), np.array([True]), np.array([[1]]), np.array([1, 1])]
)
def test_episode_reader_rejects_invalid_query_before_inspecting_amplitudes(query):
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    reader = TraceGraphEpisodeLabelReader(
        domain, _generator(domain).next_episode(), preprocessing, amplitudes=object()
    )
    with pytest.raises(ValueError, match="unique integer vector"):
        reader.read(query)


@pytest.mark.parametrize("bad_episode", ["foreign_hidden", "visible_hidden", "visible_shape"])
def test_episode_reader_keeps_hidden_pool_and_visibility_barriers(bad_episode, read_labels):
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    batch = episode.query_trace_ids[:1]
    expected = "hidden throughout"
    if bad_episode == "foreign_hidden":
        batch = np.array([99999])
        episode = replace(episode, hidden_trace_ids=np.append(episode.hidden_trace_ids, batch))
        expected = "outside the authorized training pool"
    elif bad_episode == "visible_hidden":
        episode = replace(episode, visible_mask=np.ones_like(episode.visible_mask))
    else:
        episode = replace(episode, visible_mask=episode.visible_mask[:-1])
    with pytest.raises(ValueError, match=expected):
        read_labels(domain, episode, batch, preprocessing, values)


@pytest.mark.parametrize("problem", ["dtype", "row_bounds", "time_bounds", "nonfinite"])
def test_episode_reader_defers_amplitude_validation_until_read(problem):
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    batch = episode.query_trace_ids
    expected = "outside amplitudes"
    if problem == "dtype":
        values = values.astype(np.float64)
        expected = "float32 shape"
    elif problem == "row_bounds":
        values = values[:0]
    elif problem == "time_bounds":
        values = values[:, :-1]
    else:
        values[:] = np.nan
        expected = "finite shape"
    reader = TraceGraphEpisodeLabelReader(domain, episode, preprocessing, values)
    with pytest.raises(ValueError, match=expected):
        reader.read(batch)


def test_episode_reader_opens_memmap_once_after_query_validation(tmp_path, monkeypatch):
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    path = tmp_path / "amplitudes.npy"
    np.save(path, values)
    domain = replace(domain, amplitudes_path=path)
    episode = _generator(domain).next_episode()
    calls = []
    original = np.load

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    reader = TraceGraphEpisodeLabelReader(domain, episode, preprocessing)
    assert not calls
    with pytest.raises(ValueError, match="hidden set"):
        reader.read(domain.trace_ids[episode.visible_mask][:1])
    assert not calls
    for _ in range(2):
        assert torch.isfinite(reader.read(episode.query_trace_ids)).all()
    assert len(calls) == 1
    assert calls[0][1] == {"mmap_mode": "r", "allow_pickle": False}


@pytest.mark.parametrize("problem", ["episode_ids", "time_grid", "negative_rows"])
def test_episode_reader_rejects_invalid_binding_without_inspecting_amplitudes(problem):
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    if problem == "episode_ids":
        episode = replace(episode, trace_ids=episode.trace_ids[::-1])
        expected = "episode trace IDs must match"
    elif problem == "time_grid":
        domain = replace(domain, time_s=domain.time_s + 1.0)
        expected = "training time_s must match"
    else:
        domain = replace(domain, array_rows=np.full_like(domain.array_rows, -1))
        expected = "training pool must have amplitude array rows"
    with pytest.raises(ValueError, match=expected):
        TraceGraphEpisodeLabelReader(domain, episode, preprocessing, amplitudes=object())


def test_episode_reader_defers_missing_amplitude_path_error_until_read():
    domain, _, values = make_trace_graph_training_domains()
    preprocessing = _preprocessing(domain, values)
    episode = _generator(domain).next_episode()
    reader = TraceGraphEpisodeLabelReader(domain, episode, preprocessing)
    with pytest.raises(ValueError, match="training amplitudes or an amplitudes_path are required"):
        reader.read(episode.query_trace_ids)
