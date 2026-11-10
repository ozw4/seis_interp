"""The declared training view of each data-reading method matches its test view.

A whole-shot interpolation mask removes every trace of a shot, so a test query has no
observed trace from its own source gather. Both fitted baselines re-mask the observed
set during training, and by default they re-mask individual traces, which leaves a
training query the in-gather context its test query can never have. These tests fix the
declared alternative and, just as importantly, fix that the default is unchanged: the
existing sealed runs reproduce only if the default draws the same numbers as before.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from seis_interp.processing.c3_region_search import resolve_c3_candidate
from seis_interp.training.c3_poc_trace_graph_episodes import PocTraceGraphEpisodeGenerator
from seis_interp.training.ccnet5d_observed_patches import (
    _draw_pseudo_target_mask,
    _eligible_spatial_starts,
)

GATHERS, PER_GATHER = 12, 8


def _training(gathers=GATHERS, per_gather=PER_GATHER):
    """A training pool of whole gathers, shaped like the observed half of a shot mask."""
    ffids = np.repeat(np.arange(gathers), per_gather)
    return SimpleNamespace(
        domain=SimpleNamespace(trace_ids=np.arange(gathers * per_gather), ffids=ffids)
    )


def test_the_trace_episode_still_draws_exactly_what_it_drew_before():
    """The default must not move: sealed runs replay this stream."""
    training = _training()
    generator = PocTraceGraphEpisodeGenerator(training, random_seed=7301, missing_fraction=0.25)
    episode = generator.next_episode()

    ids = training.domain.trace_ids
    expected = np.random.default_rng(7301)
    hidden = np.sort(expected.choice(ids, size=round(len(ids) * 0.25), replace=False))
    assert np.array_equal(episode.hidden_trace_ids, hidden)
    assert np.array_equal(episode.query_trace_ids, expected.permutation(hidden))
    assert episode.kind == "random_trace"


def test_the_trace_episode_leaves_a_query_most_of_its_own_gather():
    """This is the mismatch: the default training view is not the whole-shot test view."""
    training = _training()
    episode = PocTraceGraphEpisodeGenerator(
        training, random_seed=7301, missing_fraction=0.25
    ).next_episode()
    ffids = training.domain.ffids
    visible_gathers = ffids[episode.visible_mask]
    held = ffids[~episode.visible_mask]
    assert np.isin(held, visible_gathers).all()


def test_the_whole_gather_episode_hides_entire_gathers():
    training = _training()
    episode = PocTraceGraphEpisodeGenerator(
        training, random_seed=7301, missing_fraction=0.25, kind="random_whole_ffid"
    ).next_episode()
    ffids = training.domain.ffids
    hidden_gathers = np.unique(ffids[~episode.visible_mask])
    assert len(hidden_gathers) == round(GATHERS * 0.25)
    assert len(episode.hidden_trace_ids) == len(hidden_gathers) * PER_GATHER
    # No query can read its own gather, exactly as at test time under a whole-shot mask.
    assert not np.isin(hidden_gathers, ffids[episode.visible_mask]).any()
    assert episode.kind == "random_whole_ffid"
    assert np.array_equal(np.sort(episode.query_trace_ids), np.sort(episode.hidden_trace_ids))


def test_the_whole_gather_episode_redraws_a_different_gather_set():
    generator = PocTraceGraphEpisodeGenerator(
        _training(), random_seed=7301, missing_fraction=0.25, kind="random_whole_ffid"
    )
    first, second = generator.next_episode(), generator.next_episode()
    assert first.episode_id == 1 and second.episode_id == 2
    assert not np.array_equal(first.hidden_trace_ids, second.hidden_trace_ids)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"kind": "whole_shot"}, "kind must be one of"),
        ({"kind": "random_whole_ffid", "missing_fraction": 0.01}, "at least one query"),
    ],
)
def test_the_episode_generator_rejects_an_undeclarable_view(overrides, message):
    settings = {"random_seed": 7301, "missing_fraction": 0.25, **overrides}
    with pytest.raises(ValueError, match=message):
        PocTraceGraphEpisodeGenerator(_training(), **settings)


def test_the_whole_gather_episode_needs_one_ffid_for_every_trace():
    broken = SimpleNamespace(domain=SimpleNamespace(trace_ids=np.arange(8), ffids=np.arange(4)))
    with pytest.raises(ValueError, match="one ffid per training trace"):
        PocTraceGraphEpisodeGenerator(broken, random_seed=7301, missing_fraction=0.25)


def _patch_mask(stations=8, receivers=(2, 6), observed_stations=None):
    """One patch's availability: whole source stations of receiver cells."""
    mask = np.zeros((2, stations // 2, *receivers), dtype=bool)
    flat = mask.reshape(-1, *receivers)
    for index in range(stations) if observed_stations is None else observed_stations:
        flat[index] = True
    return mask


def test_the_trace_unit_still_draws_exactly_what_it_drew_before():
    """The default must not move: sealed CCNet5D runs replay this stream."""
    available = _patch_mask()
    hidden = _draw_pseudo_target_mask(available, fraction=0.25, generator=np.random.default_rng(11))
    expected = np.zeros(available.shape, dtype=bool)
    positions = np.flatnonzero(available)
    selected = np.random.default_rng(11).choice(
        positions, size=round(len(positions) * 0.25), replace=False
    )
    expected.reshape(-1)[selected] = True
    assert np.array_equal(hidden, expected)


def test_the_trace_unit_leaves_a_pseudo_target_most_of_its_own_station():
    available = _patch_mask()
    hidden = _draw_pseudo_target_mask(available, fraction=0.25, generator=np.random.default_rng(11))
    visible_per_station = (available & ~hidden).sum(axis=(2, 3))
    lines, points = np.nonzero(hidden.any(axis=(2, 3)))
    assert (visible_per_station[lines, points] > 0).all()


def test_the_station_unit_hides_whole_stations_and_keeps_the_cell_budget():
    available = _patch_mask()
    hidden = _draw_pseudo_target_mask(
        available, fraction=0.25, generator=np.random.default_rng(11), unit="source_station"
    )
    per_station_hidden = hidden.sum(axis=(2, 3))
    cells = available.shape[2] * available.shape[3]
    assert set(np.unique(per_station_hidden)) == {0, cells}
    assert per_station_hidden.astype(bool).sum() == round(8 * 0.25)
    # The hidden cell count still matches the declared fraction of the patch.
    assert hidden.sum() == round(available.sum() * 0.25)
    # No pseudo-target can read its own station, exactly as at test time.
    visible_per_station = (available & ~hidden).sum(axis=(2, 3))
    lines, points = np.nonzero(per_station_hidden)
    assert (visible_per_station[lines, points] == 0).all()
    assert not (hidden & ~available).any()


def test_the_station_unit_only_hides_available_cells():
    available = _patch_mask(observed_stations=[0, 1, 2, 3, 5, 7])
    available.reshape(-1, 2, 6)[1, 0] = False
    hidden = _draw_pseudo_target_mask(
        available, fraction=0.5, generator=np.random.default_rng(3), unit="source_station"
    )
    assert not (hidden & ~available).any()
    assert hidden.any() and (available & ~hidden).any()


def test_the_station_unit_rejects_a_patch_without_a_visible_station():
    single = _patch_mask(observed_stations=[0])
    with pytest.raises(ValueError, match="at least one pseudo-target"):
        _draw_pseudo_target_mask(
            single, fraction=0.5, generator=np.random.default_rng(3), unit="source_station"
        )


def test_an_unknown_mask_unit_is_rejected():
    with pytest.raises(ValueError, match="inner_mask_unit must be"):
        _draw_pseudo_target_mask(
            _patch_mask(), fraction=0.25, generator=np.random.default_rng(3), unit="shot"
        )


def test_eligible_starts_count_stations_when_the_unit_is_a_station():
    observed = np.zeros((2, 4, 1, 4), dtype=bool)
    observed[:, :2] = True
    patch = (2, 2, 1, 4)
    traces = _eligible_spatial_starts(observed, patch_shape=patch, inner_mask_fraction=0.25)
    stations = _eligible_spatial_starts(
        observed, patch_shape=patch, inner_mask_fraction=0.25, inner_mask_unit="source_station"
    )
    # A placement whose four stations round to one hidden station is eligible for both;
    # one with a single observed station can still hide a cell but never a whole station.
    assert (0, 0, 0, 0) in traces and (0, 0, 0, 0) in stations
    assert (0, 1, 0, 0) in traces and (0, 1, 0, 0) not in stations


def test_the_whole_gather_fraction_is_not_rounded_to_a_query_batch():
    """The gather count, not the batch fill, is what the declared fraction names."""
    config = {
        "method_id": "regsi_real",
        "training": {"inner_mask_fraction": 0.25, "episode_kind": "random_whole_ffid"},
    }
    assert resolve_c3_candidate(config, 24575, 64)["training"]["inner_mask_fraction"] == 0.25
    trace_view = {**config, "training": {"inner_mask_fraction": 0.25}}
    rounded = resolve_c3_candidate(trace_view, 24575, 64)["training"]["inner_mask_fraction"]
    assert rounded == 64 * math.floor(0.25 * 24575 / 64) / 24575 != 0.25


def test_a_mislabelled_whole_gather_episode_is_rejected():
    """The label and the hidden set must agree, or a consumer reads the wrong contract."""
    from dataclasses import replace

    from seis_interp.training.c3_poc_trace_graph_episodes import _validate_episode

    training = _training()
    episode = PocTraceGraphEpisodeGenerator(
        training, random_seed=7301, missing_fraction=0.25, kind="random_whole_ffid"
    ).next_episode()
    _validate_episode(training, episode)

    scattered = PocTraceGraphEpisodeGenerator(
        training, random_seed=7301, missing_fraction=0.25
    ).next_episode()
    with pytest.raises(ValueError, match="hide every trace of each hidden gather"):
        _validate_episode(training, replace(scattered, kind="random_whole_ffid"))
