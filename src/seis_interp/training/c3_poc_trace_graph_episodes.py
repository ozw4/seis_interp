"""Random trace or whole-gather pseudo-targets and separated O-only graph label access."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.training.trace_graph_episodes import EPISODE_KINDS, TraceGraphEpisode


class PocTraceGraphEpisodeGenerator:
    """Redraw a fixed-fraction subset of O using only a local seed.

    ``kind`` chooses the hidden unit. ``random_trace`` hides individual traces, so a
    hidden query keeps most of its own source gather as visible context.
    ``random_whole_ffid`` hides whole gathers, which is what a whole-shot
    interpolation mask does to a test query: the query has no visible trace from its
    own gather. Declare the kind that matches the mask the fit will be tested under,
    or the fit learns to read context the test never carries.
    """

    def __init__(
        self,
        training: C3PocTraceGraphTrainingData,
        *,
        random_seed: int,
        missing_fraction: float,
        kind: str = "random_trace",
    ) -> None:
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, Integral)
            or random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if (
            isinstance(missing_fraction, bool)
            or not isinstance(missing_fraction, Real)
            or not np.isfinite(missing_fraction)
            or not 0 < missing_fraction < 1
        ):
            raise ValueError("missing_fraction must be strictly between 0 and 1")
        if kind not in EPISODE_KINDS:
            raise ValueError(f"kind must be one of {EPISODE_KINDS}")
        self._kind = kind
        self._trace_ids = training.domain.trace_ids.copy()
        self._ffids = np.asarray(training.domain.ffids).copy()
        if self._ffids.shape != self._trace_ids.shape:
            raise ValueError("episodes require one ffid per training trace")
        units = self._trace_ids if kind == "random_trace" else np.unique(self._ffids)
        self._hidden_count = round(len(units) * missing_fraction)
        if not 0 < self._hidden_count < len(units):
            raise ValueError("missing_fraction must leave at least one query and one context")
        self._fraction = float(missing_fraction)
        self._rng = np.random.default_rng(int(random_seed))
        self._episode_id = 0

    def state_dict(self):
        return {"rng": self._rng.bit_generator.state, "episode_id": self._episode_id}

    def load_state_dict(self, state):
        self._rng.bit_generator.state = state["rng"]
        self._episode_id = state["episode_id"]

    def next_episode(self) -> TraceGraphEpisode:
        """Keep this visibility fixed while visiting all hidden query batches."""
        if self._kind == "random_trace":
            hidden = np.sort(
                self._rng.choice(self._trace_ids, size=self._hidden_count, replace=False)
            )
        else:
            gathers = self._rng.choice(
                np.unique(self._ffids), size=self._hidden_count, replace=False
            )
            hidden = np.sort(self._trace_ids[np.isin(self._ffids, gathers)])
        self._episode_id += 1
        return TraceGraphEpisode(
            episode_id=self._episode_id,
            kind=self._kind,
            missing_fraction=self._fraction,
            trace_ids=self._trace_ids.copy(),
            visible_mask=~np.isin(self._trace_ids, hidden),
            hidden_trace_ids=hidden,
            query_trace_ids=self._rng.permutation(hidden),
        )


def build_poc_trace_graph_context_source(
    training: C3PocTraceGraphTrainingData,
    episode: TraceGraphEpisode,
    *,
    device: str = "cpu",
) -> MaskedTraceSource:
    """Materialize O minus H only, excluding hidden waveforms even from storage."""
    _validate_episode(training, episode)
    return training.source(episode.visible_mask, device=device)


class PocTraceGraphEpisodeLabelReader:
    """Read only H in physical units and divide by the shared external s_O."""

    def __init__(
        self,
        training: C3PocTraceGraphTrainingData,
        episode: TraceGraphEpisode,
        *,
        device: str = "cpu",
    ) -> None:
        _validate_episode(training, episode)
        self._source = training.source(~episode.visible_mask)
        self._device = device

    def read(self, query_trace_ids: np.ndarray) -> torch.Tensor:
        """Return float32 [query, time] labels with float64 amplitude division."""
        values = self._source.read_observed_rows(query_trace_ids).astype(np.float64)
        return torch.from_numpy(
            (values / self._source.preprocessing.amplitude_scale).astype(np.float32)
        ).to(self._device)


def _validate_episode(training: C3PocTraceGraphTrainingData, episode: TraceGraphEpisode) -> None:
    ids = training.domain.trace_ids
    visible = episode.visible_mask
    if episode.kind not in EPISODE_KINDS or not np.array_equal(episode.trace_ids, ids):
        raise ValueError(f"PoC episodes must use {EPISODE_KINDS} over exactly O")
    if (
        visible.dtype != np.bool_
        or visible.shape != ids.shape
        or not visible.any()
        or visible.all()
        or not np.array_equal(np.sort(ids[~visible]), np.sort(episode.hidden_trace_ids))
        or not np.array_equal(np.sort(episode.query_trace_ids), np.sort(episode.hidden_trace_ids))
    ):
        raise ValueError("episode must partition O into context and hidden query traces")
    ffids = np.asarray(training.domain.ffids)
    if episode.kind == "random_whole_ffid" and np.isin(ffids[visible], ffids[~visible]).any():
        raise ValueError("a whole-gather episode must hide every trace of each hidden gather")
