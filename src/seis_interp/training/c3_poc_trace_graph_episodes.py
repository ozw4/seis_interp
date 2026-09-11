"""Random whole-trace pseudo-targets and separated O-only graph label access."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.training.trace_graph_episodes import TraceGraphEpisode


class PocTraceGraphEpisodeGenerator:
    """Redraw a fixed-fraction uniform subset of O using only a local seed."""

    def __init__(
        self,
        training: C3PocTraceGraphTrainingData,
        *,
        random_seed: int,
        missing_fraction: float,
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
        self._trace_ids = training.domain.trace_ids.copy()
        self._hidden_count = round(len(self._trace_ids) * missing_fraction)
        if not 0 < self._hidden_count < len(self._trace_ids):
            raise ValueError("missing_fraction must leave at least one query and one context")
        self._fraction = float(missing_fraction)
        self._rng = np.random.default_rng(int(random_seed))
        self._episode_id = 0

    def next_episode(self) -> TraceGraphEpisode:
        """Keep this visibility fixed while visiting all hidden query batches."""
        hidden = np.sort(self._rng.choice(self._trace_ids, size=self._hidden_count, replace=False))
        self._episode_id += 1
        return TraceGraphEpisode(
            episode_id=self._episode_id,
            kind="random_trace",
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
    if episode.kind != "random_trace" or not np.array_equal(episode.trace_ids, ids):
        raise ValueError("PoC episodes must be random_trace over exactly O")
    if (
        visible.dtype != np.bool_
        or visible.shape != ids.shape
        or not visible.any()
        or visible.all()
        or not np.array_equal(np.sort(ids[~visible]), np.sort(episode.hidden_trace_ids))
        or not np.array_equal(np.sort(episode.query_trace_ids), np.sort(episode.hidden_trace_ids))
    ):
        raise ValueError("episode must partition O into context and hidden query traces")
