"""Deterministic artificial visibility and separate authorized training labels."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch

from seis_interp.data.trace_graph_domain import TraceGraphDomain
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing

EPISODE_KINDS = ("random_trace", "random_whole_ffid")


@dataclass(frozen=True)
class TraceGraphEpisode:
    """One complete hidden set; visibility remains fixed across its query batches."""

    episode_id: int
    kind: str
    missing_fraction: float
    trace_ids: np.ndarray
    visible_mask: np.ndarray
    hidden_trace_ids: np.ndarray
    query_trace_ids: np.ndarray

    def query_batches(self, query_batch_size: int) -> Iterator[np.ndarray]:
        """Visit every hidden trace once, retaining the final smaller batch."""
        size = _positive_integer(query_batch_size, "query_batch_size")
        for start in range(0, len(self.query_trace_ids), size):
            yield self.query_trace_ids[start : start + size].copy()


class TraceGraphEpisodeGenerator:
    """Sample masks independently of Torch/global RNG and query batch size.

    Mask counts use Python's ties-to-even ``round(unit_count * fraction)``,
    matching existing interpolation masks. Hidden IDs are sorted before a
    separate seeded shuffle determines their one-pass query order.
    """

    def __init__(
        self,
        domain: TraceGraphDomain,
        *,
        random_seed: int,
        kind_probabilities: Mapping[str, float],
        missing_fractions: Sequence[float],
    ) -> None:
        validate_trace_graph_training_domain(domain)
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, Integral)
            or random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if not kind_probabilities or set(kind_probabilities) - set(EPISODE_KINDS):
            raise ValueError(f"kind_probabilities must use {EPISODE_KINDS}")
        self._kinds = tuple(kind for kind in EPISODE_KINDS if kind in kind_probabilities)
        probabilities = [kind_probabilities[kind] for kind in self._kinds]
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
            for value in probabilities
        ) or not np.isclose(sum(probabilities), 1.0, rtol=0, atol=1e-8):
            raise ValueError("kind_probabilities must be nonnegative and sum to one")
        self._probabilities = np.asarray(probabilities, dtype=np.float64)
        self._probabilities /= self._probabilities.sum()
        self._fractions = tuple(missing_fractions)
        if not self._fractions or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 < value < 1
            for value in self._fractions
        ):
            raise ValueError("missing_fractions must contain fractions strictly between 0 and 1")
        for kind, probability in zip(self._kinds, self._probabilities, strict=True):
            if probability:
                units = (
                    len(domain.trace_ids)
                    if kind == "random_trace"
                    else len(np.unique(domain.ffids))
                )
                for fraction in self._fractions:
                    _hidden_count(units, fraction, kind)
        self._domain = domain
        self._rng = np.random.default_rng(int(random_seed))
        self._episode_id = 0

    def next_episode(self) -> TraceGraphEpisode:
        """Draw an entire episode before any minibatch is selected."""
        kind = self._kinds[int(self._rng.choice(len(self._kinds), p=self._probabilities))]
        fraction = float(self._fractions[int(self._rng.integers(len(self._fractions)))])
        units = (
            np.sort(self._domain.trace_ids)
            if kind == "random_trace"
            else np.unique(self._domain.ffids)
        )
        selected = self._rng.permutation(units)[: _hidden_count(len(units), fraction, kind)]
        hidden = np.isin(
            self._domain.trace_ids if kind == "random_trace" else self._domain.ffids, selected
        )
        hidden_ids = np.sort(self._domain.trace_ids[hidden])
        queries = self._rng.permutation(hidden_ids)
        self._episode_id += 1
        return TraceGraphEpisode(
            episode_id=self._episode_id,
            kind=kind,
            missing_fraction=fraction,
            trace_ids=self._domain.trace_ids.copy(),
            visible_mask=~hidden,
            hidden_trace_ids=hidden_ids,
            query_trace_ids=queries,
        )


def validate_trace_graph_training_domain(domain: TraceGraphDomain) -> None:
    """Require the complete authorized O0, never a validation/test case."""
    if domain.pool not in ("all_train_traces", "mask_observed"):
        raise ValueError("training labels require an explicit authorized training pool")
    if domain.inputs_lock.get("partition") != "train":
        raise ValueError("training labels require a train partition domain")
    if not len(domain.trace_ids) or not np.all(domain.observed_mask) or np.any(domain.query_mask):
        raise ValueError("episodes require the complete unmasked training pool")
    if domain.array_rows is None or np.any(domain.array_rows < 0):
        raise ValueError("training pool must have amplitude array rows")


def read_trace_graph_training_labels(
    domain: TraceGraphDomain,
    episode: TraceGraphEpisode,
    query_trace_ids: np.ndarray,
    preprocessing: TraceGraphPreprocessing,
    amplitudes: np.ndarray | None = None,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Read only H intersected with this query batch, divided by fixed a_train."""
    validate_trace_graph_training_domain(domain)
    if not np.array_equal(episode.trace_ids, domain.trace_ids):
        raise ValueError("episode trace IDs must match the training domain")
    if not np.array_equal(domain.time_s, preprocessing.time_s):
        raise ValueError("training time_s must match the fixed preprocessing time grid")
    ids = np.asarray(query_trace_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("query_trace_ids must be a unique integer vector")
    if not np.all(np.isin(ids, episode.hidden_trace_ids)):
        raise ValueError("training label queries must belong to the episode hidden set")
    positions = {int(trace_id): index for index, trace_id in enumerate(domain.trace_ids)}
    try:
        indices = np.array([positions[int(trace_id)] for trace_id in ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(
            "training label queries are outside the authorized training pool"
        ) from error
    if episode.visible_mask.shape != domain.observed_mask.shape or np.any(
        episode.visible_mask[indices]
    ):
        raise ValueError("training label queries must be hidden throughout the episode")
    if amplitudes is None:
        if domain.amplitudes_path is None:
            raise ValueError("training amplitudes or an amplitudes_path are required")
        amplitudes = np.load(domain.amplitudes_path, mmap_mode="r", allow_pickle=False)
    if amplitudes.ndim != 2 or amplitudes.dtype != np.float32:
        raise ValueError("amplitudes must have float32 shape [rows, time]")
    assert domain.array_rows is not None
    rows = domain.array_rows[indices]
    start, stop = domain.time_samples
    if np.any(rows >= amplitudes.shape[0]) or stop > amplitudes.shape[1]:
        raise ValueError("training rows or time selection are outside amplitudes")
    values = np.array(amplitudes[rows, start:stop], dtype=np.float64, copy=True)
    if values.shape != (len(ids), len(domain.time_s)) or not np.all(np.isfinite(values)):
        raise ValueError("training labels must have finite shape [Q, T]")
    return torch.from_numpy((values / preprocessing.amplitude_scale).astype(np.float32)).to(device)


def _hidden_count(unit_count: int, fraction: float, kind: str) -> int:
    count = round(unit_count * fraction)
    if not 0 < count < unit_count:
        raise ValueError(
            f"{kind} fraction {fraction} must produce both visible and hidden units "
            f"among {unit_count} units"
        )
    return count


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
