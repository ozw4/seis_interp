"""Opt-in execution switches of ReGSI training and prediction (all off by default).

Every switch keeps results bit-identical to the default path. On a CPU device
with several torch threads, compare under ``torch.use_deterministic_algorithms(True)``:
there the CPU backward of large index gathers accumulates with parallel atomic
additions, so even two default runs may differ in the last bits.

- ``neighbor_table``: rank each observed trace's nearest candidates once and take
  every episode's top-k from that table (exact fallback search when a list may be
  truncated); plans are assembled with vectorized code. Requires
  ``graph.neighbor_search: exact_index``.
- ``prefetch``: prepare the next batches in one worker process and copy them from
  page-locked memory without blocking; batch order and random streams are unchanged.
- ``deferred_checks``: trust the CPU-validated batch on the device, use the
  synchronization-free group operations of the model, and read losses, gradient
  norms and finiteness flags only at report intervals. A non-finite input, loss or
  gradient still fails with the same error text, at the next report interval.
- ``cpu_threads``: torch intra-op CPU threads while training and predicting
  (restored afterwards) on a CUDA device; device math is unaffected. Rejected on a
  CPU device: in this container's PyTorch build (MKL with GNU OpenMP) a run-time
  ``torch.set_num_threads`` turns off MKL's dynamic threading, after which CPU
  transcendental kernels (cos, sin, exp, tanh) intermittently crash with SIGSEGV.
  Choose the threads of a CPU run at launch with ``OMP_NUM_THREADS``.
- ``foreach_ema``: advance each EMA average with one ``torch._foreach_lerp_`` call.
- ``shared_prediction``: build each prediction batch once and run every EMA average
  on it; prediction support searches reuse the training neighbor table.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral

import torch

EXECUTION_SWITCHES = (
    "neighbor_table",
    "prefetch",
    "deferred_checks",
    "foreach_ema",
    "shared_prediction",
)


@dataclass(frozen=True)
class TraceGraphExecution:
    """Validated execution switches; the defaults reproduce the original path."""

    neighbor_table: bool = False
    prefetch: bool = False
    deferred_checks: bool = False
    cpu_threads: int | None = None
    foreach_ema: bool = False
    shared_prediction: bool = False


def validate_trace_graph_execution(
    options: Mapping[str, object] | TraceGraphExecution | None,
    *,
    neighbor_search: str | None = None,
    device: torch.device | str | None = None,
) -> TraceGraphExecution:
    """Accept a ``training.execution`` mapping; missing keys stay off.

    With the graph's ``neighbor_search``, a neighbor table is accepted only for
    ``exact_index`` (the table ranks the candidates of that index). With the run's
    ``device``, ``cpu_threads`` is rejected on a CPU device (see the module notes).
    """
    execution = _execution_switches(options)
    if (
        device is not None
        and execution.cpu_threads is not None
        and torch.device(device).type == "cpu"
    ):
        raise ValueError(
            "training.execution.cpu_threads is not supported on a CPU device; "
            "set OMP_NUM_THREADS at launch instead"
        )
    if (
        neighbor_search is not None
        and execution.neighbor_table
        and neighbor_search != "exact_index"
    ):
        raise ValueError(
            "training.execution.neighbor_table requires graph.neighbor_search=exact_index"
        )
    return execution


def _execution_switches(
    options: Mapping[str, object] | TraceGraphExecution | None,
) -> TraceGraphExecution:
    if options is None:
        return TraceGraphExecution()
    if isinstance(options, TraceGraphExecution):
        return options
    allowed = {*EXECUTION_SWITCHES, "cpu_threads"}
    if not isinstance(options, Mapping) or set(options) - allowed:
        raise ValueError(f"training.execution must be a mapping with keys from {sorted(allowed)}")
    for name in EXECUTION_SWITCHES:
        if not isinstance(options.get(name, False), bool):
            raise ValueError(f"training.execution.{name} must be boolean")
    threads = options.get("cpu_threads")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1
    ):
        raise ValueError("training.execution.cpu_threads must be a positive integer or null")
    return TraceGraphExecution(
        **{name: options.get(name, False) for name in EXECUTION_SWITCHES},
        cpu_threads=None if threads is None else int(threads),
    )


@contextmanager
def torch_cpu_threads(count: int | None) -> Iterator[None]:
    """Use ``count`` torch intra-op threads inside the block; ``None`` changes nothing."""
    if count is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
