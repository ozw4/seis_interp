"""Post-update parameter averaging for final trace-graph inference."""

import math
from collections.abc import Sequence
from numbers import Real

import torch


def validate_trace_graph_ema_decay(value: object) -> float | None:
    """None disables EMA; enabled decay must be finite and strictly inside (0, 1)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("ema_decay must be a finite number in (0, 1) or null")
    decay = float(value)
    if not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("ema_decay must be a finite number in (0, 1) or null")
    return decay


class TraceGraphEMA:
    """Average parameters on their device; copy buffers exactly, without changing the model.

    Initialize from the first successful optimizer update. Subsequent updates use
    decay * average + (1 - decay) * parameter. Call only after an optimizer step
    that was not skipped by AMP. No model construction or random draws occur.
    ``snapshot_updates`` keeps a CPU copy of the average after each listed update
    in ``snapshots`` (update count -> state dict); it never changes the average.
    """

    def __init__(self, decay: float, snapshot_updates: tuple[int, ...] = ()) -> None:
        self.decay = validate_trace_graph_ema_decay(decay)
        if self.decay is None:
            raise ValueError("EMA requires an enabled ema_decay")
        self.updates = 0
        self._state: dict[str, torch.Tensor] = {}
        self._snapshot_updates = frozenset(snapshot_updates)
        self.snapshots: dict[int, dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self._update_from_state(model.state_dict(), model, foreach=False)

    def _update_from_state(
        self, state: dict[str, torch.Tensor], model: torch.nn.Module, *, foreach: bool
    ) -> None:
        if not self.updates:
            self._state = {key: value.detach().clone() for key, value in state.items()}
        else:
            parameter_names = dict(model.named_parameters())
            if foreach:
                keys = [key for key in state if key in parameter_names]
                # One fused call per average; the same elementwise lerp as below.
                torch._foreach_lerp_(
                    [self._state[key] for key in keys], [state[key] for key in keys], 1 - self.decay
                )
            for key, value in state.items():
                if key not in parameter_names:
                    # Fixed Fourier frequencies must remain bit-exact for strict loading.
                    self._state[key].copy_(value)
                elif not foreach:
                    self._state[key].lerp_(value, 1 - self.decay)
        self.updates += 1
        if self.updates in self._snapshot_updates:
            self.snapshots[self.updates] = self.state_dict()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return an independent CPU snapshot suitable for the normal checkpoint writer."""
        if not self.updates:
            raise ValueError("EMA has no successful optimizer updates")
        return {key: value.detach().cpu().clone() for key, value in self._state.items()}


@torch.no_grad()
def update_trace_graph_averages(
    averages: Sequence[TraceGraphEMA], model: torch.nn.Module, *, foreach: bool = False
) -> None:
    """Advance every average after one optimizer update, in order.

    ``foreach=True`` reads the model state once and moves each average with one
    ``torch._foreach_lerp_`` call; the arithmetic is the same elementwise lerp.
    """
    if not foreach:
        for average in averages:
            average.update(model)
        return
    state = model.state_dict()
    for average in averages:
        average._update_from_state(state, model, foreach=True)
