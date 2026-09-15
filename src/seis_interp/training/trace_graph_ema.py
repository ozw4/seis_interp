"""Post-update parameter averaging for final trace-graph inference."""

import math
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
    """

    def __init__(self, decay: float) -> None:
        self.decay = validate_trace_graph_ema_decay(decay)
        if self.decay is None:
            raise ValueError("EMA requires an enabled ema_decay")
        self.updates = 0
        self._state: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        if not self.updates:
            self._state = {key: value.detach().clone() for key, value in state.items()}
        else:
            parameter_names = dict(model.named_parameters())
            for key, value in state.items():
                if key in parameter_names:
                    self._state[key].lerp_(value, 1 - self.decay)
                else:
                    # Fixed Fourier frequencies must remain bit-exact for strict loading.
                    self._state[key].copy_(value)
        self.updates += 1

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return an independent CPU snapshot suitable for the normal checkpoint writer."""
        if not self.updates:
            raise ValueError("EMA has no successful optimizer updates")
        return {key: value.detach().cpu().clone() for key, value in self._state.items()}
