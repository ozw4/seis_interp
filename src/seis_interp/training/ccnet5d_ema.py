"""Validation for optional final-weight averaging in observed-only CCNet training."""

import math
from numbers import Real


def validate_ccnet5d_ema_decay(value: object) -> float | None:
    """None preserves raw weights; EMA requires a finite decay strictly inside (0, 1)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("ema_decay must be a finite number in (0, 1) or null")
    decay = float(value)
    if not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("ema_decay must be a finite number in (0, 1) or null")
    return decay
