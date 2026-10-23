"""Common geometry-only moveout canvas for compact FORGE model inputs.

Every method may run on the same extended time canvas: trace ``i`` is placed
``max_delay - rint(offset_i / velocity / dt)`` samples later, predicted there,
and returned to its original samples by the exact inverse shift. Delays depend
only on the method's geometry table; waveforms and observation roles are never
consulted.
"""

import math
from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np
import pandas as pd

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.processing.linear_moveout import (
    align_moveout,
    moveout_delay_samples,
    unalign_moveout,
)


@dataclass(frozen=True)
class MoveoutCanvas:
    """Per-cell integer delays and the shared canvas length."""

    cell_ids: np.ndarray
    delays: np.ndarray
    max_delay_samples: int
    time_sample_count: int

    @property
    def length(self) -> int:
        return self.time_sample_count + self.max_delay_samples

    def delays_for(self, cells: np.ndarray) -> np.ndarray:
        positions = pd.Index(self.cell_ids).get_indexer(np.asarray(cells))
        if np.any(positions < 0):
            raise ValueError("cells outside the aligned geometry")
        return self.delays[positions]


def validate_alignment(alignment: dict) -> None:
    """Accept a positive velocity and an optional shared maximum delay."""
    if not isinstance(alignment, dict) or "velocity_m_per_s" not in alignment:
        raise ValueError("alignment requires velocity_m_per_s")
    if set(alignment) - {"velocity_m_per_s", "max_delay_samples"}:
        raise ValueError("alignment accepts only velocity_m_per_s and max_delay_samples")
    velocity = alignment["velocity_m_per_s"]
    if (
        isinstance(velocity, bool)
        or not isinstance(velocity, Real)
        or not math.isfinite(velocity)
        or velocity <= 0
    ):
        raise ValueError("alignment velocity must be positive and finite")
    maximum = alignment.get("max_delay_samples")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, Integral) or maximum < 0
    ):
        raise ValueError("max_delay_samples must be a nonnegative integer")


def geometry_delays(geometry: pd.DataFrame, velocity_m_per_s: float, sample_interval_s: float):
    """Integer ``offset / velocity`` delays in geometry-table row order."""
    return moveout_delay_samples(
        geometry.offset_m.to_numpy(), velocity_m_per_s, float(sample_interval_s)
    )


def shared_alignment(
    velocity_m_per_s: float, geometries: list[pd.DataFrame], sample_interval_s: float
) -> dict:
    """Fix one maximum delay covering every supplied geometry, so all methods share a canvas."""
    maximum = max(
        int(geometry_delays(table, velocity_m_per_s, sample_interval_s).max())
        for table in geometries
    )
    alignment = {"velocity_m_per_s": float(velocity_m_per_s), "max_delay_samples": maximum}
    validate_alignment(alignment)
    return alignment


def moveout_canvas(inputs: ForgeModelInputs, alignment: dict) -> MoveoutCanvas:
    validate_alignment(alignment)
    interval = float(inputs.time_s[1] - inputs.time_s[0])
    delays = geometry_delays(inputs.geometry, alignment["velocity_m_per_s"], interval)
    maximum = alignment.get("max_delay_samples")
    if maximum is None:
        maximum = int(delays.max())
    elif int(delays.max()) > maximum:
        raise ValueError("geometry delay exceeds the declared max_delay_samples")
    return MoveoutCanvas(
        inputs.geometry.cell_id.to_numpy().copy(), delays, int(maximum), len(inputs.time_s)
    )


def align_model_inputs(
    inputs: ForgeModelInputs, alignment: dict, *, progress=None
) -> tuple[ForgeModelInputs, MoveoutCanvas]:
    """Place model-space observed traces on the canvas; test cells stay unread."""
    canvas = moveout_canvas(inputs, alignment)
    observed = align_moveout(
        np.asarray(inputs.observed_values).T,
        canvas.delays_for(inputs.observed_cells),
        max_delay=canvas.max_delay_samples,
        progress=progress,
    ).T
    interval = inputs.time_s[1] - inputs.time_s[0]
    aligned = replace(
        inputs,
        observed_values=np.ascontiguousarray(observed),
        time_s=np.arange(canvas.length, dtype=np.float64) * interval,
    )
    return aligned, canvas


def unalign_predictions(
    prediction: np.ndarray, cells: np.ndarray, canvas: MoveoutCanvas, *, progress=None
) -> np.ndarray:
    """Return rows ``[cell, canvas time]`` to their original ``time_sample_count`` samples."""
    if prediction.shape != (len(cells), canvas.length):
        raise ValueError("prediction does not match the aligned canvas")
    values = unalign_moveout(
        prediction.T,
        canvas.delays_for(cells),
        canvas.time_sample_count,
        max_delay=canvas.max_delay_samples,
        progress=progress,
    )
    return np.ascontiguousarray(values.T)
