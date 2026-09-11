"""Exploratory DRR parameter scoring on explicitly selected target regions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.evaluation.physical_amplitude_metrics import physical_amplitude_target_metrics
from seis_interp.processing.drr_windows import interpolate_drr_volume


def score_drr_target_regions(
    observed: ObservedC3Volume,
    *,
    interim_dir: Path,
    volume_metadata: Mapping[str, object],
    parameters: Mapping[str, object],
    regions: Sequence[tuple[slice, slice, slice, slice]],
) -> dict[str, object]:
    """Reconstruct from O, then score T; these scores are for tuning, not testing.

    Nonoverlapping pilot regions retain every time sample. Aggregate energies
    using the same evaluator as full-volume runs. Full-volume confirmation is
    required because regional scores do not measure the complete benchmark.
    """
    seen = np.zeros_like(observed.observed_trace_mask)
    totals = dict(trace_count=0, sample_count=0, reference_energy=0.0, error_energy=0.0)
    for region in regions:
        if seen[region].any():
            raise ValueError("target regions must not overlap")
        seen[region] = True
        block = replace(
            observed,
            values=observed.values[(slice(None), *region)],
            array_rows=observed.array_rows[region],
            observed_trace_mask=observed.observed_trace_mask[region],
            evaluation_target_trace_mask=observed.evaluation_target_trace_mask[region],
        )
        reconstructed = interpolate_drr_volume(
            block.values, block.observed_trace_mask, block.time_s, **parameters
        )
        scores = evaluate_c3_volume_prediction(
            reconstructed.values,
            block,
            interim_dir=interim_dir,
            volume_metadata=volume_metadata,
        )
        if scores["observed_max_abs_error"] != 0:
            raise ValueError("DRR must preserve observed amplitudes exactly")
        for key in totals:
            totals[key] += scores["evaluation_target"][key]
    if totals["trace_count"] == 0:
        raise ValueError("target regions must contain evaluation traces")
    return physical_amplitude_target_metrics(**totals)
