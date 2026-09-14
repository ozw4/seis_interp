"""Reference-only target scoring before trace RMS amplitude restoration."""

from pathlib import Path

import numpy as np

from seis_interp.data.c3_volume_adapter import volume_to_trace_predictions
from seis_interp.evaluation.physical_amplitude_metrics import (
    physical_amplitude_energies,
    physical_amplitude_target_metrics,
)


def unit_rms_reference_traces(reference: np.ndarray) -> np.ndarray:
    """Normalize complete reference traces in float64, retaining zero traces as zero."""
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("reference must be nonempty finite [trace, time] amplitudes")
    peak = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = values / np.where(peak > 0, peak, 1)
    rms = np.sqrt(np.mean(scaled * scaled, axis=1, keepdims=True))
    return scaled / np.where(rms > 0, rms, 1)


def evaluate_normalized_trace_reference(
    prediction,
    observed,
    trace_amplitude_scale,
    *,
    interim_dir: Path,
    volume_metadata,
    chunk_size: int = 1024,
) -> dict:
    """Score physical prediction / O-fitted RMS against target / true target RMS.

    True T RMS is used only in this evaluation boundary. The prediction's own
    RMS is not divided out: normalized model gain errors remain penalized.
    Positive restoration scales are required to recover the normalized output.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if prediction.shape != observed.values.shape:
        raise ValueError("prediction must cover the full volume")
    scale = np.asarray(trace_amplitude_scale, dtype=np.float64)
    mask = observed.evaluation_target_trace_mask
    if scale.shape != mask.shape or not np.isfinite(scale[mask]).all() or np.any(scale[mask] <= 0):
        raise ValueError("target restoration scales must be finite and positive")
    positions = np.flatnonzero(mask.ravel())
    if not len(positions):
        raise ValueError("reference evaluation requires target traces")
    rows, predictions = volume_to_trace_predictions(prediction, observed.array_rows)
    start, stop = volume_metadata["selection"]["time"]
    if start < 0 or stop - start != prediction.shape[0]:
        raise ValueError("time selection must match prediction")
    amplitudes = np.load(Path(interim_dir) / "amplitudes.npy", mmap_mode="r", allow_pickle=False)
    if (
        amplitudes.ndim != 2
        or stop > amplitudes.shape[1]
        or np.any(rows[positions] < 0)
        or np.any(rows[positions] >= len(amplitudes))
    ):
        raise ValueError("target rows/time exceed amplitude source")
    reference_energy = error_energy = 0.0
    zero_reference_traces = 0
    for offset in range(0, len(positions), chunk_size):
        selected = positions[offset : offset + chunk_size]
        reference = unit_rms_reference_traces(amplitudes[rows[selected], start:stop])
        normalized_prediction = (
            predictions[selected].astype(np.float64) / scale.ravel()[selected, None]
        )
        a, b = physical_amplitude_energies(reference, normalized_prediction)
        reference_energy += a
        error_energy += b
        zero_reference_traces += int(np.count_nonzero(~np.any(reference, axis=1)))
    metrics = physical_amplitude_target_metrics(
        trace_count=len(positions),
        sample_count=len(positions) * (stop - start),
        reference_energy=reference_energy,
        error_energy=error_energy,
    )
    return {
        "evaluation_domain": "evaluation_target",
        "amplitude_domain": "per_trace_unit_rms_reference",
        "prediction_domain": "model_output_before_O_only_RMS_restoration",
        "prediction_self_normalized": False,
        "target_rms_used_only_for_evaluation": True,
        "zero_reference_trace_count": zero_reference_traces,
        "evaluation_target": metrics,
    }
