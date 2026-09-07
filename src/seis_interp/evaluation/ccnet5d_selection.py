"""Evaluate frozen CCNet-5D selection patches in the physical amplitude domain."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch

from seis_interp.data.c3_supervised_source import C3SupervisedSource
from seis_interp.training.ccnet5d_patches import CCNetPatchPlan, load_ccnet_patch

SELECTION_METRIC_SCOPE = "selection_patch_instances_missing_only"


def validate_ccnet5d_selection_targets(
    source: C3SupervisedSource,
    plan: CCNetPatchPlan,
) -> None:
    """Require positive finite energy across the fixed artificial-missing targets.

    Every selection descriptor is retained and read exactly once. In particular,
    an individual all-zero patch is valid when the aggregate selection reference
    energy remains positive.
    """
    amplitude_rms = _positive_finite_float(source.amplitude_rms, "source.amplitude_rms")
    patch_count = len(plan.selection)
    if patch_count == 0:
        raise ValueError("selection patch plan must not be empty")

    reference_energy = 0.0
    for index in range(patch_count):
        inputs, labels, observed_mask = load_ccnet_patch(
            source,
            plan,
            region="selection",
            index=index,
        )
        _, labels, observed_mask = _validated_patch(
            inputs,
            labels,
            observed_mask,
            patch_shape=plan.patch_shape,
        )
        reference = _missing_physical_reference(
            labels,
            observed_mask,
            amplitude_rms=amplitude_rms,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            reference_energy += float(np.sum(np.square(reference), dtype=np.float64))
    if not math.isfinite(reference_energy) or reference_energy <= 0.0:
        raise ValueError("selection target reference energy must be positive and finite")


def evaluate_ccnet5d_selection(
    model: torch.nn.Module,
    source: C3SupervisedSource,
    plan: CCNetPatchPlan,
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Return global physical-domain energies over artificial missing traces only."""
    amplitude_rms = _positive_finite_float(source.amplitude_rms, "source.amplitude_rms")
    patch_count = len(plan.selection)
    if patch_count == 0:
        raise ValueError("selection patch plan must not be empty")

    device_value = torch.device(device)
    model.to(device_value)
    was_training = model.training
    reference_energy = 0.0
    error_energy = 0.0
    missing_sample_count = 0
    try:
        model.eval()
        with torch.no_grad():
            for index in range(patch_count):
                inputs, labels, observed_mask = load_ccnet_patch(
                    source,
                    plan,
                    region="selection",
                    index=index,
                )
                inputs, labels, observed_mask = _validated_patch(
                    inputs,
                    labels,
                    observed_mask,
                    patch_shape=plan.patch_shape,
                )
                input_tensor = torch.from_numpy(inputs[None, None]).to(device_value)
                prediction_tensor = model(input_tensor)
                if prediction_tensor.shape != input_tensor.shape:
                    raise ValueError("CCNet-5D selection output shape must match its input shape")
                predictions = prediction_tensor[0, 0].detach().cpu().numpy()
                patch_reference, patch_error, patch_samples = _missing_physical_energies(
                    labels,
                    predictions,
                    observed_mask,
                    amplitude_rms=amplitude_rms,
                )
                reference_energy += patch_reference
                error_energy += patch_error
                missing_sample_count += patch_samples
    finally:
        model.train(was_training)

    if not math.isfinite(reference_energy) or reference_energy <= 0.0:
        raise ValueError("selection reference energy must be positive and finite")
    if not math.isfinite(error_energy) or error_energy < 0.0:
        raise ValueError("selection error energy must be non-negative and finite")
    relative_squared_error = error_energy / reference_energy
    if not math.isfinite(relative_squared_error):
        raise ValueError("selection relative squared error must be finite")
    if error_energy == 0.0:
        snr_db = None
        snr_status = "perfect_reconstruction"
    else:
        snr_db = float(10.0 * (math.log10(reference_energy) - math.log10(error_energy)))
        if not math.isfinite(snr_db):
            raise ValueError("selection S/N must be finite")
        snr_status = "finite"
    return {
        "patch_count": patch_count,
        "missing_sample_count": missing_sample_count,
        "reference_energy": reference_energy,
        "error_energy": error_energy,
        "relative_squared_error": relative_squared_error,
        "snr_db": snr_db,
        "snr_status": snr_status,
        "metric_scope": SELECTION_METRIC_SCOPE,
    }


def _validated_patch(
    inputs: object,
    labels: object,
    observed_mask: object,
    *,
    patch_shape: tuple[int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(inputs, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("CCNet-5D patch inputs and labels must be NumPy arrays")
    if inputs.shape != patch_shape or labels.shape != patch_shape:
        raise ValueError("CCNet-5D patch inputs and labels must match plan.patch_shape")
    if inputs.dtype != np.float32 or labels.dtype != np.float32:
        raise TypeError("CCNet-5D patch inputs and labels must have dtype float32")
    if not isinstance(observed_mask, np.ndarray) or observed_mask.shape != patch_shape[1:]:
        raise ValueError("CCNet-5D observed mask must match the four spatial patch axes")
    if observed_mask.dtype != np.bool_:
        raise TypeError("CCNet-5D observed mask must have dtype bool")
    if not np.any(~observed_mask):
        raise ValueError("CCNet-5D selection patch must contain missing traces")
    return inputs, labels, observed_mask


def _missing_physical_energies(
    labels: np.ndarray,
    predictions: np.ndarray,
    observed_mask: np.ndarray,
    *,
    amplitude_rms: float,
) -> tuple[float, float, int]:
    if predictions.shape != labels.shape:
        raise ValueError("CCNet-5D selection prediction shape must match labels")
    missing_mask = np.broadcast_to(~observed_mask, labels.shape)
    reference = _missing_physical_reference(
        labels,
        observed_mask,
        amplitude_rms=amplitude_rms,
    )
    prediction = np.asarray(predictions)[missing_mask].astype(np.float64, copy=False)
    prediction *= amplitude_rms
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(prediction)):
        raise ValueError("CCNet-5D selection values must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        reference_energy = float(np.sum(np.square(reference), dtype=np.float64))
        error = reference - prediction
        error_energy = float(np.sum(np.square(error), dtype=np.float64))
    if not math.isfinite(reference_energy) or not math.isfinite(error_energy):
        raise ValueError("CCNet-5D selection energies must be finite")
    return reference_energy, error_energy, int(reference.size)


def _missing_physical_reference(
    labels: np.ndarray,
    observed_mask: np.ndarray,
    *,
    amplitude_rms: float,
) -> np.ndarray:
    missing_mask = np.broadcast_to(~observed_mask, labels.shape)
    return labels[missing_mask].astype(np.float64, copy=False) * amplitude_rms


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted
