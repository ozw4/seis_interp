"""Common metadata and physical prediction contract for the five PoC methods."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seis_interp.data.c3_volume_adapter import ObservedC3Volume

METADATA_FILE_NAME = "metadata.json"


def poc_run_metadata(
    inputs_lock: Mapping[str, object],
    details: Mapping[str, object],
    *,
    normalization: Mapping[str, object],
    objective: str,
    operation: Mapping[str, object],
) -> dict[str, object]:
    """Group method details beneath the common, human-readable run fields."""
    remaining = deepcopy(dict(details))
    resources = remaining.pop("resources", {})
    timing = remaining.pop("timing", {})
    timing.update({key: resources.pop(key) for key in list(resources) if key.endswith("_seconds")})
    record = {
        "method": remaining.pop("method"),
        "benchmark_id": inputs_lock["benchmark_id"],
        "case_id": inputs_lock["case_id"],
        "volume_id": inputs_lock["volume_id"],
        "status": remaining.pop("status"),
        "input_amplitude_domain": "physical",
        "output_amplitude_domain": "physical",
        "normalization": dict(normalization),
        "loss_or_native_objective": objective,
        "training_or_reconstruction": dict(operation),
        "coverage": remaining.pop("coverage", {}),
        "timing": timing,
        "resource_usage": resources,
    }
    for key in (
        "benchmark_id",
        "case_id",
        "volume_id",
        "input_amplitude_domain",
        "output_amplitude_domain",
        "normalization",
        "loss",
        "training",
    ):
        remaining.pop(key, None)
    record["method_details"] = remaining
    return record


def validate_poc_prediction(prediction: np.ndarray, observed_volume: ObservedC3Volume) -> None:
    """Require complete finite physical output and exact observed reinsertion."""
    if (
        not isinstance(prediction, np.ndarray)
        or prediction.shape != observed_volume.values.shape
        or prediction.dtype.kind not in "fiu"
    ):
        raise ValueError("PoC prediction must be a real array matching the complete volume shape")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("PoC prediction must be finite on the complete volume")
    mask = observed_volume.observed_trace_mask
    if not np.array_equal(prediction[:, mask], observed_volume.values[:, mask]):
        raise ValueError("PoC prediction must preserve observed physical amplitudes exactly")
