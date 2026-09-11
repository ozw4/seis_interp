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
    coverage: Mapping[str, object],
    compute: Mapping[str, object],
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
        "coverage": dict(coverage),
        "compute": dict(compute),
        "artifacts": poc_artifact_metadata(
            prediction=remaining.get("prediction"), checkpoint=remaining.get("checkpoint")
        ),
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


def poc_coverage_metadata(
    target_mask: np.ndarray, coverage_mask: np.ndarray, *, time_sample_count: int
) -> dict[str, object]:
    """Summarize whole-trace coverage over T, including spatial boundary targets."""
    if (
        target_mask.dtype != np.bool_
        or coverage_mask.dtype != np.bool_
        or target_mask.ndim != 4
        or coverage_mask.shape != target_mask.shape
        or time_sample_count < 1
    ):
        raise ValueError(
            "coverage requires matching boolean spatial masks and positive time length"
        )
    targets = int(np.count_nonzero(target_mask))
    covered = int(np.count_nonzero(target_mask & coverage_mask))
    missing = target_mask & ~coverage_mask
    boundary_complete = all(
        not np.any(np.take(missing, [0, -1], axis=axis)) for axis in range(target_mask.ndim)
    )
    return {
        "target_trace_count": targets,
        "covered_target_trace_count": covered,
        "target_coverage_fraction": covered / targets if targets else 1.0,
        "uncovered_trace_count": targets - covered,
        "uncovered_sample_count": (targets - covered) * time_sample_count,
        "complete": covered == targets,
        "boundary_targets_included": boundary_complete,
    }


def poc_compute_metadata(
    *,
    parameter_count: int | None = None,
    optimizer_updates: int | None = None,
    supervised_trace_presentations: int | None = None,
) -> dict[str, int | None]:
    """Use null for compute counters that do not apply to classical reconstruction."""
    return {
        "parameter_count": parameter_count,
        "optimizer_updates": optimizer_updates,
        "supervised_trace_presentations": supervised_trace_presentations,
    }


def poc_artifact_metadata(
    *, prediction: Mapping[str, object] | None, checkpoint: Mapping[str, object] | None
) -> dict[str, object]:
    """Bind produced artifacts to run-relative paths and their file digests."""
    return {
        name: None if record is None else {"path": record["artifact"], "sha256": record["sha256"]}
        for name, record in (("prediction", prediction), ("checkpoint", checkpoint))
    }


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
