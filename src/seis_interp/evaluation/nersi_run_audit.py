"""Independently rescore and restore a completed C3 NeRSI run."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_volume_inputs
from seis_interp.data.c3_benchmark_suite import (
    c3_suite_case,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.training.c3_volume_nersi_data import build_c3_volume_nersi_data
from seis_interp.training.c3_volume_nersi_prediction import predict_c3_volume_nersi
from seis_interp.training.nersi_checkpoints import (
    FIXED_STEP_FINAL_CHECKPOINT_ROLE,
    NERSI_METHOD_VARIANT,
    load_fixed_step_nersi_checkpoint,
    validate_fixed_step_nersi_checkpoint_input_binding,
)

RESTORE_RTOL = 1.0e-6
RESTORE_ATOL = 1.0e-6
RUN_FILES = (
    "run.json",
    "metrics.json",
    "config.resolved.yaml",
    "inputs.lock.json",
    "artifacts/final.pt",
    "artifacts/prediction.npy",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def _recorded_numerical_settings(resources: dict):
    original = {
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        if "cudnn_benchmark" in resources:
            torch.backends.cudnn.benchmark = resources["cudnn_benchmark"]
        if "cudnn_deterministic" in resources:
            torch.backends.cudnn.deterministic = resources["cudnn_deterministic"]
        if "cuda_matmul_allow_tf32" in resources:
            torch.backends.cuda.matmul.allow_tf32 = resources["cuda_matmul_allow_tf32"]
        if "cudnn_allow_tf32" in resources:
            torch.backends.cudnn.allow_tf32 = resources["cudnn_allow_tf32"]
        if "float32_matmul_precision" in resources:
            torch.set_float32_matmul_precision(resources["float32_matmul_precision"])
        yield
    finally:
        torch.backends.cudnn.benchmark = original["benchmark"]
        torch.backends.cudnn.deterministic = original["deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = original["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = original["cudnn_tf32"]
        torch.set_float32_matmul_precision(original["matmul_precision"])


def _target_metrics_match(actual: dict, expected: dict) -> bool:
    exact_fields = ("trace_count", "sample_count", "snr_status")
    if any(actual[name] != expected[name] for name in exact_fields):
        return False
    for name in ("snr_db", "rmse", "relative_l2", "reference_energy", "error_energy"):
        left, right = actual[name], expected[name]
        if left is None or right is None:
            if left is not right:
                return False
        elif not np.isclose(left, right, rtol=RESTORE_RTOL, atol=RESTORE_ATOL):
            return False
    return True


def audit_c3_nersi_run(
    *,
    suite_dir: Path,
    case_id: str,
    run_dir: Path,
    output_dir: Path,
    expected_suite_sha256: str,
    restore_device: torch.device | str = "cpu",
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict[str, object]:
    """Verify one immutable final NeRSI run and write a separate audit result."""
    suite = Path(suite_dir)
    run = Path(run_dir)
    output = Path(output_dir)
    _require(not output.exists(), "audit output directory must not exist")
    manifest_path = suite / "benchmark_suite.json"
    _require(file_sha256(manifest_path) == expected_suite_sha256, "suite SHA-256 differs")
    manifest = load_c3_benchmark_input_manifest(suite, case_id=case_id, dimensions=dimensions)
    _require(
        c3_suite_case(manifest, case_id)["partition"] == "validation",
        "NeRSI audit is restricted to validation",
    )
    inputs = load_c3_benchmark_volume_inputs(suite, case_id, dimensions=dimensions)
    metadata = _read(run / "run.json")
    metrics = _read(run / "metrics.json")
    inputs_lock = _read(run / "inputs.lock.json")
    _require(metadata.get("status") == "success", "audit requires a successful run")
    _require(
        metadata.get("method") == metrics.get("method") == "nersi",
        "audit requires a NeRSI run",
    )
    _require(
        metadata.get("method_variant") == metrics.get("method_variant") == NERSI_METHOD_VARIANT,
        "NeRSI method variant differs",
    )
    _require(
        metadata.get("case_id") == metrics.get("case_id") == case_id,
        "NeRSI case differs",
    )
    _require(
        metadata.get("volume_id")
        == metrics.get("volume_id")
        == inputs.volume_metadata["volume_id"],
        "NeRSI volume differs",
    )
    _require(inputs_lock == inputs.inputs_lock, "NeRSI input lock differs from verified inputs")

    checkpoint_path = run / "artifacts/final.pt"
    prediction_path = run / "artifacts/prediction.npy"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    prediction_sha256 = file_sha256(prediction_path)
    _require(
        metadata.get("checkpoint", {}).get("role") == FIXED_STEP_FINAL_CHECKPOINT_ROLE,
        "NeRSI checkpoint is not fixed-step final",
    )
    _require(
        metadata["checkpoint"].get("sha256") == checkpoint_sha256,
        "NeRSI checkpoint SHA-256 differs",
    )
    _require(
        metadata.get("prediction", {}).get("sha256") == prediction_sha256,
        "NeRSI prediction SHA-256 differs",
    )
    _require(
        metadata.get("training", {}).get("steps_completed")
        == metadata.get("training", {}).get("max_steps"),
        "NeRSI final optimizer budget is incomplete",
    )

    recorded_hashes = {name: file_sha256(run / name) for name in RUN_FILES}
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    observed = inputs.observed_volume
    _require(
        prediction.dtype == np.float32 and prediction.shape == observed.values.shape,
        "NeRSI prediction must be float32 with the full selected-volume shape",
    )
    _require(bool(np.isfinite(prediction).all()), "NeRSI prediction must be finite")
    mask = observed.observed_trace_mask
    _require(
        np.array_equal(prediction[:, mask], observed.values[:, mask].astype(np.float32)),
        "NeRSI prediction does not preserve hard observed-data consistency",
    )
    rescored = evaluate_c3_volume_prediction(
        prediction,
        observed,
        interim_dir=suite_path(suite, manifest["interim"]),
        volume_metadata=inputs.volume_metadata,
    )

    data = build_c3_volume_nersi_data(observed)
    loaded = load_fixed_step_nersi_checkpoint(checkpoint_path, device="cpu")
    validate_fixed_step_nersi_checkpoint_input_binding(loaded, inputs.inputs_lock, data)
    _require(
        loaded.global_step == metadata["training"]["max_steps"],
        "checkpoint global step differs from the declared final budget",
    )
    with _recorded_numerical_settings(metadata.get("resources", {})):
        restored = predict_c3_volume_nersi(
            loaded.model,
            data,
            observed,
            batch_size=metadata["prediction"]["batch_size"],
            device=restore_device,
        )
    difference = restored.values.astype(np.float64) - prediction.astype(np.float64)
    squared_difference = float(np.sum(np.square(difference), dtype=np.float64))
    restored_reference_energy = float(
        np.sum(np.square(prediction.astype(np.float64)), dtype=np.float64)
    )
    restoration = {
        "status": (
            "passed"
            if np.allclose(
                restored.values,
                prediction,
                rtol=RESTORE_RTOL,
                atol=RESTORE_ATOL,
            )
            else "failed_tolerance"
        ),
        "device": str(restore_device),
        "rmse": math.sqrt(squared_difference / prediction.size),
        "max_abs_error": float(np.max(np.abs(difference), initial=0.0)),
        "physical_relative_l2": (
            math.sqrt(squared_difference / restored_reference_energy)
            if restored_reference_energy > 0.0
            else (0.0 if squared_difference == 0.0 else None)
        ),
        "rtol": RESTORE_RTOL,
        "atol": RESTORE_ATOL,
        "observed_model_rmse_before_reinsertion": (restored.observed_model_rmse_before_reinsertion),
        "observed_model_max_abs_error_before_reinsertion": (
            restored.observed_model_max_abs_error_before_reinsertion
        ),
    }
    target_count = int(np.count_nonzero(observed.evaluation_target_trace_mask))
    checks = {
        "saved_metrics_match_independent_rescore": _target_metrics_match(
            rescored["evaluation_target"], metrics["evaluation_target"]
        ),
        "full_target_coverage": (
            rescored["evaluation_target"]["trace_count"] == target_count
            and rescored["evaluation_target"]["sample_count"]
            == target_count * observed.values.shape[0]
        ),
        "observed_reinsertion_exact": rescored["observed_max_abs_error"] == 0.0,
        "checkpoint_restoration": restoration["status"] == "passed",
        "run_artifacts_unchanged": all(
            file_sha256(run / name) == digest for name, digest in recorded_hashes.items()
        ),
    }
    report = {
        "status": "success" if all(checks.values()) else "failed_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "volume_id": inputs.volume_metadata["volume_id"],
        "suite_sha256": expected_suite_sha256,
        "run_directory": str(run.resolve()),
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha256},
        "prediction": {"path": str(prediction_path.resolve()), "sha256": prediction_sha256},
        "checks": checks,
        "independent_saved_output_metrics": rescored,
        "checkpoint_restoration": restoration,
        "run_artifact_sha256": recorded_hashes,
        "target_truth_used_for_restoration": False,
        "test_partition_used": False,
    }
    output.mkdir(parents=True)
    _write(output / "result.json", report)
    return report
