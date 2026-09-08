"""Read-only suite verification and a full-volume zero-fill sanity run."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from seis_interp import run_records
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_volume_inputs
from seis_interp.data.c3_benchmark_suite import (
    VerifiedC3BenchmarkSuite,
    c3_suite_case,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.runtime_environment import collect_runtime_environment


def _require_expected_hash(suite_dir: Path, expected_sha256: str) -> str:
    actual = file_sha256(suite_dir / "benchmark_suite.json")
    if actual != expected_sha256:
        raise ValueError("first-results suite SHA-256 differs from expected hash")
    return actual


def check_c3_first_results(
    *,
    suite_dir: Path,
    case_id: str,
    output_dir: Path,
    config: dict,
    expected_sha256: str,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict:
    """Verify all suite files once, then measure the selected observed volume."""
    run_records.check_new_output_directory(output_dir)
    suite_hash = _require_expected_hash(suite_dir, expected_sha256)
    started_at = run_records.utc_timestamp()
    git = run_records.current_git_metadata()
    start = time.perf_counter()
    verified = VerifiedC3BenchmarkSuite(suite_dir, dimensions=dimensions)
    verify_seconds = time.perf_counter() - start
    suite = load_c3_benchmark_input_manifest(
        suite_dir, case_id=case_id, dimensions=dimensions, verified_suite=verified
    )
    entry = c3_suite_case(suite, case_id)
    start = time.perf_counter()
    inputs = load_c3_benchmark_volume_inputs(
        suite_dir, case_id, dimensions=dimensions, verified_suite=verified
    )
    load_seconds = time.perf_counter() - start
    observed = inputs.observed_volume
    environment = collect_runtime_environment(
        suite_dir, device=config.get("execution", {}).get("device", "cuda:1")
    )
    metrics = {
        "status": "verified",
        "scope": "complete_suite_verification",
        "suite_sha256": suite_hash,
        "verified_file_count": len(suite["files"]),
        "verified_case_count": len(suite["cases"]),
        "case_id": case_id,
        "volume_id": inputs.volume_metadata["volume_id"],
        "selection": inputs.volume_metadata["selection"],
        "shape": list(observed.values.shape),
        "time_s": {
            "first": float(observed.time_s[0]),
            "last": float(observed.time_s[-1]),
            "sample_interval_s": float(observed.time_s[1] - observed.time_s[0]),
        },
        "observed_trace_count": int(np.count_nonzero(observed.observed_trace_mask)),
        "target_trace_count": int(np.count_nonzero(observed.evaluation_target_trace_mask)),
        "trace_count": int(observed.observed_trace_mask.size),
        "mask_seed": entry["random_seed"],
        "effective_mask": entry["effective_mask"],
        "train_pool": suite["train_pool"],
        "environment": environment,
    }
    run_records.write_run_outputs(
        output_dir,
        config,
        inputs.inputs_lock,
        metrics,
        {
            **git,
            "started_at_utc": started_at,
            "finished_at_utc": run_records.utc_timestamp(),
            "timings": {
                "complete_verification_seconds": verify_seconds,
                "selected_inputs_load_seconds": load_seconds,
            },
        },
    )
    return metrics


def zero_fill_c3_first_results(
    *,
    suite_dir: Path,
    case_id: str,
    output_dir: Path,
    config: dict,
    expected_sha256: str,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict:
    """Use only observed amplitudes as prediction; truth enters the native evaluator."""
    run_records.check_new_output_directory(output_dir)
    suite_hash = _require_expected_hash(suite_dir, expected_sha256)
    start = time.perf_counter()
    started_at = run_records.utc_timestamp()
    git = run_records.current_git_metadata()
    suite = load_c3_benchmark_input_manifest(suite_dir, case_id=case_id, dimensions=dimensions)
    inputs = load_c3_benchmark_volume_inputs(suite_dir, case_id, dimensions=dimensions)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    metrics = evaluate_c3_volume_prediction(
        inputs.observed_volume.values,
        inputs.observed_volume,
        interim_dir=suite_path(suite_dir, suite["interim"]),
        volume_metadata=inputs.volume_metadata,
    )
    evaluation_seconds = time.perf_counter() - start
    target = metrics["evaluation_target"]
    if target["reference_energy"] != target["error_energy"]:
        raise ValueError("zero-fill reference and error energy must match")
    if target["reference_energy"] > 0 and not np.isclose(target["snr_db"], 0, atol=1e-12):
        raise ValueError("nonzero-reference zero-fill SNR must be zero dB")
    metrics.update(
        {
            "method": "zero_fill",
            "scope": "full_volume",
            "suite_sha256": suite_hash,
            "case_id": case_id,
            "volume_id": inputs.volume_metadata["volume_id"],
        }
    )
    # Preserve the native prediction layout for direct memory-mapped comparison.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "artifacts").mkdir()
    np.save(
        output_dir / "artifacts/prediction.npy", inputs.observed_volume.values, allow_pickle=False
    )
    run_records.write_run_outputs(
        output_dir,
        config,
        inputs.inputs_lock,
        metrics,
        {
            **git,
            "method": "zero_fill",
            "started_at_utc": started_at,
            "finished_at_utc": run_records.utc_timestamp(),
            "timings": {
                "load_and_verification_seconds": load_seconds,
                "evaluation_seconds": evaluation_seconds,
            },
            "resources": collect_runtime_environment(suite_dir, device="cpu"),
        },
    )
    return metrics
