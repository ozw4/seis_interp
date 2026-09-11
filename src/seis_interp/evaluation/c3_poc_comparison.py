"""Validate and compare completed PoC run artifacts without reading amplitude arrays."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from seis_interp.data.file_checksums import file_sha256

SUMMARY_COLUMNS = (
    "method",
    "status",
    "output_directory",
    "snr_db",
    "snr_status",
    "rmse",
    "relative_l2",
    "mean_trace_relative_mse",
    "parameter_count",
    "optimizer_updates",
    "supervised_trace_presentations",
    "training_or_reconstruction_seconds",
    "prediction_seconds",
    "evaluation_seconds",
    "end_to_end_seconds",
    "process_max_rss_kib",
    "cuda_max_memory_allocated_bytes",
    "prediction_path",
    "prediction_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "error_type",
    "error_message",
)
_CLASSICAL_METHODS = {"pocs", "drr"}
_NEURAL_METHODS = {"nersi", "ccnet5d", "relational_trace_graph"}
_COMPUTE_FIELDS = (
    "parameter_count",
    "optimizer_updates",
    "supervised_trace_presentations",
)


def empty_poc_summary_row(method: str, output_directory: Path) -> dict[str, object]:
    """Give successful and failed methods the same comparison columns."""
    row = dict.fromkeys(SUMMARY_COLUMNS)
    row.update(method=method, status="not_run", output_directory=str(output_directory))
    return row


def read_poc_json(path: Path) -> dict[str, object]:
    """Read a JSON object, rejecting non-standard constants and numeric overflow."""
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
        parse_float=_finite_json_float,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_poc_run_artifacts(
    method: str, output_directory: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Return a comparable row only after verifying a successful run's recorded evidence."""
    if method not in _CLASSICAL_METHODS | _NEURAL_METHODS:
        raise ValueError(f"unsupported PoC method: {method}")
    output_directory = Path(output_directory)
    metadata = read_poc_json(output_directory / "metadata.json")
    inputs_lock = read_poc_json(output_directory / "inputs.lock.json")
    metrics = read_poc_json(output_directory / "metrics.json")
    if metadata.get("status") != "success":
        raise ValueError("metadata.status must be success")
    if metadata.get("method") != method:
        raise ValueError("metadata.method must match the requested method")
    if not inputs_lock:
        raise ValueError("inputs.lock.json must not be empty")
    for key in ("benchmark_id", "case_id", "volume_id"):
        if not isinstance(inputs_lock.get(key), str) or not inputs_lock[key]:
            raise ValueError(f"inputs.lock.json requires {key}")
        if metadata.get(key) != inputs_lock[key]:
            raise ValueError(f"metadata.{key} must match inputs.lock.json")

    row = empty_poc_summary_row(method, output_directory)
    row["coverage"] = _validated_coverage(metadata, inputs_lock)
    row.update(_validated_metrics(metrics, inputs_lock["target_trace_count"]))
    row.update(_validated_compute(metadata, neural=method in _NEURAL_METHODS))
    row.update(_validated_runtime(metadata, method))

    artifacts = _mapping(metadata, "artifacts")
    prediction = _mapping(artifacts, "prediction")
    row["prediction_path"], row["prediction_sha256"] = _validated_artifact(
        output_directory, prediction, "prediction.npy"
    )
    if method in _NEURAL_METHODS:
        checkpoint = _mapping(artifacts, "checkpoint")
        row["checkpoint_path"], row["checkpoint_sha256"] = _validated_artifact(
            output_directory, checkpoint, "final.pt"
        )
    elif "checkpoint" not in artifacts or artifacts["checkpoint"] is not None:
        raise ValueError("classical methods require a null checkpoint artifact")
    elif (output_directory / "final.pt").exists():
        raise ValueError("classical methods must not contain final.pt")
    row["status"] = "success"
    return row, inputs_lock


def matching_poc_input_lock(locks: list[dict[str, object]]) -> dict[str, object] | None:
    """Compare full verified locks, including all nested artifact hashes."""
    if not locks:
        return None
    if any(lock != locks[0] for lock in locks[1:]):
        raise ValueError("PoC methods used different verified input locks")
    return locks[0]


def write_poc_comparison(output_directory: Path, summary: dict[str, object]) -> None:
    """Write strict full-precision JSON and a fixed-column CSV for visual inspection."""
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(
        {name: _csv_cell(row[name]) for name in SUMMARY_COLUMNS} for row in summary["runs"]
    )
    output_directory = Path(output_directory)
    (output_directory / "summary.json").write_text(serialized, encoding="utf-8")
    (output_directory / "summary.csv").write_text(
        csv_buffer.getvalue(), encoding="utf-8", newline=""
    )


def _csv_cell(value: object) -> object:
    if not isinstance(value, float):
        return value
    if value == 0.0:
        return "0.0000"
    rounded = f"{value:.4f}"
    return "approximately 0" if rounded in ("0.0000", "-0.0000") else rounded


def _validated_coverage(
    metadata: dict[str, object], inputs_lock: dict[str, object]
) -> dict[str, object]:
    coverage = _mapping(metadata, "coverage")
    target_count = _nonnegative_integer(inputs_lock.get("target_trace_count"), "target count")
    if target_count == 0:
        raise ValueError("inputs.lock.json target_trace_count must be positive")
    for key, expected in (
        ("target_trace_count", target_count),
        ("covered_target_trace_count", target_count),
        ("uncovered_trace_count", 0),
        ("uncovered_sample_count", 0),
    ):
        if _nonnegative_integer(coverage.get(key), f"coverage.{key}") != expected:
            raise ValueError(f"coverage.{key} must be {expected}")
    if _finite_number(coverage.get("target_coverage_fraction"), "coverage fraction") != 1.0:
        raise ValueError("coverage.target_coverage_fraction must be 1")
    for key in ("complete", "boundary_targets_included"):
        if coverage.get(key) is not True:
            raise ValueError(f"coverage.{key} must be true")
    return coverage


def _validated_metrics(metrics: dict[str, object], target_count: int) -> dict[str, object]:
    if metrics.get("evaluation_domain") != "evaluation_target":
        raise ValueError("metrics.evaluation_domain must be evaluation_target")
    if metrics.get("amplitude_domain") != "physical":
        raise ValueError("metrics.amplitude_domain must be physical")
    target = _mapping(metrics, "evaluation_target")
    for key in ("target_trace_count", "covered_target_trace_count"):
        if _nonnegative_integer(target.get(key), f"metrics.{key}") != target_count:
            raise ValueError(f"metrics.{key} must agree with the verified input target count")
    names = ("snr_db", "snr_status", "rmse", "relative_l2", "mean_trace_relative_mse")
    if any(name not in target for name in names):
        raise ValueError("metrics.evaluation_target is missing common evaluation metrics")
    status = target["snr_status"]
    if status == "finite":
        _finite_number(target["snr_db"], "snr_db", nonnegative=False)
    elif status in ("perfect_reconstruction", "undefined_zero_reference"):
        if target["snr_db"] is not None:
            raise ValueError("degenerate SNR requires null snr_db")
    else:
        raise ValueError("metrics.snr_status is not a recognized physical-amplitude status")
    for name in ("rmse", "mean_trace_relative_mse"):
        _finite_number(target[name], name)
    if status == "undefined_zero_reference":
        if target["relative_l2"] is not None:
            raise ValueError("zero reference requires null relative_l2")
    else:
        _finite_number(target["relative_l2"], "relative_l2")
    return {name: target[name] for name in names}


def _validated_compute(metadata: dict[str, object], *, neural: bool) -> dict[str, object]:
    compute = _mapping(metadata, "compute")
    for name in _COMPUTE_FIELDS:
        if name not in compute:
            raise ValueError(f"compute.{name} is required")
        if neural:
            _nonnegative_integer(compute[name], f"compute.{name}")
        elif compute[name] is not None:
            raise ValueError(f"classical methods require null compute.{name}")
    return {name: compute[name] for name in _COMPUTE_FIELDS}


def _validated_runtime(metadata: dict[str, object], method: str) -> dict[str, object]:
    timing = _mapping(metadata, "timing")
    operation = (
        "reconstruction_seconds"
        if method in _CLASSICAL_METHODS
        else "fit_seconds"
        if method == "nersi"
        else "training_seconds"
    )
    result = {
        "training_or_reconstruction_seconds": _finite_number(timing.get(operation), operation),
        "prediction_seconds": None,
    }
    for name in ("evaluation_seconds", "end_to_end_seconds"):
        result[name] = _finite_number(timing.get(name), name)
    if method in _NEURAL_METHODS:
        result["prediction_seconds"] = _finite_number(
            timing.get("prediction_seconds"), "prediction_seconds"
        )
    resources = _mapping(metadata, "resource_usage")
    result["process_max_rss_kib"] = _nonnegative_integer(
        resources.get("process_max_rss_kib"), "process_max_rss_kib"
    )
    cuda_bytes = resources.get("cuda_max_memory_allocated_bytes")
    result["cuda_max_memory_allocated_bytes"] = (
        None
        if cuda_bytes is None
        else _nonnegative_integer(cuda_bytes, "cuda_max_memory_allocated_bytes")
    )
    return result


def _validated_artifact(
    output_directory: Path, artifact: dict[str, object], file_name: str
) -> tuple[str, str]:
    if artifact.get("path") != file_name:
        raise ValueError(f"artifact path must be {file_name}")
    path = output_directory / file_name
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    digest = file_sha256(path)
    if artifact.get("sha256") != digest:
        raise ValueError(f"{file_name} SHA-256 does not match metadata")
    return str(path), digest


def _mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: object, name: str, *, nonnegative: bool = True) -> int | float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"JSON must not contain non-finite values: {value}")


def _finite_json_float(value: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        _reject_nonfinite(value)
    return converted
