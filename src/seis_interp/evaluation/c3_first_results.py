"""Compare an explicit finite list of immutable C3 pilot prediction runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp import run_records
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_volume_inputs
from seis_interp.data.c3_benchmark_suite import (
    c3_suite_case,
    load_c3_benchmark_input_manifest,
)
from seis_interp.data.c3_benchmark_suite import (
    suite_path as resolve_suite_path,
)
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.visualization.c3_first_results import plot_c3_first_results

METHODS = ("pocs", "drr", "siren5d", "ccnet5d", "relational_trace_graph")
OPTIONAL_METHODS = ("nersi",)
_NATIVE_METHODS = {
    "pocs": "pocs_fourier_5d",
    "drr": "damped_rank_reduction_5d",
    "siren5d": "siren_5d",
    "ccnet5d": "ccnet5d",
    "relational_trace_graph": "relational_trace_graph",
    "nersi": "nersi",
}
_METRIC_FIELDS = (
    "trace_count",
    "sample_count",
    "snr_db",
    "snr_status",
    "rmse",
    "relative_l2",
    "reference_energy",
    "error_energy",
)


def summarize_c3_first_results(
    *,
    suite_path: Path,
    case_id: str,
    method_runs: Mapping[str, Path | None],
    output_dir: Path,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict[str, object]:
    """Re-score named full-volume predictions; retain failed and absent methods.

    Paths select native directories or outer action directories with request.json
    and result.json. No discovery, ranking, training, or checkpoint loading occurs.
    A zero-fill row is always evaluated from the verified observed input.
    """
    if not isinstance(method_runs, Mapping) or set(method_runs) - {
        "zero_fill",
        *METHODS,
        *OPTIONAL_METHODS,
    }:
        raise ValueError(
            "method_runs must map only the five required methods, optional nersi, and zero_fill"
        )
    if set(METHODS) - set(method_runs):
        raise ValueError(
            "method_runs must explicitly include all five methods; use null for absent runs"
        )
    paths = [str(Path(path).resolve()) for path in method_runs.values() if path is not None]
    if len(paths) != len(set(paths)):
        raise ValueError("different methods cannot select the same run path")
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    manifest_path = Path(suite_path)
    suite = load_c3_benchmark_input_manifest(
        manifest_path.parent, case_id=case_id, dimensions=dimensions
    )
    entry = c3_suite_case(suite, case_id)
    inputs = load_c3_benchmark_volume_inputs(manifest_path.parent, case_id, dimensions=dimensions)
    if inputs.case["partition"] != "validation":
        raise ValueError("first-results comparison requires the fixed validation partition")
    interim = resolve_suite_path(manifest_path.parent, suite["interim"])
    binding = _binding(inputs, file_sha256(manifest_path))
    baseline_started = time.perf_counter()
    baseline_metrics = evaluate_c3_volume_prediction(
        inputs.observed_volume.values,
        inputs.observed_volume,
        interim_dir=interim,
        volume_metadata=inputs.volume_metadata,
    )["evaluation_target"]
    include_training_context = method_runs.get("nersi") is not None
    baseline = _empty_row(
        "zero_fill",
        method_runs.get("zero_fill"),
        binding,
        include_training_context=include_training_context,
    )
    baseline.update(
        status="success", reason=None, method_variant="sanity_zero_fill", **baseline_metrics
    )
    baseline["summary_re_evaluation_seconds"] = time.perf_counter() - baseline_started
    baseline["snr_gain_over_zero_fill_db"] = 0.0 if baseline_metrics["snr_db"] is not None else None
    baseline["metric_comparison"] = {"status": "computed_from_verified_observed_input"}
    if method_runs.get("zero_fill") is not None:
        _collect_baseline(Path(method_runs["zero_fill"]), inputs, interim, binding, baseline)
    rows = [baseline]
    predictions = {}
    histories = {}
    selected_methods = (
        *METHODS,
        *(method for method in OPTIONAL_METHODS if method_runs.get(method) is not None),
    )
    for method in selected_methods:
        row = _empty_row(
            method,
            method_runs[method],
            binding,
            include_training_context=include_training_context,
        )
        if method_runs[method] is not None:
            try:
                prediction, history = _collect_run(
                    method, Path(method_runs[method]), inputs, interim, binding, row
                )
                if prediction is not None:
                    predictions[method] = prediction
                    histories[method] = history
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="invalid_result", reason=f"{type(error).__name__}: {error}")
        if row["snr_db"] is not None and baseline_metrics["snr_db"] is not None:
            row["snr_gain_over_zero_fill_db"] = row["snr_db"] - baseline_metrics["snr_db"]
        rows.append(row)
    for row in rows:
        if row["status"] == "success" and row["snr_db"] is None:
            row["null_reasons"]["snr_db"] = f"SNR status: {row['snr_status']}."
        if row["snr_gain_over_zero_fill_db"] is None:
            row["null_reasons"]["snr_gain_over_zero_fill_db"] = (
                "SNR difference requires finite method and zero-fill SNR values."
            )
        if row["status"] == "success" and row["relative_l2"] is None:
            row["null_reasons"]["relative_l2"] = "Reference energy is zero."
        row["null_reasons"] = {
            key: reason for key, reason in row["null_reasons"].items() if row.get(key) is None
        }
    output.mkdir(parents=True, exist_ok=False)
    figures = plot_c3_first_results(
        inputs=inputs,
        interim_dir=interim,
        predictions=predictions,
        histories=histories,
        output_dir=output,
    )
    result = {
        "status": "first_results_complete"
        if all(row["status"] == "success" for row in rows[1:])
        else "partial_results",
        "case_id": case_id,
        "suite_path": str(manifest_path.resolve()),
        "volume_dir": str(resolve_suite_path(manifest_path.parent, entry["volume_dir"])),
        "binding": binding,
        "selection_policy": "explicit_paths_only; final_checkpoints; no_best_or_latest_search",
        "rows": rows,
        "figures": figures,
        "coverage_definitions": {
            "uncovered_sample_count": (
                "Window overlap left target samples without a reconstruction contribution."
            ),
            "no_context_query_count": (
                "Graph query has no observed dependency context; remains in overall metric."
            ),
        },
        "limitations": [
            "Fixed-budget pilot observations; no claim of converged or publication ranking.",
            "Training exposure and losses differ; forward times cannot rank end-to-end methods.",
            "Null resource and timing values are unmeasured or inapplicable, never zero estimates.",
        ],
    }
    (output / "first_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "first_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    return result


def _binding(inputs: C3VolumeRunInputs, suite_hash: str) -> dict:
    volume = inputs.observed_volume
    targets = np.sort(volume.array_rows[volume.evaluation_target_trace_mask]).astype("<i8")
    return {
        "suite_sha256": suite_hash,
        "case_id": inputs.case["case_id"],
        "case_sha256": inputs.inputs_lock["benchmark_case"]["sha256"],
        "volume_id": inputs.volume_metadata["volume_id"],
        "volume_files": inputs.inputs_lock["benchmark_volume"]["files"],
        "partition": inputs.case["partition"],
        "mask_seed": inputs.case["mask"]["random_seed"],
        "shape": list(volume.values.shape),
        "selection": inputs.volume_metadata["selection"],
        "time_s": volume.time_s.tolist(),
        "target_array_rows_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
        "target_id_hash_encoding": "sorted array_row IDs as little-endian int64 bytes",
        "target_count": int(len(targets)),
        "sample_count": int(len(targets) * len(volume.time_s)),
        "actual_missing_fraction": float(len(targets) / volume.array_rows.size),
    }


def _empty_row(
    method: str,
    path: Path | None,
    binding: dict,
    *,
    include_training_context: bool,
) -> dict:
    regimes = {
        "zero_fill": "no_training",
        "pocs": "per_volume_iterative_reconstruction_no_learned_parameters",
        "drr": "per_volume_iterative_reconstruction_no_learned_parameters",
        "siren5d": "per_volume_observed_only_internal_learning",
        "nersi": "per_volume_observed_only_internal_learning",
        "ccnet5d": "train_partition_pretraining_then_frozen_validation_inference",
        "relational_trace_graph": (
            "masked_train_partition_pretraining_then_frozen_validation_inference"
        ),
    }
    row = {
        "method": method,
        **(
            {
                "training_regime": regimes[method],
                "additional_supervised_training_data": method
                in ("ccnet5d", "relational_trace_graph"),
                "target_volume_model_optimization": method in ("siren5d", "nersi"),
            }
            if include_training_context
            else {}
        ),
        "method_variant": None,
        "model": None,
        "parameter_count": None,
        "amplitude": None,
        "training_amplitude": None,
        "graph": None,
        "neighbors_per_relation": None,
        "prediction_settings": None,
        "run_path": None if path is None else str(Path(path).resolve()),
        "native_run_path": None,
        "training_run_path": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "checkpoint_role": None,
        "prediction_sha256": None,
        **{
            key: binding[key]
            for key in (
                "suite_sha256",
                "case_id",
                "case_sha256",
                "volume_id",
                "volume_files",
                "partition",
                "mask_seed",
                "shape",
                "selection",
                "target_array_rows_sha256",
            )
        },
        "run_metadata_sha256": None,
        "inputs_lock_sha256": None,
        "auxiliary_best_metrics": None,
        "status": "not_run",
        "reason": "No run path supplied.",
        **{key: None for key in _METRIC_FIELDS},
        "snr_gain_over_zero_fill_db": None,
        "actual_missing_fraction": binding["actual_missing_fraction"],
        "expected_target_count": binding["target_count"],
        "expected_sample_count": binding["sample_count"],
        "pretraining_seconds": None,
        "volume_fit_seconds": None,
        **(
            {"prediction_seconds": None, "total_method_seconds": None}
            if include_training_context
            else {}
        ),
        "reconstruction_seconds": None,
        "frozen_inference_seconds": None,
        "evaluation_seconds": None,
        "load_and_verification_seconds": None,
        "summary_re_evaluation_seconds": None,
        "process_max_rss_kib": None,
        "cuda_max_memory_allocated_bytes": None,
        "uncovered_sample_count": None,
        "no_context_query_count": None,
        "null_reasons": {},
        "measurement_scopes": {},
        "metric_comparison": None,
        "native_resources": {},
        "training_resources": {},
        "training_exposure": {},
        "git_and_numerical_mode": {},
    }
    for key, value in row.items():
        if value is None:
            row["null_reasons"][key] = "No completed native measurement available."
    if method in ("zero_fill", "pocs", "drr"):
        for key in (
            "model",
            "parameter_count",
            "amplitude",
            "training_amplitude",
            "checkpoint_path",
            "checkpoint_sha256",
            "pretraining_seconds",
            "volume_fit_seconds",
            "frozen_inference_seconds",
            "cuda_max_memory_allocated_bytes",
            "auxiliary_best_metrics",
        ):
            row["null_reasons"][key] = "Not applicable to this CPU baseline or classical method."
    if method != "relational_trace_graph":
        row["null_reasons"]["no_context_query_count"] = "Only defined for graph queries."
    return row


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _collect_run(
    method: str, path: Path, inputs: C3VolumeRunInputs, interim: Path, binding: dict, row: dict
):
    request = {}
    if (path / "result.json").is_file():
        result = _read_json(path / "result.json")
        request = _read_json(path / "request.json")
        if request.get("preflight") or result.get("preflight"):
            raise ValueError("preflight runs cannot enter the full-budget comparison")
        expected_action = {
            "pocs": "pocs",
            "drr": "drr",
            "siren5d": "siren",
            "ccnet5d": "ccnet-predict",
            "relational_trace_graph": "gnn-predict",
            "nersi": "nersi",
        }[method]
        if request.get("action") != expected_action:
            raise ValueError("outer action is not the selected method prediction")
        if request.get("case_id") != binding["case_id"]:
            raise ValueError("outer request case does not match selected case")
        if result.get("status") != "success":
            row.update(
                status=result.get("status", "failed"),
                reason=result.get("reason", "Action did not complete."),
            )
            return None, []
        declared = result.get("native_run_directory", request.get("native_run_directory"))
        if not declared:
            raise ValueError("successful outer result requires a native_run_directory")
        native = Path(declared)
        if not native.is_absolute():
            native = path / native
        hashes = request.get("input_hashes", {})
        recorded_suite = hashes.get("suite")
        if isinstance(recorded_suite, Mapping):
            recorded_suite = recorded_suite.get("sha256")
        if recorded_suite != binding["suite_sha256"]:
            raise ValueError("outer request suite hash does not match selected suite")
    else:
        native = path
    row["native_run_path"] = str(native.resolve())
    metadata = _read_json(native / "run.json")
    row["run_metadata_sha256"] = file_sha256(native / "run.json")
    if metadata.get("status") != "success":
        row.update(
            status=metadata.get("status", "failed"),
            reason=metadata.get("reason", "Native run did not complete."),
        )
        return None, []
    metrics = _read_json(native / "metrics.json")
    lock = _read_json(native / "inputs.lock.json")
    row["inputs_lock_sha256"] = file_sha256(native / "inputs.lock.json")
    _validate_binding(method, inputs, metadata, metrics, lock)
    declared_config = request.get("native_config", {})
    if (
        method in ("pocs", "drr")
        and declared_config
        and metadata.get(method, {}).get("n_iterations")
        != declared_config.get(method, {}).get("n_iterations")
    ):
        raise ValueError("completed iterations differ from declared budget")
    if method in ("siren5d", "nersi"):
        training = metadata.get("training", {})
        if training.get("steps_completed") != training.get("max_steps"):
            raise ValueError(f"{method} did not complete its declared optimizer budget")
    row.update(
        method_variant=metadata.get("method_variant", metadata["method"]),
        model=metadata.get("model"),
        parameter_count=metadata.get("parameter_count"),
        graph=metadata.get("graph"),
        neighbors_per_relation=metadata.get("graph", {}).get("neighbors_per_relation"),
        prediction_settings={
            key: metadata["prediction"][key]
            for key in ("query_batch_size", "batch_size", "core_shape", "halo_radius", "tile_count")
            if key in metadata.get("prediction", {})
        },
    )
    checkpoint = metadata.get("checkpoint", {})
    checkpoint_path = _checkpoint(method, native, checkpoint, row)
    prediction_path = native / "artifacts" / "prediction.npy"
    prediction_digest = file_sha256(prediction_path)
    if method == "nersi" and metadata.get("prediction", {}).get("sha256") != prediction_digest:
        raise ValueError("NeRSI prediction SHA-256 differs from run metadata")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.dtype.kind != "f":
        raise ValueError("prediction must have a floating physical-amplitude dtype")
    if (
        list(prediction.shape) != binding["shape"]
        or metadata.get("prediction", {}).get("shape") != binding["shape"]
    ):
        raise ValueError("prediction must cover the full selected volume shape")
    no_context = None
    if method == "relational_trace_graph":
        no_context = _validate_query_coverage(native, inputs, metrics)
    started = time.perf_counter()
    rescored = evaluate_c3_volume_prediction(
        prediction,
        inputs.observed_volume,
        interim_dir=interim,
        volume_metadata=inputs.volume_metadata,
    )["evaluation_target"]
    row["summary_re_evaluation_seconds"] = time.perf_counter() - started
    comparison = _compare_metrics(metrics, rescored, prediction.dtype)
    row.update(**rescored, metric_comparison=comparison, no_context_query_count=no_context)
    row["prediction_sha256"] = prediction_digest
    history = _measurements(method, native, metadata, metrics, checkpoint_path, row)
    row.update(status="success", reason=None)
    row["null_reasons"] = {
        key: reason for key, reason in row["null_reasons"].items() if row.get(key) is None
    }
    return prediction, history


def _validate_binding(
    method: str, inputs: C3VolumeRunInputs, metadata: dict, metrics: dict, lock: dict
) -> None:
    if (
        metadata.get("method") != _NATIVE_METHODS[method]
        or metrics.get("method") != _NATIVE_METHODS[method]
    ):
        raise ValueError("run method does not match explicit method mapping")
    if metrics.get("case_id") != inputs.case["case_id"]:
        raise ValueError("run case does not match selected case")
    if method == "nersi" and (
        metadata.get("case_id") != inputs.case["case_id"]
        or metadata.get("volume_id") != inputs.volume_metadata["volume_id"]
        or metrics.get("volume_id") != inputs.volume_metadata["volume_id"]
    ):
        raise ValueError("NeRSI metadata/metrics case or volume does not match selected input")
    for key in ("benchmark_case", "benchmark_volume"):
        for field, expected in inputs.inputs_lock[key].items():
            if lock.get(key, {}).get(field) != expected:
                raise ValueError(f"run {key}.{field} binding differs from selected input")
    source = metadata.get("input", {})
    if (
        source.get("partition") != inputs.case["partition"]
        or source.get("mask", {}).get("random_seed") != inputs.case["mask"]["random_seed"]
    ):
        raise ValueError("run partition or mask seed differs from selected input")
    volume = inputs.volume_metadata
    if method == "relational_trace_graph":
        if (
            source.get("time_samples") != volume["selection"]["time"]
            or source.get("time_s") != inputs.observed_volume.time_s.tolist()
        ):
            raise ValueError("graph prediction time differs from selected volume")
        if metadata.get("prediction", {}).get("layout") != "dense_volume":
            raise ValueError("graph comparison requires a full dense volume prediction")
        if lock["benchmark_volume"].get("array_rows") != inputs.index_table["array_row"].tolist():
            raise ValueError("graph volume row mapping differs from selected input")
    elif source.get("selected_volume", {}).get("selection") != volume["selection"]:
        raise ValueError("run time or spatial selection differs from selected volume")


def _checkpoint(method: str, native: Path, checkpoint: dict, row: dict) -> Path | None:
    if method not in ("siren5d", "nersi", "ccnet5d", "relational_trace_graph"):
        row["checkpoint_role"] = "declared_iterations_completed"
        return None
    role = "fixed_step_final" if method in ("siren5d", "nersi") else "final"
    if checkpoint.get("role") != role:
        raise ValueError("pilot main comparison requires the final checkpoint")
    path = native / checkpoint["artifact"] if "artifact" in checkpoint else Path(checkpoint["path"])
    digest = file_sha256(path)
    recorded_digest = checkpoint.get("sha256", digest)
    if method == "nersi" and "sha256" not in checkpoint:
        raise ValueError("NeRSI final checkpoint metadata requires its saved SHA-256")
    if recorded_digest != digest:
        raise ValueError("checkpoint hash changed since prediction")
    row.update(checkpoint_path=str(path.resolve()), checkpoint_sha256=digest, checkpoint_role=role)
    return path


def _validate_query_coverage(native: Path, inputs: C3VolumeRunInputs, metrics: dict) -> int:
    query = pd.read_parquet(native / "artifacts" / "query_index.parquet")
    required = {"prediction_row", "trace_id", "array_row", "has_observed_context"}
    if not required.issubset(query.columns) or query[list(required)].isna().any().any():
        raise ValueError("query index requires complete mapped IDs and context flags")
    expected = np.sort(
        inputs.observed_volume.array_rows[inputs.observed_volume.evaluation_target_trace_mask]
    )
    ids = query["trace_id"].to_numpy()
    rows = query["array_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(rows), expected) or not np.array_equal(ids, rows):
        raise ValueError("query IDs must cover every target exactly once")
    if not np.array_equal(query["prediction_row"].to_numpy(), np.arange(len(query))):
        raise ValueError("query prediction row order is invalid")
    context = query["has_observed_context"].to_numpy()
    if context.dtype != np.bool_:
        raise ValueError("query context flags must be boolean")
    count = int((~context).sum())
    if (
        metrics.get("zero_context_query_count") != count
        or metrics.get("without_observed_context", {}).get("trace_count") != count
    ):
        raise ValueError("native no-context metrics differ from query index")
    return count


def _compare_metrics(native: dict, actual: dict, dtype: np.dtype) -> dict:
    if (
        native.get("evaluation_domain") != "evaluation_target"
        or native.get("amplitude_domain") != "physical"
    ):
        raise ValueError("native metrics must score physical evaluation-target amplitudes")
    expected = native["evaluation_target"]
    for key in ("trace_count", "sample_count", "snr_status"):
        if expected.get(key) != actual[key]:
            raise ValueError(f"native {key} differs from full-volume re-evaluation")
    # Storage rounding tolerance follows the prediction dtype; no absolute energy floor.
    tolerance = 32.0 * np.finfo(dtype).eps
    differences = {}
    for key in ("reference_energy", "error_energy"):
        value = expected.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"native {key} must be finite")
        differences[key] = actual[key] - value
        if not math.isclose(actual[key], value, rel_tol=tolerance, abs_tol=0.0):
            raise ValueError(f"native {key} differs from full-volume re-evaluation")
    return {
        "status": "matched",
        "prediction_dtype": dtype.name,
        "relative_tolerance": tolerance,
        "absolute_tolerance": 0.0,
        "energy_differences": differences,
    }


def _measurements(
    method: str, native: Path, metadata: dict, metrics: dict, checkpoint: Path | None, row: dict
) -> list:
    resources = {**metadata.get("resources", {}), **metadata.get("timings", {})}
    row["native_resources"] = {
        **resources,
        "prediction_diagnostics": metadata.get("prediction", {}).get("diagnostics"),
    }
    row["amplitude"] = metadata.get("amplitude")
    row["uncovered_sample_count"] = metrics.get("uncovered_sample_count")
    row["git_and_numerical_mode"] = {
        key: metadata[key]
        for key in (
            "git_commit",
            "git_worktree_dirty",
            "device",
            "random_seed",
            "training_random_seed",
            "python_version",
            "numpy_version",
            "torch_version",
        )
        if key in metadata
    }
    for key in (
        "reconstruction_seconds",
        "evaluation_seconds",
        "load_and_verification_seconds",
        "process_max_rss_kib",
        "cuda_max_memory_allocated_bytes",
    ):
        row[key] = resources.get(key)
        if row[key] is not None:
            row["measurement_scopes"][key] = (
                "whole native process lifetime peak"
                if not key.endswith("seconds")
                else f"native {key.removesuffix('_seconds')} stage"
            )
    row["frozen_inference_seconds"] = resources.get("prediction_seconds")
    if "prediction_seconds" in row:
        row["prediction_seconds"] = resources.get("prediction_seconds")
    row["measurement_scopes"]["frozen_inference_seconds"] = (
        "native prediction stage including assembly and observed reinsertion when measured"
    )
    row["measurement_scopes"]["summary_re_evaluation_seconds"] = (
        "collector dense target-only evaluation, including reference reads"
    )
    row["training_exposure"] = metadata.get("training", {})
    if method in ("siren5d", "nersi"):
        row["volume_fit_seconds"] = resources.get("training_seconds")
        row["measurement_scopes"]["volume_fit_seconds"] = (
            "target volume observed-only optimizer stage"
        )
        if (
            "prediction_seconds" in row
            and row["volume_fit_seconds"] is not None
            and row["prediction_seconds"] is not None
        ):
            row["total_method_seconds"] = row["volume_fit_seconds"] + row["prediction_seconds"]
            row["measurement_scopes"]["total_method_seconds"] = (
                "per-volume observed-only training plus full-volume prediction"
            )
        return metrics.get("training", {}).get("history", [])
    if method in ("ccnet5d", "relational_trace_graph") and checkpoint is not None:
        training = checkpoint.parent.parent
        row["training_run_path"] = str(training.resolve())
        if (training / "run.json").is_file() and (training / "metrics.json").is_file():
            train_metadata = _read_json(training / "run.json")
            train_metrics = _read_json(training / "metrics.json")
            if "final" not in train_metadata.get("checkpoints", {}):
                return []
            _validate_training_budget(method, train_metadata, train_metrics, metadata)
            train_resources = {
                **train_metadata.get("resources", {}),
                **train_metadata.get("timings", {}),
            }
            row["training_resources"] = train_resources
            row["training_amplitude"] = train_metadata.get("amplitude")
            row["auxiliary_best_metrics"] = {
                key: train_metrics[key]
                for key in (
                    "best_step",
                    "best_epoch",
                    "best_selection_metrics",
                    "best_validation_metrics",
                )
                if key in train_metrics
            }
            row["training_exposure"] = {
                "training": train_metadata.get("training"),
                "input": train_metadata.get("input"),
                "supervision": train_metadata.get("supervision"),
                "patches": train_metadata.get("patches"),
                "training_data": train_metadata.get("training_data"),
                "training_mask": train_metadata.get("training_mask"),
                "metrics_training": {
                    key: train_metrics[key]
                    for key in (
                        "steps_completed",
                        "episodes_completed",
                        "query_count",
                        "sample_count",
                    )
                    if key in train_metrics
                },
            }
            if method == "ccnet5d":
                training_lock = _read_json(training / "inputs.lock.json")
                regions = training_lock["source_inputs_lock"]["regions"]
                row["training_exposure"]["source_regions"] = regions
                row["training_exposure"]["region_trace_counts"] = {
                    name: math.prod(region["shape"][1:]) for name, region in regions.items()
                }
                row["training_exposure"]["region_trace_count_source"] = (
                    "product of spatial dimensions in native source_inputs_lock; "
                    "independent of outer request counts"
                )
            timing_key = (
                "training_and_selection_seconds" if method == "ccnet5d" else "elapsed_seconds"
            )
            row["pretraining_seconds"] = train_resources.get(timing_key)
            row["measurement_scopes"]["pretraining_seconds"] = (
                "training and selection stages"
                if method == "ccnet5d"
                else "whole native training action including input and validation"
            )
            return train_metrics.get("training_history", [])
    return []


def _validate_training_budget(method: str, metadata: dict, metrics: dict, prediction: dict) -> None:
    if metadata.get("status") != "success":
        raise ValueError("final checkpoint training action did not complete successfully")
    training = metadata.get("training", {})
    checkpoint = prediction["checkpoint"]
    if method == "ccnet5d":
        if (
            metrics.get("epochs_completed") != training.get("max_epochs")
            or checkpoint.get("epoch") != metrics.get("epochs_completed")
            or checkpoint.get("global_step") != metrics.get("steps_completed")
        ):
            raise ValueError("CCNet final checkpoint differs from completed training budget")
    elif metrics.get("steps_completed") != training.get("max_steps") or checkpoint.get(
        "step"
    ) != metrics.get("steps_completed"):
        raise ValueError("GNN final checkpoint differs from completed training budget")


def _collect_baseline(
    path: Path, inputs: C3VolumeRunInputs, interim: Path, binding: dict, row: dict
) -> None:
    native = path
    if (path / "result.json").is_file():
        result = _read_json(path / "result.json")
        request = _read_json(path / "request.json")
        if (
            result.get("status") != "success"
            or request.get("action") != "zero-fill"
            or request.get("case_id") != binding["case_id"]
        ):
            raise ValueError("explicit zero-fill baseline must be a successful matching action")
        native = Path(result["native_run_directory"])
        if not native.is_absolute():
            native = path / native
    metrics = _read_json(native / "metrics.json")
    metadata = _read_json(native / "run.json")
    lock = _read_json(native / "inputs.lock.json")
    if (
        metrics.get("method") != "zero_fill"
        or metrics.get("case_id") != binding["case_id"]
        or metrics.get("scope") != "full_volume"
        or metrics.get("suite_sha256") != binding["suite_sha256"]
        or lock != inputs.inputs_lock
    ):
        raise ValueError("explicit zero-fill baseline binding differs from selected full volume")
    prediction_path = native / "artifacts" / "prediction.npy"
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    observed = inputs.observed_volume
    if prediction.dtype != observed.values.dtype or not np.array_equal(prediction, observed.values):
        raise ValueError(
            "saved zero-fill prediction differs from verified zero-filled observed input"
        )
    started = time.perf_counter()
    actual = evaluate_c3_volume_prediction(
        prediction, observed, interim_dir=interim, volume_metadata=inputs.volume_metadata
    )["evaluation_target"]
    row["summary_re_evaluation_seconds"] += time.perf_counter() - started
    row["metric_comparison"] = _compare_metrics(metrics, actual, prediction.dtype)
    row.update(**actual, prediction_sha256=file_sha256(prediction_path))
    row["native_run_path"] = str(native.resolve())
    row["run_metadata_sha256"] = file_sha256(native / "run.json")
    row["inputs_lock_sha256"] = file_sha256(native / "inputs.lock.json")
    _measurements("zero_fill", native, metadata, metrics, None, row)
    row["measurement_scopes"]["summary_re_evaluation_seconds"] = (
        "collector independent zero-fill and saved baseline target-only evaluations"
    )
