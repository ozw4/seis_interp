"""Audit completed benchmark graph runs without changing their saved artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_benchmark_inputs import (
    load_c3_benchmark_graph_domain,
    load_c3_benchmark_volume_inputs,
)
from seis_interp.data.c3_benchmark_suite import (
    c3_suite_case,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.relational_trace_graph_run_inputs import (
    validate_trace_graph_checkpoint_provenance,
)
from seis_interp.data.trace_graph_prediction_store import (
    trace_graph_query_index,
    trace_graph_query_positions,
)
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_training_config,
)
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph

ENERGY_RTOL = 1e-6
ENERGY_ATOL = 1e-12
CPU_THRESHOLDS = {
    "normalized_difference_rmse": 1e-4,
    "normalized_difference_max_abs": 1e-3,
    "physical_relative_l2": 1e-3,
}
RUN_FILES = (
    "run.json",
    "metrics.json",
    "config.resolved.yaml",
    "inputs.lock.json",
    "artifacts/prediction.npy",
    "artifacts/query_index.parquet",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _query_index(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path)
    required = {"prediction_row", "trace_id", "array_row", "has_observed_context"}
    _require(required.issubset(table.columns), "query index is missing required columns")
    _require(len(table) > 0, "query index must not be empty")
    for name in ("prediction_row", "trace_id", "array_row"):
        values = table[name]
        _require(
            pd.api.types.is_integer_dtype(values.dtype)
            and not values.isna().any()
            and values.is_unique
            and bool((values >= 0).all()),
            f"query index {name} must contain unique nonnegative integers",
        )
    _require(
        np.array_equal(table["prediction_row"].to_numpy(), np.arange(len(table))),
        "query index prediction_row must preserve consecutive saved prediction order",
    )
    _require(
        table["has_observed_context"].dtype == np.bool_,
        "query context flags must be boolean",
    )
    return table


def _native_batches(table: pd.DataFrame, batch_size: int) -> list[dict]:
    _require(
        isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0,
        "native prediction query_batch_size must be a positive integer",
    )
    count = math.ceil(len(table) / batch_size)
    indices = sorted({0, count // 2, count - 1})
    return [
        {
            "batch_index": index,
            "prediction_rows": [index * batch_size, min((index + 1) * batch_size, len(table))],
            "query_trace_ids": table["trace_id"]
            .iloc[index * batch_size : (index + 1) * batch_size]
            .tolist(),
        }
        for index in indices
    ]


def _energy_comparison(actual: dict, expected: dict) -> dict:
    counts = all(actual[name] == expected[name] for name in ("trace_count", "sample_count"))
    energies = {
        name: bool(np.isclose(actual[name], expected[name], rtol=ENERGY_RTOL, atol=ENERGY_ATOL))
        for name in ("reference_energy", "error_energy")
    }
    return {
        "status": "passed" if counts and all(energies.values()) else "failed_tolerance",
        "counts_equal": counts,
        "energy_checks": energies,
        "rtol": ENERGY_RTOL,
        "atol": ENERGY_ATOL,
        "actual": actual,
        "expected": expected,
    }


def _volume_positions(array_rows: np.ndarray, selected_rows: np.ndarray) -> np.ndarray:
    flattened = np.asarray(array_rows).ravel()
    available = np.flatnonzero(flattened >= 0)
    index = pd.Index(flattened[available])
    _require(index.is_unique, "volume array rows must be unique")
    positions = index.get_indexer(selected_rows)
    _require(bool(np.all(positions >= 0)), "query array rows must belong to the fixed volume")
    return available[positions]


def _score_saved(directory, table, domain, inputs, interim):
    ids = table["trace_id"].to_numpy(dtype=np.int64)
    context = table["has_observed_context"].to_numpy(dtype=bool)
    trace_graph_query_positions(domain, ids, require_complete=True)
    expected_index = trace_graph_query_index(
        domain, query_trace_ids=ids, has_observed_context=context
    )
    pd.testing.assert_frame_equal(table, expected_index)
    prediction = np.load(directory / "artifacts/prediction.npy", mmap_mode="r", allow_pickle=False)
    _require(
        prediction.dtype == np.float32
        and tuple(prediction.shape) == tuple(inputs.volume_metadata["shape"]),
        "saved prediction must be physical float32 with the fixed dense volume shape",
    )
    _require(bool(np.isfinite(prediction).all()), "saved prediction must be entirely finite")
    columns = _volume_positions(
        inputs.observed_volume.array_rows, table["array_row"].to_numpy(dtype=np.int64)
    )
    query_values = prediction.reshape(len(domain.time_s), -1)[:, columns].T
    native = evaluate_trace_graph_prediction(
        query_values,
        domain,
        query_trace_ids=ids,
        has_observed_context=context,
    )
    dense = evaluate_c3_volume_prediction(
        prediction,
        inputs.observed_volume,
        interim_dir=interim,
        volume_metadata=inputs.volume_metadata,
    )
    _require(dense["observed_max_abs_error"] == 0.0, "observed reinsertion must be exact")
    _require(
        native["evaluation_target"]["trace_count"] == int(domain.query_mask.sum())
        and native["evaluation_target"]["sample_count"]
        == int(domain.query_mask.sum()) * len(domain.time_s),
        "scoring must cover every target and time sample",
    )
    return {
        "native_order_metrics": native,
        "dense_order_metrics": dense,
        "dense_versus_native_order": _energy_comparison(
            dense["evaluation_target"], native["evaluation_target"]
        ),
        "all_prediction_samples_finite": True,
        "query_index_matches_complete_domain": True,
        "observed_reinsertion_max_abs_error": 0.0,
    }, prediction


def _checkpoint_records(training, prediction, domain, metadata, metrics):
    config = load_resolved_config(training / "config.resolved.yaml")
    model_config, settings, options = validate_relational_trace_graph_training_config(config)
    with torch.random.fork_rng(devices=[]):
        model_config = RelationalTraceGraphInterpolator(**model_config).constructor_config()
        checkpoints = {
            label: load_relational_trace_graph_checkpoint(
                training / f"artifacts/{label}.pt", device="cpu"
            )
            for label in ("best", "final")
        }
    provenance = _read(training / "inputs.lock.json")
    for label, loaded in checkpoints.items():
        role = "final" if label == "final" else "best_validation"
        expected_step = (
            metrics["training"]["steps_completed"]
            if label == "final"
            else metrics["training"]["best_step"]
        )
        _require(loaded.checkpoint_role == role, f"{label} checkpoint role differs")
        _require(loaded.global_step == expected_step, f"{label} checkpoint step differs")
        _require(loaded.training_provenance == provenance, "checkpoint training provenance differs")
        validate_trace_graph_checkpoint_provenance(loaded.training_provenance, domain)
        _require(
            loaded.model.constructor_config() == model_config, "checkpoint constructor differs"
        )
        _require(loaded.graph_settings == settings, "checkpoint graph settings differ")
        _require(
            loaded.training_mask == config["training_mask"], "checkpoint training mask differs"
        )
        _require(
            loaded.preprocessing.amplitude_scale
            == metadata["training"]["amplitude"]["amplitude_scale"],
            "checkpoint amplitude scale differs from native training metadata",
        )
        _require(loaded.training_random_seed == options["random_seed"], "checkpoint seed differs")
        _require(
            bool(np.array_equal(loaded.preprocessing.time_s, domain.time_s)),
            "checkpoint time samples differ from the fixed domain",
        )
        _require(
            all(bool(torch.isfinite(value).all()) for value in loaded.model.state_dict().values()),
            "checkpoint weights must be finite",
        )
        _require(
            loaded.selection_metrics == metrics["training"][f"{label}_validation_metrics"],
            f"{label} checkpoint selection metrics differ from training records",
        )
    final = checkpoints["final"]
    _require(
        checkpoints["best"].preprocessing == final.preprocessing,
        "best and final fixed preprocessing differs",
    )
    _require(
        final.preprocessing.amplitude_scale
        == metadata["prediction"]["amplitude"]["amplitude_scale"],
        "frozen prediction amplitude scale differs from its checkpoint",
    )
    _require(
        final.global_step == options["max_steps"], "final checkpoint did not complete its budget"
    )
    record = metadata["prediction"]["checkpoint"]
    _require(
        record["role"] == "final" and record["step"] == final.global_step,
        "frozen role/step differs",
    )
    _require(
        Path(record["path"]).resolve() == (training / "artifacts/final.pt").resolve()
        and record["sha256"] == file_sha256(training / "artifacts/final.pt"),
        "frozen prediction must bind the exact final checkpoint path and hash",
    )
    _require(record["training_provenance"] == provenance, "frozen checkpoint provenance differs")
    prediction_lock = _read(prediction / "inputs.lock.json")
    _require(prediction_lock.get("checkpoint") == record, "frozen input lock checkpoint differs")
    _require(
        all(prediction_lock.get(key) == value for key, value in domain.inputs_lock.items()),
        "frozen input lock differs from the verified benchmark domain",
    )
    best_state, final_state = (checkpoints[name].model.state_dict() for name in ("best", "final"))
    return checkpoints, {
        "status": "passed",
        "final_step": final.global_step,
        "best_step": checkpoints["best"].global_step,
        "final_checkpoint_sha256": record["sha256"],
        "training_provenance_and_fixed_preprocessing_verified": True,
        "fixed_preprocessing": json.loads(json.dumps(asdict(final.preprocessing))),
        "best_and_final_weights_equal": all(
            torch.equal(best_state[k], final_state[k]) for k in best_state
        ),
        "best_and_final_equality_required": False,
    }


def _cpu_restoration(loaded, domain, inputs, table, saved, request):
    scale = float(loaded.preprocessing.amplitude_scale)
    _require(math.isfinite(scale) and scale > 0, "checkpoint amplitude scale must be positive")
    observed_columns = _volume_positions(
        inputs.observed_volume.array_rows, domain.array_rows[domain.observed_mask]
    )
    observed = np.ascontiguousarray(
        inputs.observed_volume.values.reshape(len(domain.time_s), -1)[:, observed_columns].T
    )
    difference_energy = reference_energy = normalized_energy = max_normalized = 0.0
    sample_count = 0
    records = []
    for batch in request["cpu_restoration_batches"]:
        start, stop = batch["prediction_rows"]
        ids = np.asarray(batch["query_trace_ids"], dtype=np.int64)
        predicted = predict_relational_trace_graph(
            loaded.model,
            domain,
            loaded.preprocessing,
            graph_settings=loaded.graph_settings,
            query_trace_ids=ids,
            query_batch_size=request["native_query_batch_size"],
            observed_waveforms=observed,
            device="cpu",
            measure_resources=False,
        )
        _require(np.array_equal(predicted.query_trace_ids, ids), "CPU prediction order differs")
        _require(
            np.array_equal(
                predicted.has_observed_context, table["has_observed_context"].iloc[start:stop]
            ),
            "CPU restored graph context differs from saved prediction",
        )
        columns = _volume_positions(
            inputs.observed_volume.array_rows,
            table["array_row"].iloc[start:stop].to_numpy(dtype=np.int64),
        )
        expected = np.asarray(saved.reshape(len(domain.time_s), -1)[:, columns].T, dtype=np.float64)
        values = np.asarray(predicted.prediction, dtype=np.float64)
        _require(
            values.shape == expected.shape and np.isfinite(values).all(),
            "CPU predictions must be finite and aligned",
        )
        difference = values - expected
        normalized = difference / scale
        difference_energy += float(np.sum(difference * difference, dtype=np.float64))
        reference_energy += float(np.sum(expected * expected, dtype=np.float64))
        normalized_energy += float(np.sum(normalized * normalized, dtype=np.float64))
        max_normalized = max(max_normalized, float(np.abs(normalized).max()))
        sample_count += values.size
        records.append({**batch, "has_observed_context": predicted.has_observed_context.tolist()})
    relative_l2 = (
        math.sqrt(difference_energy / reference_energy)
        if reference_energy > 0
        else 0.0
        if difference_energy == 0
        else None
    )
    measured = {
        "normalized_difference_rmse": math.sqrt(normalized_energy / sample_count),
        "normalized_difference_max_abs": max_normalized,
        "physical_relative_l2": relative_l2,
    }
    decisions = {
        name: value is not None and math.isfinite(value) and value <= CPU_THRESHOLDS[name]
        for name, value in measured.items()
    }
    return {
        "status": "passed" if all(decisions.values()) else "failed_tolerance",
        "metrics": measured,
        "physical_relative_l2_null_reason": "zero_saved_prediction_energy"
        if relative_l2 is None
        else None,
        "thresholds": dict(CPU_THRESHOLDS),
        "normalized_difference_definition": (
            "(CPU physical - saved physical) / checkpoint amplitude_scale"
        ),
        "threshold_decisions": decisions,
        "sample_count": sample_count,
        "query_count": sample_count // len(domain.time_s),
        "batches": records,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "target_truth_passed_to_forward": False,
        "saved_checkpoint_preprocessing_used_without_refit": True,
        "diagnostic_scope": (
            "CPU restoration on first/central/last complete native query batches; "
            "native GPU numerical execution differs. Fixed engineering criteria, "
            "not a full-target bit-equality guarantee or the separate cross-run energy tolerance."
        ),
    }


def audit_c3_proposed_gnn(
    *,
    suite: Path,
    case: str,
    training_run: Path,
    prediction_run: Path,
    output: Path,
    expected_suite_sha256: str,
) -> dict:
    """Create a new immutable audit; failed numerical criteria remain explicit outcomes."""
    suite, training_run, prediction_run, output = (
        Path(value).resolve() for value in (suite, training_run, prediction_run, output)
    )
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    _require(
        bool(re.fullmatch(r"[0-9a-fA-F]{64}", expected_suite_sha256)),
        "expected suite SHA256 must contain 64 hexadecimal digits",
    )
    _require(bool(case.strip()), "case must not be empty")
    _require(training_run != prediction_run, "training and frozen prediction runs must be distinct")
    _require(
        not any(output.is_relative_to(path) for path in (suite, training_run, prediction_run)),
        "audit output must be outside the immutable input directories",
    )
    suite_hash = file_sha256(suite / "benchmark_suite.json")
    _require(
        suite_hash == expected_suite_sha256.lower(), "suite SHA256 does not match the declaration"
    )
    declared_suite = _read(suite / "benchmark_suite.json")
    declared_case = c3_suite_case(declared_suite, case)
    _require(
        declared_case["partition"] == "validation",
        "this audit must not evaluate the test partition",
    )
    directories = {"training": training_run, "prediction": prediction_run}
    metadata = {name: _read(path / "run.json") for name, path in directories.items()}
    for name, record in metadata.items():
        _require(
            record.get("status") == "success", f"{name} native run must be completed successfully"
        )
        _require(
            record.get("prediction", {}).get("layout") == "dense_volume",
            "audit requires saved dense-volume predictions",
        )
    _require(
        metadata["training"]["prediction"].get("checkpoint_role") == "best_validation",
        "training saved prediction must identify its best-validation role",
    )
    indices = {
        name: _query_index(path / "artifacts/query_index.parquet")
        for name, path in directories.items()
    }
    batch_size = metadata["prediction"]["prediction"]["query_batch_size"]
    batches = _native_batches(indices["prediction"], batch_size)
    implementation_files = sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
    files = [suite / "benchmark_suite.json", *implementation_files]
    files.extend(path / name for path in directories.values() for name in RUN_FILES)
    files.extend(training_run / f"artifacts/{label}.pt" for label in ("best", "final"))
    hashes = {str(path): file_sha256(path) for path in files}
    request = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": str(suite),
        "case_id": case,
        "training_native_run": str(training_run),
        "prediction_native_run": str(prediction_run),
        "expected_suite_sha256": expected_suite_sha256.lower(),
        "source_sha256": hashes,
        "implementation_source_count": len(implementation_files),
        "implementation_source_scope": (
            "all Python sources in the seis_interp package at audit time; "
            "training-time source identity is a separate run provenance record"
        ),
        "suite_declared_file_bindings": declared_suite["files"],
        "suite_binding_scope": (
            "public loaders verify the common inputs and selected validation case"
        ),
        "native_query_batch_size": batch_size,
        "cpu_restoration_selection": (
            "first, central floor(batch_count/2), last native batches; deduplicate batch indices"
        ),
        "cpu_restoration_batches": batches,
        "cpu_restoration_thresholds": dict(CPU_THRESHOLDS),
        "cross_run_energy_tolerance": {"rtol": ENERGY_RTOL, "atol": ENERGY_ATOL},
        "thresholds_fixed_before_forward": True,
        "diagnostic_scope": (
            "CPU subset restoration versus native execution; engineering criteria "
            "separate from whole-target cross-run energy agreement"
        ),
        "device": "cpu",
        "cpu_numerical_settings": {
            "torch_version": str(torch.__version__),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "new_training": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "request.json", request)
    report = {"status": "failed", "request_sha256": file_sha256(output / "request.json")}
    try:
        manifest = load_c3_benchmark_input_manifest(suite, case_id=case)
        entry = c3_suite_case(manifest, case)
        _require(
            entry["partition"] == "validation", "this audit must not evaluate the test partition"
        )
        volume_dir = suite_path(suite, entry["volume_dir"])
        domain = load_c3_benchmark_graph_domain(suite, case, volume_dir=volume_dir)
        inputs = load_c3_benchmark_volume_inputs(suite, case)
        _require(metadata["prediction"]["input"]["case_id"] == case, "prediction run case differs")
        metrics = {name: _read(path / "metrics.json") for name, path in directories.items()}
        loaded, checkpoint_report = _checkpoint_records(
            training_run, prediction_run, domain, metadata, metrics
        )
        report["checkpoint_integrity"] = checkpoint_report
        scores = {}
        final_saved = None
        for name, directory in directories.items():
            scores[name], saved = _score_saved(
                directory, indices[name], domain, inputs, suite_path(suite, manifest["interim"])
            )
            if name == "prediction":
                final_saved = saved
        report["saved_output_scores"] = scores
        native = scores["prediction"]["native_order_metrics"]
        native_exact = all(native[key] == metrics["prediction"][key] for key in native)
        comparisons = {
            "frozen_native_order_exact_reconciliation": {
                "status": "passed" if native_exact else "failed_exact_match"
            },
            "frozen_vs_training_final": _energy_comparison(
                native["evaluation_target"],
                metrics["training"]["final_validation_metrics"]["evaluation_target"],
            ),
            "auxiliary_best_vs_training_best": _energy_comparison(
                scores["training"]["native_order_metrics"]["evaluation_target"],
                metrics["training"]["best_validation_metrics"]["evaluation_target"],
            ),
        }
        report["metric_comparisons"] = comparisons
        report["verified_domain_binding"] = {
            "status": "passed",
            "partition": domain.inputs_lock["partition"],
            "case_id": case,
            "query_count": int(domain.query_mask.sum()),
            "observed_count": int(domain.observed_mask.sum()),
            "time_sample_count": len(domain.time_s),
            "native_input_lock_sha256": hashes[str(prediction_run / "inputs.lock.json")],
        }
        _require(
            all(file_sha256(Path(path)) == value for path, value in hashes.items()),
            "source changed before CPU restoration",
        )
        # Read back the committed request so the actual forward uses its fixed query IDs.
        committed = _read(output / "request.json")
        _require(committed == request, "audit request changed before CPU restoration")
        with torch.random.fork_rng(devices=[]):
            cpu = _cpu_restoration(
                loaded["final"], domain, inputs, indices["prediction"], final_saved, committed
            )
        report["cpu_checkpoint_restoration"] = cpu
        checks_pass = (
            all(value["status"] == "passed" for value in comparisons.values())
            and all(
                score["dense_versus_native_order"]["status"] == "passed"
                for score in scores.values()
            )
            and cpu["status"] == "passed"
        )
        report["status"] = "success" if checks_pass else "failed_checks"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    try:
        unchanged = all(file_sha256(Path(path)) == value for path, value in hashes.items())
        request_unchanged = file_sha256(output / "request.json") == report["request_sha256"]
    except OSError:
        unchanged = request_unchanged = False
    report["request_unchanged"] = request_unchanged
    report["source_hashes_unchanged"] = unchanged
    if not unchanged or not request_unchanged:
        report["status"] = "failed_source_changed"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(output / "result.json", report)
    return report
