"""Rescore completed CCNet runs and restore fixed native tiles on CPU."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_volume_inputs
from seis_interp.data.c3_benchmark_suite import (
    c3_suite_case,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.data.ccnet5d_benchmark_inputs import validate_ccnet5d_benchmark_provenance
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.processing.ccnet5d_tiles import iter_ccnet5d_tiles
from seis_interp.training.ccnet5d_checkpoints import load_ccnet5d_checkpoint

ENERGY_RTOL = 1e-6
ENERGY_ATOL = 1e-12
CPU_THRESHOLDS = {
    "normalized_difference_rmse": 1e-4,
    "normalized_difference_max_abs": 1e-3,
    "physical_relative_l2": 1e-3,
}
RUN_FILES = ("run.json", "metrics.json", "config.resolved.yaml", "inputs.lock.json")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tile_record(index, tile):
    return {
        "tile_index": index,
        **{
            key: [[part.start, part.stop] for part in getattr(tile, key)]
            for key in ("input_slices", "core_slices", "local_core_slices")
        },
    }


def restore_ccnet5d_cpu_tiles(loaded, observed, prediction, tiles):
    """Compare missing samples of preselected full native tiles, without truth reads."""
    source = np.where(observed.observed_trace_mask[None], observed.values, 0)
    source = source / loaded.amplitude_rms
    model = loaded.model.cpu().eval()
    squared_difference = 0.0
    reference_energy = 0.0
    maximum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for tile in tiles:
            block = np.ascontiguousarray(source[tile.input_slices], dtype=np.float32)
            output = model(torch.from_numpy(block)[None, None])[0, 0]
            physical = output[tile.local_core_slices].numpy() * loaded.amplitude_rms
            mask = observed.evaluation_target_trace_mask[tile.core_slices[1:]]
            actual = physical[:, mask].astype(np.float64)
            saved = prediction[tile.core_slices][:, mask].astype(np.float64)
            _require(np.isfinite(actual).all(), "CPU reconstruction must be finite")
            error = actual - saved
            squared_difference += float(np.sum(error**2, dtype=np.float64))
            reference_energy += float(np.sum(saved**2, dtype=np.float64))
            maximum = max(maximum, float(np.max(np.abs(error), initial=0.0)))
            sample_count += error.size
    _require(sample_count > 0, "fixed CPU tiles must contain evaluation targets")
    relative = (
        math.sqrt(squared_difference / reference_energy)
        if reference_energy > 0
        else (0.0 if squared_difference == 0 else None)
    )
    measured = {
        "normalized_difference_rmse": math.sqrt(squared_difference / sample_count)
        / loaded.amplitude_rms,
        "normalized_difference_max_abs": maximum / loaded.amplitude_rms,
        "physical_relative_l2": relative,
    }
    checks = {
        name: value is not None and value <= CPU_THRESHOLDS[name]
        for name, value in measured.items()
    }
    return {
        "status": "passed" if all(checks.values()) else "failed_tolerance",
        "sample_count": sample_count,
        "measured": measured,
        "thresholds": CPU_THRESHOLDS,
        "checks": checks,
        "target_truth_read": False,
        "normalization": "checkpoint_fit_region_global_rms",
    }


def _verify_checkpoint(training, prediction, inputs):
    train_run = _read(training / "run.json")
    infer_run = _read(prediction / "run.json")
    metrics = _read(training / "metrics.json")
    provenance = _read(training / "inputs.lock.json")
    config = load_resolved_config(training / "config.resolved.yaml")
    _require(
        file_sha256(training / "artifacts/patch_plan.json") == provenance["patch_plan_sha256"],
        "training patch plan SHA-256 differs",
    )
    for region in ("fit", "selection"):
        _require(
            provenance["source_inputs_lock"]["regions"][region]["selection"]
            == config["supervision"][f"{region}_region"],
            "training region differs from checkpoint provenance",
        )
    for record in (train_run, infer_run):
        _require(record["status"] == "success", "audit requires completed successful runs")
        _require(record["method"] == "ccnet5d", "audit requires CCNet runs")
    expected_steps = config["training"]["max_epochs"] * math.ceil(
        config["patches"]["fit_count"] / config["training"]["batch_size"]
    )
    _require(metrics["steps_completed"] == expected_steps, "training budget is incomplete")
    _require(
        metrics["epochs_completed"] == config["training"]["max_epochs"],
        "training epochs are incomplete",
    )
    final = None
    for label, role in (("best", "best_selection"), ("final", "final")):
        with torch.random.fork_rng(devices=[]):
            loaded = load_ccnet5d_checkpoint(training / f"artifacts/{label}.pt", device="cpu")
        _require(loaded.checkpoint_role == role, "checkpoint role differs")
        _require(loaded.training_provenance == provenance, "checkpoint provenance differs")
        validate_ccnet5d_benchmark_provenance(loaded.training_provenance, inputs)
        _require(
            loaded.model.constructor_config()
            == {key: value for key, value in config["model"].items() if key != "name"},
            "checkpoint model differs from training config",
        )
        _require(
            all(bool(torch.isfinite(value).all()) for value in loaded.model.state_dict().values()),
            "checkpoint weights must be finite",
        )
        _require(
            loaded.amplitude_rms == train_run["amplitude"]["amplitude_rms"],
            "checkpoint RMS differs from training record",
        )
        _require(
            loaded.selection_metrics == metrics[f"{label}_selection_metrics"],
            "checkpoint selection metrics differ",
        )
        step = metrics["best_step"] if label == "best" else metrics["steps_completed"]
        epoch = metrics["best_epoch"] if label == "best" else metrics["epochs_completed"]
        _require(loaded.global_step == step and loaded.epoch == epoch, "checkpoint step differs")
        if label == "final":
            final = loaded
    checkpoint_sha = file_sha256(training / "artifacts/final.pt")
    prediction_lock = _read(prediction / "inputs.lock.json")
    _require(
        {key: value for key, value in prediction_lock.items() if key != "checkpoint"}
        == inputs.inputs_lock,
        "prediction input lock differs from verified volume",
    )
    for record in (infer_run, prediction_lock):
        _require(record["checkpoint"]["sha256"] == checkpoint_sha, "prediction checkpoint differs")
        _require(
            record["checkpoint"]["training_provenance"] == provenance,
            "prediction provenance differs",
        )
    _require(infer_run["checkpoint"]["role"] == "final", "prediction must use final checkpoint")
    _require(
        infer_run["checkpoint"]["global_step"] == final.global_step,
        "prediction checkpoint step differs",
    )
    prediction_config = load_resolved_config(prediction / "config.resolved.yaml")
    _require(
        prediction_config["prediction"]["core_shape"] == infer_run["prediction"]["core_shape"],
        "prediction core shape differs from its config",
    )
    _require(
        infer_run["prediction"]["halo_radius"] == final.model.halo_radius,
        "prediction halo differs from checkpoint",
    )
    return final, infer_run


def audit_c3_ccnet5d(*, suite, case, training_run, prediction_run, output, expected_suite_sha256):
    """Verify complete physical outputs and fixed CPU tolerances in a new directory."""
    suite, training, prediction, output = map(Path, (suite, training_run, prediction_run, output))
    _require(not output.exists(), "audit output directory must not exist")
    manifest_path = suite / "benchmark_suite.json"
    _require(file_sha256(manifest_path) == expected_suite_sha256, "suite SHA-256 differs")
    manifest = load_c3_benchmark_input_manifest(suite, case_id=case)
    _require(
        c3_suite_case(manifest, case)["partition"] == "validation",
        "audit is restricted to validation",
    )
    inputs = load_c3_benchmark_volume_inputs(suite, case_id=case)
    _require(inputs.case["partition"] == "validation", "audit is restricted to validation")
    files = [manifest_path]
    files.extend(
        training / name
        for name in (
            *RUN_FILES,
            "artifacts/best.pt",
            "artifacts/final.pt",
            "artifacts/patch_plan.json",
        )
    )
    files.extend(prediction / name for name in (*RUN_FILES, "artifacts/prediction.npy"))
    files.extend(sorted(Path(__file__).parents[1].rglob("*.py")))
    files.extend(suite_path(suite, item["path"]) for item in manifest["files"])
    before = {str(path): file_sha256(path) for path in files}
    _require(
        all(
            before[str(suite_path(suite, item["path"]))] == item["sha256"]
            for item in manifest["files"]
        ),
        "suite referenced file SHA-256 differs",
    )
    loaded, metadata = _verify_checkpoint(training, prediction, inputs)
    _require(metadata["case_id"] == case, "prediction case differs")
    _require(metadata["volume_id"] == inputs.volume_metadata["volume_id"], "volume differs")
    observed = inputs.observed_volume
    values = np.load(prediction / "artifacts/prediction.npy", mmap_mode="r", allow_pickle=False)
    _require(
        values.dtype == np.float32 and values.shape == observed.values.shape, "invalid shape/dtype"
    )
    _require(np.isfinite(values).all(), "saved prediction must be finite")
    tiles = list(
        iter_ccnet5d_tiles(
            values.shape, metadata["prediction"]["core_shape"], halo_radius=loaded.model.halo_radius
        )
    )
    _require(len(tiles) == metadata["prediction"]["tile_count"], "native tile count differs")
    selected = sorted({0, len(tiles) // 2, len(tiles) - 1})
    output.mkdir(parents=True)
    _write(
        output / "plan.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "first_middle_last_complete_native_tiles_before_CPU_forward",
            "tiles": [_tile_record(index, tiles[index]) for index in selected],
            "cpu_thresholds": CPU_THRESHOLDS,
            "energy_tolerances": {"rtol": ENERGY_RTOL, "atol": ENERGY_ATOL},
            "input_sha256": before,
            "implementation_scope": (
                "imported_package_at_audit_time_training_source_identity_recorded_separately"
            ),
            "cpu_numerical_settings": {
                "torch_version": str(torch.__version__),
                "torch_num_threads": torch.get_num_threads(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        },
    )
    cpu = restore_ccnet5d_cpu_tiles(loaded, observed, values, [tiles[index] for index in selected])
    score = evaluate_c3_volume_prediction(
        values,
        observed,
        interim_dir=suite_path(suite, manifest["interim"]),
        volume_metadata=inputs.volume_metadata,
    )
    native = _read(prediction / "metrics.json")
    target, recorded = score["evaluation_target"], native["evaluation_target"]
    target_count = int(observed.evaluation_target_trace_mask.sum())
    checks = {
        "native_target_metrics_exact": target == recorded,
        "full_target_coverage": target["trace_count"] == target_count
        and target["sample_count"] == target_count * values.shape[0],
        "native_counts_equal": all(
            target[key] == recorded[key] for key in ("trace_count", "sample_count")
        ),
        "observed_reinsertion_exact": score["observed_max_abs_error"] == 0.0,
        "CPU_restoration": cpu["status"] == "passed",
        **{
            key: bool(np.isclose(target[key], recorded[key], rtol=ENERGY_RTOL, atol=ENERGY_ATOL))
            for key in ("reference_energy", "error_energy")
        },
    }
    checks["inputs_unchanged"] = all(
        file_sha256(Path(path)) == digest for path, digest in before.items()
    )
    report = {
        "status": "success" if all(checks.values()) else "failed_verification",
        "checks": checks,
        "independent_saved_output_metrics": score,
        "CPU_restoration": cpu,
        "primary_checkpoint": "final",
        "global_step": loaded.global_step,
        "training_selection_scope": "held_out_train_partition_patch_instances",
        "training_selection_compared_to_benchmark_score": False,
    }
    _write(output / "result.json", report)
    return report
