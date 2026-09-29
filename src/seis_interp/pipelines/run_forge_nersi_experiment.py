"""Train one config-driven FORGE NeRSI scratch experiment on a sealed preparation.

The workspace config sets every NeRSI knob; the preparation fixes the data
binding (alignment canvas, mask, amplitude scaling, geometry). Every config and
preparation check runs before a run directory exists. Each run writes one
immutable ``<UTC>_<sha12>_<name>/`` directory, and the cross-run
``summary.csv`` is rebuilt afterwards. Test targets are read once, after the
prediction is saved, and only in memory. Metrics reuse the original test set
with oracle trace-RMS restoration, so they are not an independent test estimate.
"""

import fcntl
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

import seis_interp
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    load_model_inputs,
    read_selected_traces,
    verify_hashes,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity, peak_memory, utc_now
from seis_interp.evaluation.forge_mvp_metrics import (
    evaluate_traces,
    validate_predictions,
    verify_metrics_from_waveforms,
    verify_saved_metrics,
)
from seis_interp.forge_nersi_experiment_config import (
    FRAME_AXES,
    budget_comparison,
    config_differences,
    config_sha256,
    ema_initial_weight_fraction,
    fourier_max_angular_frequencies,
    load_experiment_inputs,
    parse_experiment_config,
    resolve_run_config,
    validate_against_preparation,
    validate_device_available,
)
from seis_interp.models.nersi_trace import BAND_HIGH_RAMP_HZ, BAND_LOW_RAMP_HZ
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces
from seis_interp.run_records import (
    CONFIG_FILE_NAME,
    INPUTS_LOCK_FILE_NAME,
    METRICS_FILE_NAME,
    RUN_FILE_NAME,
)
from seis_interp.training.forge_mvp_methods import (
    canvas_method_config,
    initialize_model,
    run_method,
)
from seis_interp.visualization.forge_mvp import plot_sections, plot_training_loss

# Every FORGE loader uses this name, including for preparations with other missing fractions.
MASK_FILE = "masks/m1_random80_mvp_v1.parquet"
MANIFEST_FILE = "input/m1_trace_manifest.parquet"
SUMMARY_FILE = "summary.csv"
# Byte-for-byte copy of the config file that was parsed; can be passed back to --config.
REQUESTED_CONFIG_FILE = "config.requested.yaml"
EMA_WARNING_FRACTION = 0.05
FIXED_BY_CODE = (
    "Adam (coupled L2), constant LR, fp32 weights (TF32 on CUDA), "
    "masked_trace_relative_mse, no clipping"
)
IDENTITY_COLUMNS = (
    "run_id",
    "name",
    "status",
    "failed_stage",
    "start_time",
    "runtime_training_seconds",
    "device",
    "git_commit",
    "git_worktree_dirty",
    "preparation_hash",
)
KNOB_COLUMNS = (
    "model_seed",
    "torch_threads",
    "model.coordinate_frame",
    "model.polar_offset",
    "model.moveout_velocity_m_per_s",
    "model.fourier_components",
    "model.frequency_base",
    "model.encoder_width",
    "model.latent_channels",
    "model.decoder_channels",
    "model.kernel_size",
    "model.head_slowness_ms_per_m",
    "model.static_shift_samples",
    "model.residual_slowness_ms_per_m",
    "model.output_band_hz",
    "model.output_window_mask",
    "training.learning_rate",
    "training.max_steps",
    "training.batch_size",
    "training.weight_decay",
    "training.weight_decay_after",
    "training.ema_decay",
    "training.report_interval",
    "prediction.batch_size",
)
DERIVED_COLUMNS = (
    "changed_keys",
    "changes",
    "parameter_count",
    "canvas_sample_count",
    "parameter_ratio",
    "budget_matched",
    "ema_initial_weight_fraction",
)
METRIC_COLUMNS = (
    "clean_target_relative_mse",
    "all_eligible_relative_mse",
    "clean_global_nmse",
    "clean_median_correlation",
    "final_minibatch_loss",
)
SUMMARY_COLUMNS = (
    *IDENTITY_COLUMNS,
    *KNOB_COLUMNS,
    *DERIVED_COLUMNS,
    *METRIC_COLUMNS,
    "peak_cuda_allocated_bytes",
)


def check_forge_nersi_experiment(
    repo: Path, config_path: Path, inputs_path: Path, *, device_override: str | None = None
) -> dict:
    """Validate and resolve without training, reading test data or creating files."""
    return _check_report(repo, _prepare_experiment(repo, config_path, inputs_path, device_override))


def run_forge_nersi_experiment(
    repo: Path,
    config_path: Path,
    inputs_path: Path,
    output_root: Path,
    *,
    device_override: str | None = None,
    reporter=print,
) -> Path:
    """Train, save, evaluate and plot one experiment; return its run directory."""
    experiment = _prepare_experiment(repo, config_path, inputs_path, device_override)
    report = _check_report(repo, experiment)
    requested, run_config = experiment["requested"], experiment["run_config"]
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_{identity['git_commit'][:12]}_{requested['name']}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / REQUESTED_CONFIG_FILE).write_bytes(experiment["config_bytes"])
    (run_dir / CONFIG_FILE_NAME).write_text(yaml.safe_dump(run_config, sort_keys=False))
    write_json(
        run_dir / INPUTS_LOCK_FILE_NAME,
        {
            "preparation": experiment["spec"]["preparation"],
            "preparation_manifest_sha256": experiment["spec"]["preparation_manifest_sha256"],
            "input_binding": experiment["inputs"].binding,
        },
    )
    record = _initial_record(run_id, identity, experiment, report)
    _write_json_atomic(run_dir / RUN_FILE_NAME, record)
    device = run_config["device"]
    previous_threads = torch.get_num_threads()
    started = perf_counter()
    stage = "setup"
    try:
        torch.set_num_threads(requested["torch_threads"])
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        with (run_dir / "training.log").open("w") as log:

            def report_line(message):
                line = f"{utc_now()} {message}"
                print(line, file=log, flush=True)
                reporter(line)

            report_line(f"run {run_id}")
            for line in _preamble_lines(report):
                report_line(line)
            stage = "training"
            result = run_method(experiment["inputs"], run_config, reporter=report_line)
            record["runtime_training_seconds"] = perf_counter() - started
            _check_trained_shape(result, experiment)
            record["initialization_hash"] = result.diagnostics["initialization_hash"]
            stage = "saving"
            report_line("saving prediction, checkpoint and loss log")
            manifest, mask = _read_manifest_and_mask(experiment["preparation"])
            prediction, index = _save_training_outputs(run_dir, result, experiment, mask)
            evaluation_started = perf_counter()
            stage = "target_read"
            report_line(
                "verifying the raw SEG-Y hashes and reading the test traces for evaluation "
                "(several GB for a full preparation; no progress lines until done)"
            )
            target = _read_test_targets(repo, experiment, manifest, mask)
            stage = "evaluation"
            report_line("evaluating")
            metrics = _evaluate_and_save(run_dir, prediction, target, index)
            stage = "figures"
            report_line("plotting sections")
            _plot_run_sections(run_dir, experiment, manifest, mask, index, target, prediction)
            del target
            record["runtime_evaluation_seconds"] = perf_counter() - evaluation_started
            clean = metrics["clean_target"]
            record.update(
                status="completed",
                failed_stage=None,
                metrics_summary={
                    "clean_target_relative_mse": clean["masked_trace_relative_mse"],
                    "all_eligible_relative_mse": metrics["all_eligible"][
                        "masked_trace_relative_mse"
                    ],
                    "clean_global_nmse": clean["global_nmse"],
                    "clean_median_correlation": clean["correlation"]["median"],
                    "final_minibatch_loss": result.history[-1]["loss"],
                },
            )
            report_line(
                f"completed {run_dir} clean_target relative MSE "
                f"{clean['masked_trace_relative_mse']:.4f}"
            )
    except KeyboardInterrupt:
        record.update(status="interrupted", failed_stage=stage, failure_reason="KeyboardInterrupt")
        raise
    except Exception as error:
        record.update(
            status="failed", failed_stage=stage, failure_reason=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        torch.set_num_threads(previous_threads)
        record.update(
            end_time=utc_now(),
            runtime_seconds=perf_counter() - started,
            **peak_memory(device),
        )
        record["artifact_hashes"] = {
            str(path.relative_to(run_dir)): sha256_file(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path != run_dir / RUN_FILE_NAME
        }
        _write_json_atomic(run_dir / RUN_FILE_NAME, record)
        _update_summary(output_root, reporter)
    return run_dir


def write_experiment_summary(output_root: Path) -> Path:
    """Rebuild ``summary.csv`` from every run record under an exclusive lock."""
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / SUMMARY_FILE
    with (output_root / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [
            _summary_row(path)
            for path in sorted(output_root.iterdir())
            if (path / RUN_FILE_NAME).is_file()
        ]
        frame = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
        with tempfile.NamedTemporaryFile(
            "w", dir=output_root, prefix=".summary.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            frame.to_csv(temporary, index=False)
            temporary.chmod(0o644)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def format_experiment_summary(summary_path: Path) -> str:
    """Completed runs ranked by clean-target relative MSE, then the other runs."""
    frame = pd.read_csv(summary_path)
    if frame.empty:
        return f"no runs yet ({summary_path})"
    completed = frame.loc[frame.status.eq("completed")].sort_values("clean_target_relative_mse")
    others = frame.loc[~frame.status.eq("completed")]
    lines = [f"summary {summary_path}"]
    if len(completed):
        columns = [
            "run_id",
            "clean_target_relative_mse",
            "all_eligible_relative_mse",
            "clean_median_correlation",
            "parameter_count",
            "budget_matched",
            "changes",
        ]
        shown = completed[columns].fillna({"changes": ""})
        table = shown.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        lines += ["completed runs (original test set; not an independent test estimate)", table]
    if len(others):
        lines += ["other runs", others[["run_id", "status", "failed_stage"]].to_string(index=False)]
    return "\n".join(lines)


def format_check_report(report: dict) -> str:
    """Human-readable ``--check`` output; numbers rounded to 4 decimals."""
    fixed = report["fixed_by_preparation"]
    alignment = fixed["alignment"]
    aligned = (
        f"alignment {alignment['velocity_m_per_s']} m/s, max delay {alignment['max_delay_samples']}"
        if alignment
        else "alignment none"
    )
    changes = report["changes_from_preparation_config"]
    breakdown = " | ".join(f"{name} {count:,}" for name, count in report["parameter_breakdown"])
    frequencies = ", ".join(f"{value:.4f}" for value in report["fourier_max_angular_frequencies"])
    polar = ", polar" if report["polar_offset"] else ""
    lines = [
        "FORGE NeRSI experiment check (no training, no test data)",
        f"config        {report['config_path']} (sha256 {report['config_sha256'][:12]})",
        f"preparation   {report['preparation']} (manifest hash verified)",
        f"fixed by preparation  geometry {fixed['geometry_mode']} | {aligned} | "
        f"canvas {fixed['canvas_sample_count']} samples "
        f"(T {fixed['time_sample_count']} at {fixed['sample_interval_us']} us)",
        f"                      amplitude {fixed['amplitude_scaling']} | "
        f"observed {fixed['observed_trace_count']}, test {fixed['test_trace_count']}, "
        f"clean {fixed['clean_test_trace_count']}",
        f"fixed by code         {report['fixed_by_code']}",
        f"changes from preparation config  {'; '.join(changes) if changes else 'none'}",
        f"coordinates   {report['coordinate_frame']}{polar} -> "
        f"[{', '.join(report['coordinate_axes'])}]",
        f"fourier max angular frequency  {frequencies}",
        _moveout_line(report["moveout"]),
        _heads_line(report["heads"]),
        _statics_line(report["statics"]),
        _residual_slowness_line(report["residual_slowness"]),
        _band_line(report["output_band"]),
        _window_line(report["output_window"]),
        _weight_decay_line(report["weight_decay"]),
        f"parameters    {report['parameter_count']:,} ({breakdown})",
        _budget_line(report),
        _ema_line(report["ema_initial_weight_fraction"]),
        f"device        {report['device_status']}",
        *(f"WARNING: {warning}" for warning in report["warnings"]),
        "VALID",
    ]
    return "\n".join(lines)


def _prepare_experiment(repo, config_path, inputs_path, device_override):
    """All static and preparation-dependent checks; builds the model once to count it."""
    # Read the config once so config_sha256 always describes the bytes that were parsed.
    config_bytes = Path(config_path).read_bytes()
    requested = parse_experiment_config(config_bytes, config_path, device_override=device_override)
    spec = load_experiment_inputs(inputs_path)
    preparation = repo / spec["preparation"]
    if not preparation.is_dir():
        raise FileNotFoundError(f"preparation not found: {preparation}; edit inputs.yaml")
    manifest_path = preparation / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"preparation manifest not found: {manifest_path}")
    actual = sha256_file(manifest_path)
    if actual != spec["preparation_manifest_sha256"]:
        raise ValueError(
            f"preparation manifest hash mismatch: {manifest_path}: inputs.yaml has "
            f"{spec['preparation_manifest_sha256']}, the file has {actual} "
            "(sha256sum <preparation>/preparation_manifest.json)"
        )
    inputs, sealed = load_model_inputs(preparation, "nersi_real")
    seal = json.loads(manifest_path.read_text())
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    canvas = int(seal.get("canvas_sample_count", seal["time_sample_count"]))
    run_config = resolve_run_config(
        requested, sealed, contract, inputs.geometry, inputs.spatial_shape
    )
    validate_against_preparation(run_config, len(inputs.observed_cells))
    validate_device_available(run_config["device"])
    try:
        model = initialize_model(canvas_method_config(run_config, canvas))
    except (TypeError, ValueError) as error:
        raise ValueError(f"model: {error}") from error
    breakdown = {}
    for name, parameter in model.named_parameters():
        # Parameters owned by the model itself (the statics) are grouped under their own name.
        group = ".".join(name.split(".")[:-1][:2]) or name
        breakdown[group] = breakdown.get(group, 0) + parameter.numel()
    return {
        "requested": requested,
        "spec": spec,
        "config_path": Path(config_path),
        "config_bytes": config_bytes,
        "config_sha256": config_sha256(config_bytes),
        "inputs_path": Path(inputs_path),
        "preparation": preparation,
        "inputs": inputs,
        "sealed": sealed,
        "seal": seal,
        "contract": contract,
        "canvas": canvas,
        "run_config": run_config,
        "parameter_count": sum(breakdown.values()),
        "parameter_breakdown": list(breakdown.items()),
        "time_padding": model.time_padding,
        "decoder_layout": {
            "pad_left": model.pad_left,
            "pad_right": model.pad_right,
            "decoder_time_count": model.decoder_time_count,
        },
    }


def _check_report(repo, experiment):
    requested, run_config = experiment["requested"], experiment["run_config"]
    seal, contract = experiment["seal"], experiment["contract"]
    model, training = requested["model"], run_config["training"]
    count = experiment["parameter_count"]
    warnings = []
    declared = contract.get("matched_training")
    try:
        budget = budget_comparison(run_config, count, declared)
    except ValueError as error:
        budget = None
        warnings.append(f"budget in input/input_contract.json: {error}")
    ema = ema_initial_weight_fraction(training)
    alignment = run_config.get("alignment")
    moveout = None
    if model["moveout_velocity_m_per_s"] is not None:
        velocity = model["moveout_velocity_m_per_s"]
        moveout = {
            "velocity_m_per_s": velocity,
            "samples_per_radius": run_config["model"]["moveout_samples_per_radius"],
            "padding_samples": experiment["time_padding"],
            # Delays add: events flat in the decoder frame have this velocity on the raw gather.
            "combined_velocity_m_per_s": (
                1 / (1 / alignment["velocity_m_per_s"] + 1 / velocity) if alignment else None
            ),
        }
    if ema is not None and ema > EMA_WARNING_FRACTION:
        warnings.append(
            f"EMA keeps {ema:.4f} of the step-1 weights; use ema_decay <= 1 - 3/max_steps or null"
        )
    if budget is not None and not budget["parameters_within_tolerance"]:
        warnings.append("parameters outside the matched band")
    frame, polar = model["coordinate_frame"], model["polar_offset"]
    return {
        "config_path": _display_path(repo, experiment["config_path"]),
        "config_sha256": experiment["config_sha256"],
        "inputs_path": _display_path(repo, experiment["inputs_path"]),
        "preparation": experiment["spec"]["preparation"],
        "preparation_hash": experiment["spec"]["preparation_manifest_sha256"],
        "fixed_by_preparation": {
            "geometry_mode": run_config["geometry_mode"],
            "alignment": alignment,
            "canvas_sample_count": experiment["canvas"],
            "time_sample_count": contract["time_sample_count"],
            "sample_interval_us": contract["sample_interval_us"],
            "amplitude_scaling": contract["preprocessing"]["amplitude_scaling"],
            "observed_trace_count": seal["observed_trace_count"],
            "test_trace_count": seal["test_trace_count"],
            "clean_test_trace_count": seal["clean_test_trace_count"],
        },
        "fixed_by_code": FIXED_BY_CODE,
        "changes_from_preparation_config": config_differences(run_config, experiment["sealed"]),
        "coordinate_frame": frame,
        "polar_offset": polar,
        "coordinate_axes": FRAME_AXES[(frame, polar)],
        "fourier_max_angular_frequencies": fourier_max_angular_frequencies(model),
        "moveout": moveout,
        **_decoder_prior_report(requested, run_config, experiment),
        "parameter_count": count,
        "parameter_breakdown": experiment["parameter_breakdown"],
        "budget": budget,
        "budget_declared": declared is not None,
        "ema_initial_weight_fraction": ema,
        "device_status": _device_status(run_config["device"]),
        "warnings": warnings,
        "run_config": run_config,
    }


def _decoder_prior_report(requested, run_config, experiment):
    """--check facts for slowness heads, statics, residual slowness, band, window and decay."""
    model, resolved = requested["model"], run_config["model"]
    training = run_config["training"]
    heads = None
    if resolved["head_samples_per_radius"] is not None:
        heads = {
            "slowness_ms_per_m": list(model["head_slowness_ms_per_m"]),
            "samples_per_radius": list(resolved["head_samples_per_radius"]),
            **experiment["decoder_layout"],
            "canvas_sample_count": experiment["canvas"],
        }
    statics = None
    if resolved["static_shift_samples"]:
        statics = {
            "samples": resolved["static_shift_samples"],
            "spatial_shape": list(resolved["spatial_shape"]),
        }
    residual = None
    if resolved["residual_slowness_samples_per_radius"]:
        residual = {
            "ms_per_m": model["residual_slowness_ms_per_m"],
            "samples_per_radius": resolved["residual_slowness_samples_per_radius"],
        }
    return {
        "heads": heads,
        "statics": statics,
        "residual_slowness": residual,
        "output_band": resolved["output_band_hz"],
        "output_window": resolved["output_window"],
        "weight_decay": {
            "initial": training["weight_decay"],
            "after": training["weight_decay_after"],
        },
    }


def _initial_record(run_id, identity, experiment, report):
    requested, contract = experiment["requested"], experiment["contract"]
    evaluation = contract.get("evaluation", {})
    return {
        "run_id": run_id,
        "name": requested["name"],
        "status": "running",
        "failed_stage": None,
        "failure_reason": None,
        "start_time": utc_now(),
        "end_time": None,
        "runtime_seconds": None,
        "runtime_training_seconds": None,
        "runtime_evaluation_seconds": None,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "git_commit": identity["git_commit"],
        "git_worktree_dirty": identity["dirty_worktree"],
        "hardware": identity.get("hardware"),
        "software_versions": identity.get("software_versions"),
        "source_hashes": _source_hashes(),
        "config_path": report["config_path"],
        "config_sha256": report["config_sha256"],
        "inputs_path": report["inputs_path"],
        "requested_config": requested,
        "preparation": report["preparation"],
        "preparation_hash": report["preparation_hash"],
        "model_seed": requested["model_seed"],
        "device": experiment["run_config"]["device"],
        "torch_threads": requested["torch_threads"],
        "canvas_sample_count": experiment["canvas"],
        "parameter_count": report["parameter_count"],
        "parameter_breakdown": dict(report["parameter_breakdown"]),
        "initialization_hash": None,
        "budget": report["budget"],
        "ema_initial_weight_fraction": report["ema_initial_weight_fraction"],
        "changes_from_preparation_config": report["changes_from_preparation_config"],
        "warnings": report["warnings"],
        "evaluation": {
            "domain": "original_clean_target_test_set",
            "independent_test_estimate": False,
            "amplitude_scaling": contract["preprocessing"]["amplitude_scaling"],
            "test_rms_usage": evaluation.get("test_rms_usage"),
            "primary_metric": evaluation.get("primary"),
            "primary_domain": evaluation.get("primary_domain"),
        },
        "metrics_summary": None,
    }


def _check_trained_shape(result, experiment):
    if result.diagnostics["parameter_count"] != experiment["parameter_count"]:
        raise ValueError("trained parameter count differs from the checked count")
    time_count = experiment["contract"]["time_sample_count"]
    if result.diagnostics.get("alignment_canvas_sample_count", time_count) != experiment["canvas"]:
        raise ValueError("trained canvas differs from the sealed canvas")


def _read_manifest_and_mask(preparation):
    manifest = pd.read_parquet(preparation / MANIFEST_FILE)
    mask = pd.read_parquet(preparation / MASK_FILE)
    if not np.array_equal(manifest.cell_id.to_numpy(), mask.cell_id.to_numpy()):
        raise ValueError("trace manifest and mask row order differ")
    return manifest, mask


def _save_training_outputs(run_dir, result, experiment, mask):
    """Persist prediction, checkpoint and loss log before any target is read."""
    time_count = experiment["contract"]["time_sample_count"]
    order = validate_predictions(result.prediction, result.cell_ids, mask, time_count)
    prediction = result.prediction[order]
    index = mask.loc[mask.split.eq("test")].reset_index(drop=True)
    index["prediction_row"] = np.arange(len(index))
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    np.save(artifacts / "prediction.npy", prediction, allow_pickle=False)
    index.to_parquet(artifacts / "prediction_index.parquet", index=False)
    torch.save(result.checkpoint, artifacts / "final.pt")
    history = pd.DataFrame(result.history, columns=["step", "loss", "learning_rate"])
    history.to_parquet(run_dir / "training_log.parquet", index=False)
    (run_dir / "figures").mkdir()
    plot_training_loss(
        result.history,
        run_dir / "figures/training_loss.png",
        report_interval=experiment["run_config"]["training"]["report_interval"],
    )
    return prediction, index


def _read_test_targets(repo, experiment, manifest, mask):
    """Held-out waveforms in mask test order, for evaluation and figures only."""
    preparation, contract = experiment["preparation"], experiment["contract"]
    raw = json.loads((preparation / "input/raw_input_hashes.json").read_text())
    root = repo / raw["dataset_root"]
    verify_hashes(root, raw["hashes"])
    native = native_time_contract(contract)
    target = read_selected_traces(
        root,
        manifest.loc[mask.split.eq("test").to_numpy()],
        native["time_sample_count"],
        native["sample_interval_us"],
    )
    return resample_forge_traces(target, contract)


def _evaluate_and_save(run_dir, prediction, target, index):
    metrics, traces = evaluate_traces(prediction, target, index)
    write_json(run_dir / METRICS_FILE_NAME, metrics)
    traces.to_parquet(run_dir / "metrics_by_trace.parquet", index=False)
    verify_saved_metrics(
        json.loads((run_dir / METRICS_FILE_NAME).read_text()),
        pd.read_parquet(run_dir / "metrics_by_trace.parquet"),
    )
    verify_metrics_from_waveforms(metrics, prediction, target, index.clean_target.to_numpy())
    return metrics


def _plot_run_sections(run_dir, experiment, manifest, mask, index, target, prediction):
    """Materialize only the two predeclared sections, never a full truth cube."""
    contract = experiment["contract"]
    settings = contract["visualization"]
    selected = (
        manifest.source_cell.eq(settings["source_cell"])
        | manifest.receiver_cell.eq(settings["receiver_cell"])
    ).to_numpy()
    section_manifest = manifest.loc[selected].reset_index(drop=True)
    section_mask = mask.loc[selected].reset_index(drop=True)
    test_rows = section_mask.split.eq("test").to_numpy()
    test_positions = pd.Index(index.cell_id).get_indexer(section_mask.cell_id[test_rows])
    if np.any(test_positions < 0):
        raise ValueError("section test cells are missing from the prediction index")
    # Excluded (validation-preparation parent test) traces are never read; they stay zero.
    observed_rows = section_mask.split.eq("observed").to_numpy()
    truth = np.zeros((len(section_mask), contract["time_sample_count"]), dtype=np.float32)
    truth[test_rows] = target[test_positions]
    truth[observed_rows] = experiment["inputs"].read_observed(
        section_mask.cell_id[observed_rows].to_numpy()
    )
    section = truth.copy()
    section[test_rows] = prediction[test_positions]
    plot_sections(
        section_manifest, section_mask, truth, {"NeRSI": section}, contract, run_dir / "figures"
    )


def _summary_row(run_dir):
    try:
        record = json.loads((run_dir / RUN_FILE_NAME).read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        record = None
    if not isinstance(record, dict):
        return {"run_id": run_dir.name, "status": "unreadable"}
    row = {key: record.get(key) for key in IDENTITY_COLUMNS}
    requested = record.get("requested_config") or {}
    for column in KNOB_COLUMNS:
        value = requested
        for key in column.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        row[column] = json.dumps(value) if isinstance(value, (list, dict)) else value
    budget = record.get("budget") or {}
    changes = record.get("changes_from_preparation_config") or []
    row.update(
        changed_keys="; ".join(item.split(":", 1)[0] for item in changes),
        changes="; ".join(changes),
        parameter_count=record.get("parameter_count"),
        canvas_sample_count=record.get("canvas_sample_count"),
        parameter_ratio=budget.get("parameter_ratio"),
        budget_matched=budget.get("budget_matched"),
        ema_initial_weight_fraction=record.get("ema_initial_weight_fraction"),
        peak_cuda_allocated_bytes=record.get("peak_cuda_allocated_bytes"),
    )
    metrics = (record.get("metrics_summary") or {}) if row["status"] == "completed" else {}
    row.update({key: metrics.get(key) for key in METRIC_COLUMNS})
    return row


def _update_summary(output_root, reporter):
    """A summary failure never changes the finished run record."""
    try:
        write_experiment_summary(output_root)
    except Exception as error:
        reporter(f"WARNING: summary not updated: {type(error).__name__}: {error}")


def _write_json_atomic(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_hashes():
    """Hashes of the imported package, identifying the code this run actually used."""
    package = Path(seis_interp.__file__).resolve().parent
    return {
        str(path.relative_to(package.parents[1])): sha256_file(path)
        for path in sorted(package.rglob("*.py"))
    }


def _display_path(root, path):
    path = Path(path).resolve()
    return (
        str(path.relative_to(root.resolve())) if path.is_relative_to(root.resolve()) else str(path)
    )


def _device_status(device):
    if device == "cpu":
        return "cpu"
    return f"{device} available ({torch.cuda.device_count()} CUDA devices)"


def _preamble_lines(report):
    lines = [
        f"parameters {report['parameter_count']:,} at canvas "
        f"{report['fixed_by_preparation']['canvas_sample_count']}",
        _budget_line(report),
        _ema_line(report["ema_initial_weight_fraction"]),
    ]
    return [*lines, *(f"WARNING: {warning}" for warning in report["warnings"])]


def _moveout_line(moveout):
    if moveout is None:
        return "moveout       off"
    line = (
        f"moveout       {moveout['velocity_m_per_s']} m/s -> "
        f"{moveout['samples_per_radius']:.4f} samples per unit radius, "
        f"+{moveout['padding_samples']} padding samples"
    )
    if moveout["combined_velocity_m_per_s"] is not None:
        line += (
            f"; adds to the alignment delay, so decoder-flat events have "
            f"{moveout['combined_velocity_m_per_s']:.1f} m/s on the raw gather"
        )
    return line


def _heads_line(heads):
    if heads is None:
        return "heads         off"
    slowness, shifts = heads["slowness_ms_per_m"], heads["samples_per_radius"]
    return (
        f"heads         {len(slowness)} heads, slowness {min(slowness)}..{max(slowness)} ms/m -> "
        f"shifts {min(shifts):+.1f}..{max(shifts):+.1f} samples per unit radius; "
        f"pad {heads['pad_left']} + {heads['pad_right']}, decoder length "
        f"{heads['decoder_time_count']} (canvas {heads['canvas_sample_count']})"
    )


def _statics_line(statics):
    if statics is None:
        return "statics       off"
    lines, points, receiver_lines, receiver_points = statics["spatial_shape"]
    return (
        f"statics       +-{statics['samples']:g} samples per source and per receiver "
        f"(tanh-bounded, zero start), grids {lines}x{points} + {receiver_lines}x{receiver_points}"
    )


def _residual_slowness_line(residual):
    if residual is None:
        return "residual dm   off"
    return (
        f"residual dm   +-{residual['ms_per_m']:g} ms/m per head -> "
        f"+-{residual['samples_per_radius']:.4f} samples per unit radius (zero start)"
    )


def _band_line(band):
    if band is None:
        return "output band   off"
    low, high = band
    return (
        f"output band   {low:g}..{high:g} Hz flat, zero-phase raised-cosine ramps "
        f"{low - BAND_LOW_RAMP_HZ:g}..{low:g} and {high:g}..{high + BAND_HIGH_RAMP_HZ:g} Hz"
    )


def _window_line(window):
    if window is None:
        return "output window off"
    return (
        f"output window zero outside [{window['max_delay_samples']} - "
        f"rint(r * {window['delay_samples_per_radius']:.4f}), +{window['window_samples']}) samples"
    )


def _weight_decay_line(weight_decay):
    after = weight_decay["after"]
    if after is None:
        return f"weight decay  {weight_decay['initial']:g} constant"
    return (
        f"weight decay  {weight_decay['initial']:g} before step {after['step']}, "
        f"{after['value']:g} from step {after['step']}"
    )


def _budget_line(report):
    budget, training = report["budget"], report["run_config"]["training"]
    if budget is None:
        if report["budget_declared"]:
            return "budget        not compared (see WARNING)"
        return "budget        none declared by the preparation"
    low, high = budget["parameter_band"]
    within = "within" if budget["parameters_within_tolerance"] else "outside"

    def compare(label, actual, expected, equal):
        return f"{label} {actual}{'=' if equal else '!='}{expected}"

    return " | ".join(
        [
            f"budget        target {budget['parameter_count_target']:,} "
            f"+-{budget['parameter_count_tolerance'] * 100:.1f}% band {low}..{high}: "
            f"ratio {budget['parameter_ratio']:+.4f} {within}",
            compare("steps", training["max_steps"], budget["max_steps"], budget["steps_equal"]),
            compare(
                "lr",
                training["learning_rate"],
                budget["learning_rate"],
                budget["learning_rate_equal"],
            ),
            compare("batch", training["batch_size"], budget["batch_size"], budget["batch_equal"]),
            f"matched {'yes' if budget['budget_matched'] else 'no'}",
        ]
    )


def _ema_line(fraction):
    if fraction is None:
        return "EMA           off"
    return f"EMA           weight left on first update {fraction:.4f}"
