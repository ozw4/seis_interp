"""Tune FORGE methods on a sealed preparation under its matched neural budget.

Candidates may change method-internal settings only. The shared preparation,
alignment canvas, mask, amplitude restoration and evaluator stay fixed; neural
candidates keep the study's update count, learning rate, batch size and
parameter-count band. A neural candidate may set a top-level ``model_seed``
(seed replicates); the preparation still validates the study's contract seed.

The scored domain is the preparation's ``selection_domain``: the validation
holdout of a ``_validation`` preparation, or the preparation's test set
(``final_test``, the default). Scores are an independent test estimate only
for a final-test preparation searched with ``purpose: final_evaluation``.
With ``training.ema_eval_decays`` every EMA shadow of a run is saved and scored
under ``<candidate>/ema_<decay>/``, one summary row per (candidate, decay), with
the shadow's final weights in ``ema_<decay>/final.pt``; ``training.ema_snapshot_updates``
saves every EMA average after those updates as ``update_<n>/ema_<decay>.pt``.
``evaluate: false`` saves predictions and training records without reading any
target or computing a metric. A study whose contract declares a
``validation_holdout`` selects on that holdout only: on its final-test
preparation the tuner runs ``purpose: final_evaluation`` with ``evaluate: false``
and nothing else. A ``reference_only: true`` selection search skips only the
parameter-count band (updates, learning rate and loss traces stay matched); its
manifest records the flag, and fair-grid stages never generate one.
A ReGSI candidate may set ``training.execution`` (``trace_graph_execution``), which
changes only how the fit runs; invalid switches are rejected before any training.
"""

import gc
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.c3_fair_grid_truth import read_c3_scored_targets
from seis_interp.data.forge_aligned_inputs import align_model_inputs, unalign_predictions
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    load_model_inputs,
    read_selected_traces,
    verify_hashes,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.evaluation.forge_mvp_metrics import (
    evaluate_traces,
    validate_predictions,
    verify_metrics_from_waveforms,
    verify_saved_metrics,
)
from seis_interp.forge_mvp_config import (
    TUNABLE_NEURAL_METHODS,
    validate_matched_training,
    validate_parameter_counts,
)
from seis_interp.processing.c3_fair_grid_contract import C3_DATASET_ID
from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces
from seis_interp.processing.pocs_batched import interpolate_pocs_volume_batched
from seis_interp.run_records import runtime_resource_metadata
from seis_interp.training.forge_mvp_methods import (
    ForgeMethodResult,
    ema_eval_decays,
    ema_snapshot_updates,
    model_parameter_count,
    run_method,
    shadow_key,
)
from seis_interp.training.trace_graph_execution import validate_trace_graph_execution

TUNABLE_SECTIONS = {
    "pocs_grid": ("algorithm",),
    "drr_grid": ("algorithm",),
    "ccnet5d_grid": ("model", "training", "patches", "prediction"),
    "nersi_real": ("model", "training", "prediction"),
    "inr_real": ("model", "training", "prediction"),
    "cinr_real": ("model", "training", "prediction", "context"),
    "nersi_profile": ("model", "training", "prediction"),
    "regsi_real": ("model", "training", "graph", "geometry_features", "prediction"),
    "regsi_grid": ("model", "training", "graph", "geometry_features", "prediction"),
}
OPTIONAL_KEYS = {
    ("ccnet5d_grid", "model"): {"time_kernel_size", "time_dilation"},
    ("ccnet5d_grid", "training"): {"ema_decay", "ema_eval_decays", "ema_snapshot_updates"},
    ("nersi_real", "model"): {"coordinate_transform", "polar_offset"},
    ("nersi_real", "training"): {"ema_eval_decays", "ema_snapshot_updates"},
    ("inr_real", "training"): {"ema_eval_decays", "ema_snapshot_updates"},
    ("cinr_real", "training"): {"ema_eval_decays", "ema_snapshot_updates"},
    ("nersi_profile", "model"): {
        "time_upsample_channels",
        "time_kernel_size",
        "latent_spatial_rank",
    },
    ("nersi_profile", "training"): {
        "ema_eval_decays",
        "ema_snapshot_updates",
        "nuclear_norm_weight",
        "coordinate_jitter_cells",
    },
    ("regsi_real", "training"): {"ema_eval_decays", "ema_snapshot_updates", "execution"},
    ("regsi_grid", "training"): {"ema_eval_decays", "ema_snapshot_updates", "execution"},
}
SELECTION_DOMAINS = ("validation_holdout", "final_test")
PURPOSES = ("selection", "final_evaluation")
FIXED_KEYS = {
    "training": {
        "optimizer",
        "learning_rate",
        "max_steps",
        "learning_rate_schedule",
        "mixed_precision",
        "batch_size",
        "query_batch_size",
        "supervised_traces_per_update",
    },
    "model": {
        "time_sample_count",
        "profile_shape",
        "input_features",
        "activation",
        "output_activation",
        "moveout_samples_per_radius",
    },
    "algorithm": {"frequency_min_hz", "frequency_max_hz"},
}


def tune_forge_matched(
    repo: Path, preparation: Path, search_config: Path, *, device: str | None = None
):
    """Evaluate every listed candidate; record the best without replacing the study configs.

    ``device`` overrides where the search executes. The search yaml is recorded byte for
    byte, so the run keeps its planned identity; the override is written to the manifest.
    """
    search = override_search_device(yaml.safe_load(search_config.read_text()), device)
    method = search["method_id"]
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    scoring = search_scoring(seal, search)
    inputs, base_config = load_model_inputs(preparation, method)
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    require_holdout_selection(contract, scoring)
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo
        / "runs"
        / contract["study_id"]
        / f"{stamp}_{identity['git_commit'][:12]}_{method}_matched_tuning"
    )
    output.mkdir(parents=True, exist_ok=False)
    print(output, flush=True)
    (output / "search.yaml").write_bytes(search_config.read_bytes())
    source_hashes = {}
    for source in [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / "scripts/forge_matched_tuning.py",
    ]:
        relative = source.relative_to(repo)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        source_hashes[str(relative)] = sha256_file(source)
    record = dict(
        **identity,
        status="running",
        start_time=utc_now(),
        method_id=method,
        preparation=str(preparation.relative_to(repo)),
        preparation_hash=sha256_file(preparation / "preparation_manifest.json"),
        input_binding=inputs.binding,
        alignment=base_config.get("alignment"),
        matched_training=contract.get("matched_training"),
        implementation_hashes=source_hashes,
        **scoring,
        target_relative_mse=search.get("target_relative_mse"),
        **({"device_override": device} if device is not None else {}),
    )
    write_json(output / "manifest.json", record)
    try:
        canvas = seal.get("canvas_sample_count", seal["time_sample_count"])
        summaries = _evaluate_candidates(
            repo, preparation, inputs, base_config, contract, canvas, search, output
        )
        record["status"] = "completed"
        if scoring["evaluate"]:
            best = min(summaries, key=lambda row: row["primary_relative_mse"])
            record.update(
                best_candidate=best,
                target_met=best["primary_relative_mse"] <= search["target_relative_mse"],
            )
    except (Exception, KeyboardInterrupt) as error:
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failure_reason=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        record["end_time"] = utc_now()
        record["artifacts"] = {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "manifest.json"
        }
        write_json(output / "manifest.json", record)
    return output


def override_search_device(search: dict, device: str | None) -> dict:
    """The search with its execution device replaced by another device of the same kind."""
    if device is None:
        return search
    planned = search["device"]
    same_kind = device == "cpu" if planned == "cpu" else device.startswith("cuda:")
    if not same_kind or (device != "cpu" and not device[5:].isdigit()):
        raise ValueError(f"device override {device!r} must be the same kind as {planned!r}")
    return {**search, "device": device}


def search_scoring(seal: dict, search: dict) -> dict:
    """Scored domain from the preparation seal and the search's declared purpose."""
    domain = seal.get("selection_domain", "final_test")
    purpose = search.get("purpose", "selection")
    evaluate = search.get("evaluate", True)
    if domain not in SELECTION_DOMAINS or purpose not in PURPOSES:
        raise ValueError("unsupported selection domain or search purpose")
    if purpose == "final_evaluation" and domain != "final_test":
        raise ValueError("final evaluation requires a final-test preparation")
    if not isinstance(evaluate, bool):
        raise ValueError("evaluate must be boolean")
    if evaluate and "target_relative_mse" not in search:
        raise ValueError("an evaluated search declares target_relative_mse")
    if not evaluate and search.get("stop_when_target_met", False):
        raise ValueError("stop_when_target_met requires evaluate: true")
    reference = search.get("reference_only", False)
    if not isinstance(reference, bool) or (reference and purpose != "selection"):
        raise ValueError("reference_only must be a boolean of a selection search")
    return {
        "selection_domain": domain,
        "purpose": purpose,
        "evaluate": evaluate,
        "independent_test_estimate": purpose == "final_evaluation",
        **({"reference_only": True} if reference else {}),
    }


def require_holdout_selection(contract: dict, scoring: dict) -> None:
    """A study with a validation holdout never scores its final test during a search."""
    if (
        "validation_holdout" in contract
        and scoring["selection_domain"] == "final_test"
        and (scoring["purpose"] != "final_evaluation" or scoring["evaluate"])
    ):
        raise ValueError(
            "this study selects on its validation holdout: search on the _validation "
            "preparation; its final-test preparation takes purpose: final_evaluation "
            "with evaluate: false only"
        )


def matched_candidate_config(
    base_config,
    candidate,
    device,
    budget,
    canvas_length,
    observed_trace_count,
    *,
    parameter_band=True,
):
    """Apply method-internal overrides and reject anything the matched comparison fixes.

    A neural candidate may replace ``model_seed``; it seeds initialization and
    every training stream exactly as the study seed does. ``parameter_band=False``
    (reference-only searches) still counts the parameters but accepts any count.
    """
    method = base_config["method_id"]
    sections = TUNABLE_SECTIONS[method]
    if set(candidate) - {"name", "model_seed", *sections}:
        raise ValueError(f"{method}: only {sections} may be tuned")
    config = deepcopy(base_config)
    if "model_seed" in candidate:
        seed = candidate["model_seed"]
        if (
            method not in TUNABLE_NEURAL_METHODS
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError(f"{method}: model_seed must be an integer of a neural method")
        config["model_seed"] = seed
    for section in sections:
        overrides = candidate.get(section, {})
        supported = set(config[section]) | OPTIONAL_KEYS.get((method, section), set())
        if set(overrides) - supported:
            raise ValueError(f"{method}: unsupported {section} parameters")
        fixed = FIXED_KEYS.get(section, set())
        if any(overrides[key] != config[section].get(key) for key in set(overrides) & fixed):
            raise ValueError(f"{method}: {section} keys are fixed by the matched comparison")
        config[section].update(deepcopy(overrides))
    parameter_count = 0
    if (
        method == "ccnet5d_grid"
        and config["patches"]["shape"][0] != (base_config["patches"]["shape"][0])
    ):
        raise ValueError("CCNet5D patches must span complete traces")
    if method in TUNABLE_NEURAL_METHODS:
        config["checkpoint_rule"] = (
            "fixed_final_ema" if config["training"].get("ema_decay") else "fixed_final"
        )
        ema_eval_decays(config["training"])
        ema_snapshot_updates(config["training"])
        if budget is not None:
            validate_matched_training({method: config}, budget, observed_trace_count)
            parameter_count = model_parameter_count(config, canvas_length)
            if parameter_band:
                validate_parameter_counts({method: parameter_count}, budget)
    config["device"] = device
    if method.startswith("regsi"):
        # training.execution changes only how the fit runs; reject invalid switches here.
        validate_trace_graph_execution(
            config["training"].get("execution"),
            neighbor_search=config["graph"]["neighbor_search"],
            device=device,
        )
    config["changes_from_source"] = [
        "Tune method-internal parameters on the sealed preparation and evaluation domain."
    ]
    if "model_seed" in candidate:
        config["changes_from_source"].append("Replace the model seed for a seed replicate.")
    return config, parameter_count


def _classical_batched(inputs, config, search) -> ForgeMethodResult:
    """Batched POCS/DRR on the same model-space canvas and global-RMS units as ``run_method``.

    The model-space global RMS is 1.0 for trace-scaled preparations and the observed
    global RMS of a global-RMS (C3) preparation, as in ``run_method``.
    """
    model_inputs = inputs.model_space()
    canvas = None
    if config.get("alignment") is not None:
        model_inputs, canvas = align_model_inputs(model_inputs, config["alignment"])
    volume = model_inputs.volume()
    observed = (volume.values.astype(np.float64) / model_inputs.global_rms).astype(np.float32)
    kwargs = dict(config["algorithm"], device=search["device"])
    if config["method_id"] == "drr_grid":
        kwargs["workers"] = search.get("workers", 1)
        result = interpolate_drr_volume_batched(
            observed, volume.observed_trace_mask, model_inputs.time_s, **kwargs
        )
        uncovered = result.uncovered_trace_count
    else:
        result = interpolate_pocs_volume_batched(observed, volume.observed_trace_mask, **kwargs)
        uncovered = result.uncovered_sample_count
    if uncovered:
        raise ValueError("prediction has uncovered samples")
    np.testing.assert_array_equal(
        result.values[:, volume.observed_trace_mask],
        observed[:, volume.observed_trace_mask],
    )
    time_count = len(model_inputs.time_s)
    prediction = result.values.reshape(time_count, -1)[:, inputs.test_cells].T
    prediction = (prediction.astype(np.float64) * model_inputs.global_rms).astype(np.float32)
    if canvas is not None:
        prediction = unalign_predictions(
            np.ascontiguousarray(prediction), inputs.test_cells, canvas
        )
    if inputs.test_rms is not None:
        prediction = prediction.astype(np.float64) * inputs.test_rms[:, None]
    return ForgeMethodResult(
        np.ascontiguousarray(prediction, dtype=np.float32),
        inputs.test_cells.copy(),
        [],
        None,
        {"block_count": result.block_count, "execution_backend": "batched"},
    )


def _evaluate_candidates(repo, preparation, inputs, base_config, contract, canvas, search, output):
    method = base_config["method_id"]
    evaluate = search.get("evaluate", True)
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    index = mask.loc[mask.split.eq("test")].reset_index(drop=True)
    index["prediction_row"] = np.arange(len(index))
    index.to_parquet(output / "prediction_index.parquet", index=False)
    budget = contract.get("matched_training")
    backend = search.get("backend", "mvp")
    if backend not in ("mvp", "batched") or (
        backend == "batched" and method not in ("pocs_grid", "drr_grid")
    ):
        raise ValueError("backend must be mvp, or batched for POCS/DRR")
    target = None
    summaries = []
    torch.set_num_threads(search.get("torch_threads", 4))
    cuda = search["device"].startswith("cuda")
    for candidate in search["candidates"]:
        name = candidate["name"]
        config, parameter_count = matched_candidate_config(
            base_config,
            candidate,
            search["device"],
            budget,
            canvas,
            len(inputs.observed_cells),
            parameter_band=not search.get("reference_only", False),
        )
        path = output / name
        path.mkdir(exist_ok=False)
        (path / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Starting {name}: {candidate}", flush=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats(search["device"])
        started = perf_counter()
        with (path / "training.log").open("w") as log:

            def report(message):
                line = f"{utc_now()} {message}"
                print(line, file=log, flush=True)
                print(line, flush=True)

            if backend == "batched":
                result = _classical_batched(inputs, config, search)
            else:
                result = run_method(inputs, config, reporter=report)
        runtime = perf_counter() - started
        if not all(
            np.isfinite(v).all() for v in (result.prediction, *result.shadow_predictions.values())
        ):
            raise ValueError(f"{name}: non-finite prediction")
        order = validate_predictions(result.prediction, result.cell_ids, mask, len(inputs.time_s))
        prediction = result.prediction[order]
        np.save(path / "prediction.npy", prediction, allow_pickle=False)
        shadows = {}
        for key, values in result.shadow_predictions.items():
            validate_predictions(values, result.cell_ids, mask, len(inputs.time_s))
            shadows[key] = values[order]
            (path / f"ema_{key}").mkdir()
            np.save(path / f"ema_{key}/prediction.npy", shadows[key], allow_pickle=False)
        if result.checkpoint is not None:
            torch.save(result.checkpoint, path / "final.pt")
        for key, states in result.ema_states.items():
            for update, state in states.items():
                final = update == config["training"]["max_steps"]
                target_path = path / (
                    f"ema_{key}/final.pt" if final else f"update_{update}/ema_{key}.pt"
                )
                target_path.parent.mkdir(exist_ok=True)
                torch.save(
                    {
                        **result.checkpoint,
                        "model_state_dict": state,
                        "ema_decay": float(key),
                        "steps_completed": update,
                    },
                    target_path,
                )
        if result.history:
            pd.DataFrame(result.history).to_parquet(path / "training_log.parquet", index=False)
        run = dict(
            name=name,
            model_seed=config.get("model_seed"),
            runtime_seconds=runtime,
            **(
                {
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(search["device"]),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(search["device"]),
                }
                if cuda
                else {}
            ),
            **({"final_batch_training_loss": result.history[-1]["loss"]} if result.history else {}),
        )
        run.update(result.diagnostics)
        conv_kernels = run.pop("conv_kernels_first_update", None)
        if parameter_count and run["parameter_count"] != parameter_count:
            raise ValueError("trained parameter count differs from the validated count")
        primary = config.get("training", {}).get("ema_decay")
        primary = None if primary is None else shadow_key(primary)
        scored = shadows or {primary: prediction}
        if evaluate and target is None:
            target = _read_targets(repo, preparation, contract, mask)
        rows = []
        for key, values in scored.items():
            row = {"name": name, "ema_decay": key, "primary_checkpoint": key == primary}
            if evaluate:
                directories = [path / f"ema_{key}"] if shadows else []
                if key == primary:
                    directories.append(path)
                row.update(_score(values, target, index, directories))
            rows.append({**row, **run})
        manifest = {**next(row for row in rows if row["primary_checkpoint"])}
        manifest["numerics"] = runtime_resource_metadata(torch.device(search["device"]))
        if conv_kernels is not None:
            manifest["conv_kernels_first_update"] = conv_kernels
        if shadows:
            manifest["ema_shadows"] = {row["ema_decay"]: _scores_only(row) for row in rows}
        write_json(path / "run_manifest.json", manifest)
        summaries.extend(rows)
        pd.DataFrame(summaries).to_csv(output / "search_results.csv", index=False)
        for row in rows:
            print(json.dumps(row), flush=True)
        del result, prediction, shadows, scored
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        if search.get("stop_when_target_met", False) and any(
            row["primary_relative_mse"] <= search["target_relative_mse"] for row in rows
        ):
            break
    return summaries


def _read_targets(repo, preparation, contract, mask):
    """Read the scored split once: the validation holdout or the final test set."""
    if contract.get("dataset_id") == C3_DATASET_ID:
        return read_c3_scored_targets(repo, preparation, mask)
    raw = json.loads((preparation / "input/raw_input_hashes.json").read_text())
    verify_hashes(repo / raw["dataset_root"], raw["hashes"])
    rows = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    native = native_time_contract(contract)
    target = read_selected_traces(
        repo / raw["dataset_root"],
        rows.loc[mask.split.eq("test")],
        native["time_sample_count"],
        native["sample_interval_us"],
    )
    return resample_forge_traces(target, contract)


def _score(prediction, target, index, directories):
    """Verified metrics of one prediction, written to every listed directory."""
    metrics, traces = evaluate_traces(prediction, target, index)
    verify_saved_metrics(metrics, traces)
    verify_metrics_from_waveforms(metrics, prediction, target, index.clean_target.to_numpy())
    for directory in directories:
        write_json(directory / "metrics.json", metrics)
        traces.to_parquet(directory / "metrics_by_trace.parquet", index=False)
    clean = metrics["clean_target"]
    return dict(
        primary_relative_mse=clean["masked_trace_relative_mse"],
        all_eligible_relative_mse=metrics["all_eligible"]["masked_trace_relative_mse"],
        global_nmse=clean["global_nmse"],
        median_correlation=clean["correlation"]["median"],
    )


def _scores_only(row):
    keys = (
        "primary_relative_mse",
        "all_eligible_relative_mse",
        "global_nmse",
        "median_correlation",
    )
    return {"primary_checkpoint": row["primary_checkpoint"]} | {
        key: row[key] for key in keys if key in row
    }
