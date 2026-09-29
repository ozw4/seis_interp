"""Compare linear and asinh amplitude training without per-trace RMS."""

import json
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.forge_compressed_inputs import compressed_model_inputs
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces, verify_saved_metrics
from seis_interp.processing.amplitude_compression import transform_amplitudes
from seis_interp.processing.forge_condition_inputs import nested_condition_mask
from seis_interp.training.forge_condition_search import train_regsi_condition


def _resolved_model_config(source, settings, variant, scale, device, steps):
    config = yaml.safe_load(source.read_text())
    config["reference_condition_metadata"] = {
        key: config.pop(key)
        for key in ("amplitude_scaling", "global_rms", "mask_id", "mask_hash", "checkpoint_rule")
        if key in config
    }
    config.update(
        device=device,
        model_seed=settings["model_seed"],
        loss=settings["loss"],
        amplitude_scaling=f"common_percentile_{variant}",
        global_rms=1.0,
        amplitude_transform={
            **settings["amplitude_transform"],
            "variant": variant,
            "fitted_scale": scale,
        },
        checkpoint_rule="fixed_final_raw_and_ema",
        mask_id=f"nested_missing_{settings['missing_fraction']:g}",
        time_decimation_factor=settings["time_decimation_factor"],
    )
    config["training"].update(settings["training"])
    if steps is not None:
        config["training"]["max_steps"] = steps
    config["prediction"]["query_batch_size"] = config["training"]["query_batch_size"]
    config["model"]["max_edge_time_shift_samples"] = settings["max_edge_time_shift_samples"]
    if config["model"]["amplitude_mode"] != "train_global_rms":
        raise ValueError("model must not apply per-trace RMS")
    if config["loss"] != "masked_trace_mse":
        raise ValueError("compression comparison requires unweighted model-space MSE")
    return config


def run_forge_amplitude_compression(
    repo, study, variant, *, device="cuda:0", steps=None, scale_multiplier=1.0
):
    """Fit the observed-only transform, train, invert predictions, then open targets."""
    settings = yaml.safe_load((study / "config.yaml").read_text())
    input_paths = yaml.safe_load((study / "inputs.yaml").read_text())
    transform = settings["amplitude_transform"]
    if variant not in transform["variants"] or variant not in ("linear", "asinh"):
        raise ValueError("unknown amplitude transform variant")
    if transform["per_trace_rms"] or transform["test_rms"]:
        raise ValueError("per-trace RMS is not supported in this comparison")
    cache = repo / input_paths["cache"]
    reference = repo / input_paths["reference_run"]
    reference_record = json.loads((reference / "manifest.json").read_text())
    if reference_record["status"] != "completed":
        raise ValueError("reference experiment is incomplete")
    verify_hashes(reference, {"config.yaml": reference_record["artifacts"]["config.yaml"]})
    seal = json.loads((cache / "manifest.json").read_text())
    if seal["status"] != "prepared":
        raise ValueError("condition cache is incomplete")
    factor = settings["time_decimation_factor"]
    names = [f"waveforms_q{factor}.npy", "base_mask.parquet", "geometry.parquet", "probe_cells.npy"]
    verify_hashes(cache, {name: seal["artifacts"][name] for name in names})
    mask = nested_condition_mask(
        pd.read_parquet(cache / "base_mask.parquet"), settings["missing_fraction"]
    )
    geometry = pd.read_parquet(cache / "geometry.parquet")
    values = np.load(cache / f"waveforms_q{factor}.npy", mmap_mode="r")
    observed = mask.split.eq("observed").to_numpy()
    time_s = np.arange(values.shape[1]) * factor * seal["original_sample_interval_us"] / 1e6
    query_cells = np.load(cache / "probe_cells.npy")
    inputs, scale = compressed_model_inputs(
        geometry,
        mask,
        values[observed],
        seal["spatial_shape"],
        time_s,
        variant,
        transform["percentile"],
        {"cache_hash": sha256_file(cache / "manifest.json")},
        scale_multiplier=scale_multiplier,
    )
    config = _resolved_model_config(
        reference / "config.yaml", settings, variant, scale, device, steps
    )
    config["amplitude_transform"]["scale_multiplier"] = scale_multiplier
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = variant if scale_multiplier == 1 else f"{variant}_scale{scale_multiplier:g}"
    output = (
        repo / "runs" / settings["study_id"] / (f"{stamp}_{identity['git_commit'][:12]}_{suffix}")
    )
    output.mkdir(parents=True, exist_ok=False)
    mask.to_parquet(output / "mask.parquet", index=False)
    config["mask_hash"] = sha256_file(output / "mask.parquet")
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    write_json(
        output / "preprocessing.json",
        {
            "transform": variant,
            "percentile": transform["percentile"],
            "scale": scale,
            "scale_multiplier": scale_multiplier,
            "fit_domain": "outer_observed_samples_only",
            "observed_trace_count": int(observed.sum()),
            "per_trace_rms": False,
            "test_rms": False,
            "runtime_amplitude_scale": 1.0,
            "model_space_max_abs": float(np.max(np.abs(inputs.observed_values))),
        },
    )
    sources = [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / "scripts/forge_amplitude_compression.py",
        study / "config.yaml",
        study / "inputs.yaml",
    ]
    for source in sources:
        destination = output / "source" / source.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    record = {
        **identity,
        "status": "running",
        "start_time": utc_now(),
        "variant": variant,
        "factor": factor,
        "missing_fraction": settings["missing_fraction"],
        "scale": scale,
        "scale_multiplier": scale_multiplier,
        "per_trace_rms": False,
        "test_rms": False,
        "loss": config["loss"],
        "cache": str(cache.relative_to(repo)),
        "cache_hash": sha256_file(cache / "manifest.json"),
        "reference_run": str(reference.relative_to(repo)),
        "reference_manifest_hash": sha256_file(reference / "manifest.json"),
        "observed_trace_count": len(inputs.observed_cells),
        "test_trace_count": len(inputs.test_cells),
        "evaluated_trace_count": len(query_cells),
        "sample_interval_us": factor * seal["original_sample_interval_us"],
        "sample_count": len(time_s),
    }
    write_json(output / "manifest.json", record)
    print(output, flush=True)
    print(
        f"Transform={variant}; effective scale={scale:.10g}; "
        f"observed P{transform['percentile']:g} x {scale_multiplier:g}; no per-trace RMS",
        flush=True,
    )
    started = perf_counter()
    try:
        result = train_regsi_condition(
            inputs, config, query_cells, reporter=lambda message: print(message, flush=True)
        )
        history = pd.DataFrame(result["history"])
        history.to_parquet(output / "training_history.parquet", index=False)
        for name, state in result["states"].items():
            torch.save(state, output / f"state_{name}.pt")
        restored = {}
        for name, prediction in result["predictions"].items():
            np.save(output / f"prediction_transformed_{name}.npy", prediction, allow_pickle=False)
            restored[name] = transform_amplitudes(prediction, scale, variant, inverse=True)
            np.save(output / f"prediction_{name}.npy", restored[name], allow_pickle=False)
        # Evaluation boundary: no held-out samples or target RMS were used above.
        positions = pd.Index(mask.cell_id).get_indexer(query_cells)
        index = mask.iloc[positions].reset_index(drop=True)
        index.to_parquet(output / "evaluation_index.parquet", index=False)
        target = np.asarray(values[positions], dtype=np.float32)
        np.save(output / "target.npy", target, allow_pickle=False)
        summaries = {}
        for name, prediction in restored.items():
            metrics, per_trace = evaluate_traces(prediction, target, index)
            verify_saved_metrics(metrics, per_trace)
            write_json(output / f"metrics_{name}.json", metrics)
            per_trace.to_parquet(output / f"metrics_by_trace_{name}.parquet", index=False)
            summaries[name] = metrics
        write_json(output / "summary.json", summaries)
        record.update(
            status="completed",
            runtime_seconds=perf_counter() - started,
            initialization_hash=result["initialization_hash"],
            query_presentations=result["query_presentations"],
            no_context_training_queries=result["no_context_query_count"],
            no_context_test_queries=int((~result["context"]).sum()),
            test_waveforms_read_only_after_prediction=True,
            mean_last_500_training_loss=float(history.loss.tail(500).mean()),
        )
        print(json.dumps({k: v["clean_target"] for k, v in summaries.items()}), flush=True)
    except Exception as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["end_time"] = utc_now()
        record["artifacts"] = {
            str(p.relative_to(output)): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "manifest.json"
        }
        write_json(output / "manifest.json", record)
    return output
