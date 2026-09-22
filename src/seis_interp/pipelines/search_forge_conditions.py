"""Prepare and execute explicitly oracle-RMS FORGE condition diagnostics."""

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import read_selected_traces, verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces, verify_saved_metrics
from seis_interp.processing.forge_condition_inputs import (
    condition_model_inputs,
    downsample_traces,
    nested_condition_mask,
)
from seis_interp.processing.forge_time_sampling import native_time_contract
from seis_interp.training.forge_condition_search import train_regsi_condition


def _new_run(repo, suffix):
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo
        / "runs/study_054_forge_regsi_conditions"
        / (f"{stamp}_{identity['git_commit'][:12]}_{suffix}")
    )
    output.mkdir(parents=True, exist_ok=False)
    identity.update(start_time=utc_now(), status="running")
    hashes = {}
    sources = [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / "scripts/forge_regsi_conditions.py",
    ]
    for source in sources:
        relative = source.relative_to(repo)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        hashes[str(relative)] = sha256_file(source)
    identity["implementation_hashes"] = hashes
    print(output, flush=True)
    return output, identity


def _finish(output, manifest):
    manifest["end_time"] = utc_now()
    manifest["artifacts"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != output / "manifest.json"
    }
    write_json(output / "manifest.json", manifest)


def prepare_forge_conditions(repo: Path, preparation: Path, *, factors=(1, 8, 16)):
    """Read original eligible waveforms once, for new masks and authorized oracle RMS."""
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    raw = json.loads((preparation / "input/raw_input_hashes.json").read_text())
    verify_hashes(repo / raw["dataset_root"], raw["hashes"])
    output, record = _new_run(repo, "data")
    record.update(
        preparation=str(preparation.relative_to(repo)),
        preparation_hash=sha256_file(preparation / "preparation_manifest.json"),
        oracle_test_rms_authorized=True,
        raw_input_hashes=raw,
        downsampling={
            "factors": list(factors),
            "filter": "Hamming sinc 20*q+1 taps, cutoff 0.45/q, reflect padding",
        },
    )
    write_json(output / "manifest.json", record)
    try:
        rows = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
        mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
        geometry = pd.read_parquet(preparation / "features/real_geometry.parquet")
        if not np.array_equal(rows.cell_id, mask.cell_id) or not np.array_equal(
            rows.cell_id, geometry.cell_id
        ):
            raise ValueError("preparation rows do not align")
        contract = json.loads((preparation / "input/input_contract.json").read_text())
        native = native_time_contract(contract)
        values = read_selected_traces(
            repo / raw["dataset_root"],
            rows,
            native["time_sample_count"],
            native["sample_interval_us"],
        )
        for factor in factors:
            print(f"Saving time factor {factor}", flush=True)
            np.save(
                output / f"waveforms_q{factor}.npy",
                downsample_traces(values, factor),
                allow_pickle=False,
            )
        mask.to_parquet(output / "base_mask.parquet", index=False)
        geometry.to_parquet(output / "geometry.parquet", index=False)
        pool = mask.loc[mask.hash_rank <= round(len(mask) * 0.05), "cell_id"].to_numpy()
        probes = np.sort(
            np.random.default_rng(5401).choice(pool, size=min(512, len(pool)), replace=False)
        )
        np.save(output / "probe_cells.npy", probes, allow_pickle=False)
        (output / "regsi_real.yaml").write_bytes(
            (preparation / "configs/regsi_real.yaml").read_bytes()
        )
        record.update(
            status="prepared",
            spatial_shape=contract["spatial_shape"],
            original_sample_interval_us=native["sample_interval_us"],
            original_time_sample_count=native["time_sample_count"],
            probe_seed=5401,
        )
    except Exception as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        _finish(output, record)
    return output


def run_forge_condition(
    repo: Path,
    cache: Path,
    *,
    factor=8,
    missing_fraction=0.1,
    normalization="trace_oracle",
    steps=5000,
    batch_size=32,
    device="cuda:0",
    inner_fraction=0.1,
    learning_rate=0.001,
    time_shift=0,
    all_test=False,
):
    """Train independently and evaluate either the common probe or all missing cells."""
    cache_record = json.loads((cache / "manifest.json").read_text())
    if cache_record["status"] != "prepared":
        raise ValueError("condition cache is incomplete")
    names = [
        f"waveforms_q{factor}.npy",
        "base_mask.parquet",
        "geometry.parquet",
        "probe_cells.npy",
        "regsi_real.yaml",
    ]
    verify_hashes(cache, {name: cache_record["artifacts"][name] for name in names})
    mask = nested_condition_mask(pd.read_parquet(cache / "base_mask.parquet"), missing_fraction)
    geometry = pd.read_parquet(cache / "geometry.parquet")
    values = np.load(cache / f"waveforms_q{factor}.npy", mmap_mode="r")
    time_s = np.arange(values.shape[1]) * factor * cache_record["original_sample_interval_us"] / 1e6
    query_cells = (
        mask.loc[mask.split.eq("test"), "cell_id"].to_numpy()
        if all_test
        else np.load(cache / "probe_cells.npy")
    )
    inputs, scales = condition_model_inputs(
        geometry,
        mask,
        values,
        cache_record["spatial_shape"],
        time_s,
        normalization,
        {"cache_hash": sha256_file(cache / "manifest.json")},
    )
    config = yaml.safe_load((cache / "regsi_real.yaml").read_text())
    config["source_preparation_metadata"] = {
        key: config.pop(key)
        for key in (
            "amplitude_scaling",
            "trace_scale_hash",
            "idw",
            "mask_hash",
            "mask_id",
            "global_rms",
            "changes_from_source",
            "checkpoint_rule",
        )
        if key in config
    }
    config.update(
        amplitude_scaling=normalization,
        global_rms=inputs.global_rms,
        mask_id=f"nested_missing_{missing_fraction:g}",
        checkpoint_rule="fixed_final_raw_and_ema",
    )
    config["device"] = device
    config["training"].update(
        max_steps=steps,
        query_batch_size=batch_size,
        inner_mask_fraction=inner_fraction,
        learning_rate=learning_rate,
    )
    config["prediction"]["query_batch_size"] = batch_size
    config["model"]["max_edge_time_shift_samples"] = time_shift
    output, record = _new_run(repo, f"q{factor}_missing{missing_fraction:g}_{normalization}")
    record.update(
        cache=str(cache.relative_to(repo)),
        cache_hash=sha256_file(cache / "manifest.json"),
        factor=factor,
        missing_fraction=missing_fraction,
        normalization=normalization,
        oracle_test_rms=normalization == "trace_oracle",
        all_test=all_test,
        note="Oracle-RMS diagnostic; held-out waveforms never enter training or model inputs.",
    )
    mask.to_parquet(output / "mask.parquet", index=False)
    config["mask_hash"] = sha256_file(output / "mask.parquet")
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    write_json(output / "manifest.json", record)
    started = perf_counter()
    try:
        result = train_regsi_condition(
            inputs, config, query_cells, reporter=lambda message: print(message, flush=True)
        )
        pd.DataFrame(result["history"]).to_parquet(output / "training_history.parquet", index=False)
        index = mask.set_index("cell_id").loc[query_cells].reset_index()
        index.to_parquet(output / "evaluation_index.parquet", index=False)
        positions = pd.Index(mask.cell_id).get_indexer(query_cells)
        target = np.asarray(values[positions], dtype=np.float32)
        np.save(output / "target.npy", target, allow_pickle=False)
        np.save(output / "evaluation_rms.npy", scales[positions], allow_pickle=False)
        summaries = {}
        for variant, prediction in result["predictions"].items():
            # Global-mode prediction is already physical; trace-mode output has unit-RMS labels.
            if normalization == "trace_oracle":
                prediction = (prediction.astype(float) * scales[positions, None]).astype(np.float32)
            np.save(output / f"prediction_{variant}.npy", prediction, allow_pickle=False)
            torch.save(result["states"][variant], output / f"state_{variant}.pt")
            metrics, traces = evaluate_traces(prediction, target, index)
            verify_saved_metrics(metrics, traces)
            write_json(output / f"metrics_{variant}.json", metrics)
            traces.to_parquet(output / f"metrics_by_trace_{variant}.parquet", index=False)
            summaries[variant] = metrics
        write_json(output / "summary.json", summaries)
        history = pd.DataFrame(result["history"])
        record.update(
            status="completed",
            runtime_seconds=perf_counter() - started,
            observed_trace_count=len(inputs.observed_cells),
            test_trace_count=len(inputs.test_cells),
            evaluated_trace_count=len(query_cells),
            initialization_hash=result["initialization_hash"],
            query_presentations=result["query_presentations"],
            no_context_training_queries=result["no_context_query_count"],
            no_context_test_queries=int((~result["context"]).sum()),
            mean_last_500_training_loss=float(history.loss.tail(500).mean()),
        )
        print(
            json.dumps({key: value["clean_target"] for key, value in summaries.items()}), flush=True
        )
    except Exception as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        _finish(output, record)
    return output
