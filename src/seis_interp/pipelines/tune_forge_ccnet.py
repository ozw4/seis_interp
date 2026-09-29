"""Tune CCNet-5D parameters while preserving a completed FORGE input contract."""

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
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces
from seis_interp.training.forge_mvp_methods import run_method

TIME_AXIS_MODEL_KEYS = ("time_kernel_size", "time_dilation")
OPTIONAL_TRAINING_KEYS = ("ema_decay", "supervised_traces_per_update")


def tune_forge_ccnet(repo: Path, source_run: Path, search_config: Path):
    """Record every candidate and explicitly label selection on the original test set."""
    search = yaml.safe_load(search_config.read_text())
    matrix = json.loads((source_run / "matrix_manifest.json").read_text())
    preparation = repo / matrix["preparation"]
    verify_hashes(preparation, {"preparation_manifest.json": matrix["preparation_hash"]})
    inputs, base_config = load_model_inputs(preparation, "ccnet5d_grid")
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo / "runs" / contract["study_id"] / f"{stamp}_{identity['git_commit'][:12]}_ccnet_tuning"
    )
    output.mkdir(parents=True, exist_ok=False)
    print(output, flush=True)
    (output / "search.yaml").write_bytes(search_config.read_bytes())
    source_hashes = {}
    for source in [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / "scripts/forge_ccnet_tuning.py",
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
        source_run=str(source_run.relative_to(repo)),
        preparation=str(preparation.relative_to(repo)),
        preparation_hash=matrix["preparation_hash"],
        input_binding=inputs.binding,
        implementation_hashes=source_hashes,
        selection_domain="original_clean_target_test_set",
        independent_test_estimate=False,
        target_relative_mse=search["target_relative_mse"],
        baseline_metrics=json.loads((source_run / "runs/ccnet5d_grid/metrics.json").read_text()),
    )
    write_json(output / "manifest.json", record)
    try:
        summaries = _evaluate_candidates(repo, preparation, inputs, base_config, search, output)
        best = min(summaries, key=lambda row: row["primary_relative_mse"])
        record.update(
            status="completed",
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


def _candidate_config(base_config, candidate, device):
    """Allow only CCNet-internal parameters; the loss, optimizer and full-trace patch stay fixed."""
    if set(candidate) - {"name", "model", "training", "patches", "prediction", "alignment"}:
        raise ValueError("only CCNet model, training, patch and prediction parameters may be tuned")
    config = deepcopy(base_config)
    if "alignment" in candidate:
        if set(candidate["alignment"]) != {"velocity_m_per_s"}:
            raise ValueError("alignment accepts exactly velocity_m_per_s")
        config["alignment"] = dict(candidate["alignment"])
    for section in ("model", "training", "patches", "prediction"):
        overrides = candidate.get(section, {})
        supported = set(config[section])
        if section == "model":
            supported.update(TIME_AXIS_MODEL_KEYS)
        if section == "training":
            supported.update(OPTIONAL_TRAINING_KEYS)
        if set(overrides) - supported:
            raise ValueError(f"unsupported CCNet {section} parameters")
        config[section].update(overrides)
    if config["model"]["output_activation"] != base_config["model"]["output_activation"]:
        raise ValueError("CCNet tuning must preserve output_activation")
    for key in ("optimizer", "learning_rate_schedule", "mixed_precision"):
        if config["training"][key] != base_config["training"][key]:
            raise ValueError(f"unsupported CCNet change: {key}")
    if config["patches"]["shape"][0] != base_config["patches"]["shape"][0]:
        raise ValueError("CCNet loss must use complete traces")
    config["checkpoint_rule"] = (
        "fixed_final_ema" if config["training"].get("ema_decay") else "fixed_final"
    )
    config["device"] = device
    config["changes_from_source"] = [
        "Tune only CCNet parameters with the source preparation and evaluation domain fixed."
    ]
    return config


def _evaluate_candidates(repo, preparation, inputs, base_config, search, output):
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    index = mask.loc[mask.split.eq("test")].reset_index(drop=True)
    index["prediction_row"] = np.arange(len(index))
    index.to_parquet(output / "prediction_index.parquet", index=False)
    target = None
    summaries = []
    torch.set_num_threads(4)
    for candidate in search["candidates"]:
        name = candidate["name"]
        config = _candidate_config(base_config, candidate, search["device"])
        path = output / name
        path.mkdir(exist_ok=False)
        (path / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Starting {name}: {candidate}", flush=True)
        started = perf_counter()
        with (path / "training.log").open("w") as log:

            def report(message):
                line = f"{utc_now()} {message}"
                print(line, file=log, flush=True)
                print(line, flush=True)

            result = run_method(inputs, config, reporter=report)
        order = validate_predictions(result.prediction, result.cell_ids, mask, len(inputs.time_s))
        prediction = result.prediction[order]
        runtime = perf_counter() - started
        np.save(path / "prediction.npy", prediction, allow_pickle=False)
        torch.save(result.checkpoint, path / "final.pt")
        pd.DataFrame(result.history).to_parquet(path / "training_log.parquet", index=False)
        if target is None:
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
            target = resample_forge_traces(target, contract)
        metrics, traces = evaluate_traces(prediction, target, index)
        verify_saved_metrics(metrics, traces)
        verify_metrics_from_waveforms(metrics, prediction, target, index.clean_target.to_numpy())
        write_json(path / "metrics.json", metrics)
        traces.to_parquet(path / "metrics_by_trace.parquet", index=False)
        row = dict(
            name=name,
            primary_relative_mse=metrics["clean_target"]["masked_trace_relative_mse"],
            all_eligible_relative_mse=metrics["all_eligible"]["masked_trace_relative_mse"],
            runtime_seconds=runtime,
            final_batch_training_loss=result.history[-1]["loss"],
            **result.diagnostics,
        )
        write_json(path / "run_manifest.json", row)
        summaries.append(row)
        pd.DataFrame(summaries).to_csv(output / "search_results.csv", index=False)
        print(json.dumps(row), flush=True)
        del result, prediction
        gc.collect()
        if search["device"].startswith("cuda"):
            torch.cuda.empty_cache()
        if (
            search.get("stop_when_target_met", False)
            and row["primary_relative_mse"] <= search["target_relative_mse"]
        ):
            break
    return summaries
