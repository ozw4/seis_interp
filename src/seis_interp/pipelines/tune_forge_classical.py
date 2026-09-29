"""Reproducible POCS and DRR parameter diagnostics on a completed FORGE run."""

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
from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
from seis_interp.processing.drr_windows import interpolate_drr_volume
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces
from seis_interp.processing.pocs_batched import interpolate_pocs_volume_batched
from seis_interp.processing.pocs_windows import interpolate_pocs_volume


def tune_forge_classical(repo: Path, source_run: Path, search_config: Path, *, method_id: str):
    """Keep the original mask and evaluator; record test-set tuning explicitly."""
    if method_id not in ("pocs_grid", "drr_grid"):
        raise ValueError("method_id must be pocs_grid or drr_grid")
    method = method_id.removesuffix("_grid")
    search = yaml.safe_load(search_config.read_text())
    matrix = json.loads((source_run / "matrix_manifest.json").read_text())
    preparation = repo / matrix["preparation"]
    verify_hashes(preparation, {"preparation_manifest.json": matrix["preparation_hash"]})
    inputs, base_config = load_model_inputs(preparation, method_id)
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo
        / "runs"
        / contract["study_id"]
        / (f"{stamp}_{identity['git_commit'][:12]}_{method}_tuning")
    )
    output.mkdir(parents=True, exist_ok=False)
    print(output, flush=True)
    (output / "search.yaml").write_bytes(search_config.read_bytes())
    source_hashes = {}
    for source in [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / f"scripts/forge_{method}_tuning.py",
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
        baseline_metrics=json.loads((source_run / "runs" / method_id / "metrics.json").read_text()),
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


def _evaluate_candidates(repo, preparation, inputs, base_config, search, output):
    if base_config.get("alignment") is not None:
        raise ValueError("aligned preparations are tuned with tune_forge_matched")
    volume = inputs.model_space().volume()
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
        path = output / name
        path.mkdir(exist_ok=False)
        config = deepcopy(base_config)
        config["algorithm"] = candidate["algorithm"]
        config["device"] = search["device"]
        config["execution_backend"] = search["backend"]
        if base_config["method_id"] == "drr_grid" and search["backend"] == "batched":
            config["execution_workers"] = search.get("workers", 1)
        (path / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Starting {name}: {candidate['algorithm']}", flush=True)
        started = perf_counter()
        kwargs = dict(candidate["algorithm"])
        function = interpolate_pocs_volume
        is_drr = base_config["method_id"] == "drr_grid"
        if is_drr:
            function = interpolate_drr_volume
            kwargs["time_s"] = inputs.time_s
        if search["backend"] == "batched":
            function = interpolate_drr_volume_batched if is_drr else interpolate_pocs_volume_batched
            kwargs["device"] = search["device"]
            if is_drr:
                kwargs["workers"] = search.get("workers", 1)
                kwargs["reporter"] = lambda message: print(message, flush=True)
        elif search["backend"] != "numpy":
            raise ValueError("backend must be numpy or batched")
        result = function(volume.values, volume.observed_trace_mask, **kwargs)
        uncovered = (
            result.uncovered_trace_count * len(inputs.time_s)
            if is_drr
            else result.uncovered_sample_count
        )
        if uncovered:
            raise ValueError("prediction has uncovered samples")
        np.testing.assert_array_equal(
            result.values[:, volume.observed_trace_mask],
            volume.values[:, volume.observed_trace_mask],
        )
        prediction = result.values.reshape(len(inputs.time_s), -1)[:, inputs.test_cells].T
        if inputs.test_rms is not None:
            prediction = (prediction.astype(np.float64) * inputs.test_rms[:, None]).astype(
                np.float32
            )
        else:
            prediction = prediction.copy()
        validate_predictions(prediction, inputs.test_cells, mask, len(inputs.time_s))
        runtime = perf_counter() - started
        np.save(path / "prediction.npy", prediction, allow_pickle=False)
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
            block_count=result.block_count,
            observed_max_abs_error=0.0,
            uncovered_sample_count=uncovered,
        )
        write_json(path / "run_manifest.json", dict(**row, algorithm=candidate["algorithm"]))
        summaries.append(row)
        pd.DataFrame(summaries).to_csv(output / "search_results.csv", index=False)
        print(json.dumps(row), flush=True)
        del result, prediction
        if (
            search.get("stop_when_target_met", False)
            and row["primary_relative_mse"] <= search["target_relative_mse"]
        ):
            break
    return summaries
