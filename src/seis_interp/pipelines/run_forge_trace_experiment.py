"""Run one frozen C3-selected setting per method on a shared FORGE trace split."""

import fcntl
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import peak_memory, utc_now
from seis_interp.data.forge_trace_experiment import (
    load_forge_trace_inputs,
    read_forge_trace_targets,
)
from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces, validate_predictions
from seis_interp.evaluation.forge_shot_results import finalize_forge_shot_experiment
from seis_interp.evaluation.trace_snr import trace_snr_db
from seis_interp.processing.c3_region_search import resolve_c3_candidate
from seis_interp.training.cuda_run_slot import cuda_run_slot
from seis_interp.training.forge_c3_methods import run_forge_c3_method


def write_forge_worker_state(path, state):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, state)
    temporary.replace(path)


def run_forge_trace_method(repo: Path, experiment: Path, method: str):
    """Wait for execution memory, train once, then open validation/test scoring labels."""
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    if method not in contract["methods"]:
        raise ValueError("method is not active in this experiment")
    config = yaml.safe_load((experiment / "study" / f"{method}.yaml").read_text())
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    path = workers / f"{method}.json"
    state = {"method_id": method, "pid": os.getpid(), "status": "running"}

    def waiting(details):
        state.update(phase="waiting_for_gpu", resource_wait=details)
        write_forge_worker_state(path, state)

    with (workers / f"{method}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_forge_worker_state(path, state)
        try:
            minimum = (
                contract.get("cuda_execution", {}).get("minimum_free_memory_gib", {}).get(method, 0)
            )
            with cuda_run_slot(workers, config["device"], minimum, waiting):
                state.pop("resource_wait", None)
                state.update(phase="training_or_reconstruction")
                write_forge_worker_state(path, state)

                def report(message):
                    state["last_message"] = message
                    match = re.search(r"step (\d+)/(\d+)", message)
                    if match:
                        state["updates_completed"] = int(match[1])
                        if match[1] == match[2]:
                            state["phase"] = "prediction"
                    write_forge_worker_state(path, state)

                record = execute_forge_trace_method(repo, experiment, method, report)
            state.update(status=record["status"], phase="completed", metrics=record["metrics"])
            if contract.get("mask_family") not in ("periodic_shot", "random_shot"):
                summarize_forge_trace_experiment(experiment)
            return record
        except BaseException as error:
            state.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
            if contract.get("mask_family") in ("periodic_shot", "random_shot"):
                manifest = experiment / "method_runs" / method / "manifest.json"
                if not manifest.exists():
                    write_json(
                        manifest,
                        {
                            **state,
                            "ended_at": utc_now(),
                            "artifacts": {},
                            "metrics": {},
                            "test_scoring_calls": 0,
                        },
                    )
            raise
        finally:
            state["updated_at"] = utc_now()
            write_forge_worker_state(path, state)
            if contract.get("mask_family") in ("periodic_shot", "random_shot"):
                finalize_forge_shot_experiment(experiment)


def execute_forge_trace_method(repo, experiment, method, progress=None):
    """Finalize the complete prediction before performing either scoring pass."""
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    whole_shot = contract.get("mask_family") in ("periodic_shot", "random_shot")
    output = experiment / "method_runs" / method
    if output.exists():
        record = json.loads((output / "manifest.json").read_text())
        if record["status"] != "completed" and not (whole_shot and record["status"] == "predicted"):
            raise ValueError("an unfinished method run is never overwritten")
        verify_hashes(output, record["artifacts"])
        return record
    inputs, config, partition = load_forge_trace_inputs(experiment, method)
    nominal_mask = config.get("training", {}).get("inner_mask_fraction")
    config = resolve_c3_candidate(
        config, len(inputs.observed_cells), contract["matched_training"]["loss_traces_per_update"]
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    record = {
        "method_id": method,
        "status": "running",
        "started_at": utc_now(),
        "input_binding": inputs.binding,
        "model_seed": config["model_seed"],
        "observed_trace_count": len(inputs.observed_cells),
        "test_scoring_calls": 0,
        "metrics": {},
    }
    if whole_shot:
        record["selection_source"] = contract["fixed_settings"][method]
        if nominal_mask is not None:
            record["inner_mask_fraction"] = {
                "nominal": nominal_mask,
                "effective": config["training"]["inner_mask_fraction"],
            }
    write_json(output / "manifest.json", record)

    def report(message):
        with (output / "execution.log").open("a") as stream:
            stream.write(f"{utc_now()} {message}\n")
        print(f"{method}: {message}", flush=True)
        if progress is not None:
            progress(message)

    started = time.monotonic()
    try:
        if config["device"].startswith("cuda:"):
            torch.cuda.set_device(config["device"])
            torch.cuda.reset_peak_memory_stats(config["device"])
        options = {"checkpoint_path": output / "final.pt"} if method == "ccnet5d_grid" else {}
        result = run_forge_c3_method(inputs, config, report, **options)
        seal = json.loads((experiment / "experiment.json").read_text())
        if (
            method in seal["parameter_counts"]
            and result.diagnostics["parameter_count"] != seal["parameter_counts"][method]
        ):
            raise ValueError("runtime parameter count differs from the frozen setting")
        mask = partition.copy()
        mask["split"] = np.where(mask.role.eq("train"), "observed", "test")
        validate_predictions(result.prediction, result.cell_ids, mask, len(inputs.time_s))
        np.save(output / "prediction.npy", result.prediction, allow_pickle=False)
        np.save(output / "cell_ids.npy", result.cell_ids, allow_pickle=False)
        if result.checkpoint is not None:
            torch.save(result.checkpoint, output / "final.pt")
        pd.DataFrame(result.history).to_csv(output / "history.csv", index=False)
        record.update(diagnostics=result.diagnostics, prediction_finalized_at=utc_now())
        write_json(output / "manifest.json", record)
        if whole_shot:
            record["status"] = "predicted"
            report("prediction finalized; waiting for all method outcomes before test scoring")
            return record
        positions = pd.Index(result.cell_ids)
        for domain in ("validation", "test"):
            rows, targets = read_forge_trace_targets(experiment, domain)
            indices = positions.get_indexer(rows.cell_id)
            if np.any(indices < 0):
                raise ValueError("prediction is missing scoring cells")
            predicted = result.prediction[indices]
            metrics, traces = evaluate_traces(predicted, targets, rows)
            snr = trace_snr_db(targets, predicted)
            for name, selected in (
                ("all_eligible", np.ones(len(rows), dtype=bool)),
                ("clean_target", rows.clean_target.to_numpy()),
            ):
                values = snr[selected]
                metrics[name]["mean_trace_snr_db"] = (
                    float(values.mean()) if np.isfinite(values).all() else None
                )
            traces.to_parquet(output / f"{domain}_per_trace.parquet", index=False)
            record["metrics"][domain] = metrics
            if domain == "test":
                record["test_scoring_calls"] += 1
        record.update(
            status="completed",
            validation_score=record["metrics"]["validation"]["clean_target"][
                "masked_trace_relative_mse"
            ],
        )
        report(f"completed V={record['validation_score']:.10g}")
    except BaseException as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(
            ended_at=utc_now(),
            runtime_seconds=time.monotonic() - started,
            **peak_memory(config["device"]),
        )
        record["artifacts"] = {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_forge_worker_state(output / "manifest.json", record)
        if record["status"] == "completed":
            for path in output.iterdir():
                if path.is_file():
                    path.chmod(0o444)
        gc.collect()
        if config["device"].startswith("cuda:"):
            torch.cuda.empty_cache()
    return record


def summarize_forge_trace_experiment(experiment):
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    if contract.get("mask_family") in ("periodic_shot", "random_shot"):
        return finalize_forge_shot_experiment(experiment)
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    with (workers / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = []
        for path in sorted((experiment / "method_runs").glob("*/manifest.json")):
            record = json.loads(path.read_text())
            if record["status"] != "completed":
                continue
            for domain, metrics in record["metrics"].items():
                for population, values in metrics.items():
                    records.append(
                        {
                            "method_id": record["method_id"],
                            "domain": domain,
                            "population": population,
                            **{
                                key: value
                                for key, value in values.items()
                                if not isinstance(value, dict)
                            },
                        }
                    )
        pd.DataFrame(records).to_csv(experiment / "summary.csv", index=False)


def launch_forge_trace_experiment(repo: Path, experiment: Path):
    """Detach one worker per method; CUDA workers serialize and wait for free memory."""
    seal = json.loads((experiment / "experiment.json").read_text())
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    if (experiment / "launch.json").exists():
        raise ValueError("the experiment has already been launched")
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1"
    )
    environment["PYTHONPATH"] = str(experiment / "source/src")
    launched = {}
    for method in seal["methods"]:
        with (workers / f"{method}.log").open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(experiment / "source/scripts/forge_trace_experiment.py")
                    if (experiment / "source/scripts/forge_trace_experiment.py").is_file()
                    else str(repo / "scripts/forge_trace_experiment.py"),
                    "worker",
                    "--experiment",
                    str(experiment),
                    "--method",
                    method,
                ],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched[method] = process.pid
    write_json(experiment / "launch.json", {"launched_at": utc_now(), "pids": launched})
    return launched
