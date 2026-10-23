"""Orchestrate FORGE validation-only S1, the method barrier, and three fresh splits."""

import fcntl
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.experiment_import_coverage import s1_candidates_without_admission
from seis_interp.data.experiment_imports import (
    find_prediction_import,
    imported_prediction,
    saved_input_fingerprint,
)
from seis_interp.data.forge_grid_inputs import load_forge_grid_inputs, read_forge_grid_targets
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import peak_memory, utc_now
from seis_interp.data.run_memory import RunRSSSampler
from seis_interp.evaluation.forge_grid_results import (
    forge_seed_metric_rows,
    score_forge_grid,
    summarize_forge_grid_seeds,
)
from seis_interp.evaluation.forge_mvp_metrics import validate_predictions
from seis_interp.processing.c3_region_search import (
    c3_search_candidates,
    resolve_c3_candidate,
    select_c3_candidate,
)
from seis_interp.training.candidate_schedule import candidate_outcomes
from seis_interp.training.exact_resume import atomic_torch_save
from seis_interp.training.forge_c3_methods import run_forge_c3_method
from seis_interp.training.forge_mvp_methods import ForgeMethodResult
from seis_interp.training.gpu_admission import candidate_admission, memory_fitting_candidates
from seis_interp.training.rehearsal import Rehearsal
from seis_interp.training.run_recovery import (
    S1BarrierError,
    check_free_disk,
    failure_kind,
    gpu_utilization,
    require_finite_metrics,
    requires_process_restart,
    retry_run,
    write_run_manifest,
)
from seis_interp.training.s1_barrier import s1_barrier_ready
from seis_interp.training.worker_processes import (
    process_start_time,
    report_admission,
    report_admission_wait,
    source_root,
    wait_for_workers,
    write_worker_state,
)
from seis_interp.training.worker_restarts import (
    refuse_live_workers,
    restart_exited_worker,
    start_dead_workers,
)
from seis_interp.training.worker_shards import (
    collect_shard_records,
    parse_shard,
    shard_candidates,
    shard_worker_name,
)


def _write_state(path, state):
    write_worker_state(path, state)


def forge_s1_barrier_ready(experiment):
    seal = json.loads((experiment / "experiment.json").read_text())
    return s1_barrier_ready(experiment, {method: experiment for method in seal["methods"]})


@retry_run
def execute_forge_grid_run(repo, experiment, candidate, seed, stage, *, prediction_only=False):
    if stage not in ("S1", "evaluation"):
        raise ValueError("only S1 and three-seed evaluation are supported")
    if stage == "evaluation" and not prediction_only and not forge_s1_barrier_ready(experiment):
        method_id = candidate["config"]["method_id"]
        if not (experiment / "workers" / f"{method_id}_selection.json").exists():
            raise S1BarrierError("all methods must finish S1 before three-seed evaluation")
        while not forge_s1_barrier_ready(experiment):
            time.sleep(10)
    method = candidate["config"]["method_id"]
    name = candidate["id"] if stage == "S1" else f"seed_{seed}"
    output = experiment / "method_runs" / method / stage / name
    rehearsal = Rehearsal(experiment)
    attempt = len(list((output / "attempts").iterdir())) if (output / "attempts").exists() else 0
    saved_result = None
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous["status"] in ("completed", "diverged"):
            verify_hashes(output, previous["artifacts"])
            return previous
        if previous["status"] == "awaiting_barrier":
            if prediction_only:
                return previous
            saved_result = torch.load(output / "prediction_result.pt", weights_only=False)
        elif previous["status"] != "retrying":
            raise ValueError(f"unfinished run is never overwritten: {output}")
    domains = ("validation",) if stage == "S1" else ("validation", "test")
    inputs, base, partition = load_forge_grid_inputs(experiment, method, seed, domains)
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    config = resolve_c3_candidate(
        candidate["config"] | {"alignment": base["alignment"]},
        len(inputs.observed_cells),
        contract["matched_training"]["loss_traces_per_update"],
    )
    input_hash = saved_input_fingerprint(experiment, seed, method, domains)
    provenance = find_prediction_import(
        repo, contract.get("import_from", []), config, input_hash, seed, stage
    )
    admission_record = {}

    def waiting(details):
        report_admission_wait(experiment, method, name, details)

    def admitted(details):
        admission_record.update(details)
        report_admission(experiment, method, name, stage, details)

    admission = (
        nullcontext(config["device"])
        if provenance is not None or saved_result is not None
        else candidate_admission(
            repo,
            config,
            contract.get("cuda_execution"),
            candidate["id"],
            waiting,
            admitted=admitted,
            reservation=candidate.get("admission_reservation"),
        )
    )
    rehearsal.inject("pre_admission", method, name, attempt)
    free_disk = check_free_disk(
        experiment, contract.get("cuda_execution", {}).get("minimum_free_disk_gib", 5.0)
    )
    with admission as device:
        free_disk = check_free_disk(
            experiment, contract.get("cuda_execution", {}).get("minimum_free_disk_gib", 5.0)
        )
        config["device"] = device
        output.mkdir(parents=True, exist_ok=True)
        record = {
            "id": candidate["id"],
            "method_id": method,
            "stage": stage,
            "trace_seed": seed,
            "model_seed": config["model_seed"],
            "axis_a": candidate["axis_a"],
            "axis_b": candidate["axis_b"],
            "status": "running",
            "pid": os.getpid(),
            "source_root": source_root(),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "threads_at_start": len(list(Path("/proc/self/task").iterdir())),
            "started_at": utc_now(),
            "input_binding": inputs.binding,
            "input_hash": input_hash,
            "observed_trace_count": len(inputs.observed_cells),
            "test_scoring_calls": 0,
            "metrics": {},
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "cuda_admission": admission_record,
            "free_disk_bytes_at_admission": free_disk,
            "external_gpu_utilization_at_admission": gpu_utilization(device),
            "attempt": len(list((output / "attempts").iterdir()))
            if (output / "attempts").exists()
            else 0,
        }

        def save_config():
            (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            record["config_sha256"] = sha256_file(output / "config.resolved.yaml")
            write_run_manifest(output / "manifest.json", record)

        if saved_result is not None:
            record["prediction_finalized_at"] = previous.get("prediction_finalized_at") or utc_now()
            if previous.get("prediction_execution") is not None:
                record["prediction_execution"] = previous["prediction_execution"]

        watchdog = rehearsal.watchdog(
            output, config, resumable=provenance is None and saved_result is None
        )

        def report(message):
            watchdog.report(message)
            with (output / "execution.log").open("a") as stream:
                stream.write(f"{utc_now()} {message}\n")
            print(f"{method} {stage} {name}: {message}", flush=True)
            rehearsal.report(message, method, name, attempt)

        save_config()
        rss = RunRSSSampler()
        rss.start()
        started = time.monotonic()
        watchdog.start()
        try:
            if saved_result is not None:
                result = saved_result
            elif provenance is not None:
                record["imported_from"] = provenance
                result = ForgeMethodResult(
                    imported_prediction(provenance, inputs.test_cells),
                    inputs.test_cells.copy(),
                    [],
                    None,
                    {"imported": True},
                )
                source_checkpoint = Path(provenance["source_path"]) / "final.pt"
                if source_checkpoint.exists():
                    shutil.copyfile(source_checkpoint, output / "final.pt")
            else:
                if device.startswith("cuda:"):
                    torch.cuda.set_device(device)
                    torch.cuda.reset_peak_memory_stats(device)
                threads = config.get("execution", {}).get("cpu_threads")
                if threads is not None:
                    torch.set_num_threads(threads)
                options = (
                    {"checkpoint_path": output / "final.pt"} if method == "ccnet5d_grid" else {}
                )
                if method not in ("pocs_grid", "drr_grid"):
                    options["recovery"] = rehearsal.recovery(output, method, name, attempt, report)
                rehearsal.inject("pre_method", method, name, attempt)
                result = run_forge_c3_method(inputs, config, report, **options)
                seal = json.loads((experiment / "experiment.json").read_text())
                parameters = seal["parameter_counts"][method]
                if parameters and result.diagnostics["parameter_count"] != parameters:
                    raise ValueError("runtime parameter count differs from the frozen search")
                record["parameter_count"] = parameters
            rehearsal.inject("after_prediction", method, name, attempt, result=result)
            mask = partition.copy()
            mask["split"] = np.select(
                [mask.role.eq("train"), mask.role.isin(domains)], ["observed", "test"], "excluded"
            )
            validate_predictions(result.prediction, result.cell_ids, mask, len(inputs.time_s))
            np.save(output / "prediction.npy", result.prediction, allow_pickle=False)
            np.save(output / "cell_ids.npy", result.cell_ids, allow_pickle=False)
            if result.checkpoint is not None:
                torch.save(result.checkpoint, output / "final.pt")
            pd.DataFrame(result.history).to_csv(output / "history.csv", index=False)
            record["diagnostics"] = result.diagnostics
            if saved_result is None:
                record["prediction_finalized_at"] = utc_now()
            write_run_manifest(output / "manifest.json", record)
            if stage == "evaluation" and prediction_only:
                atomic_torch_save(result, output / "prediction_result.pt")
                record["status"] = "awaiting_barrier"
                return record
            if stage == "evaluation" and not forge_s1_barrier_ready(experiment):
                raise RuntimeError("test scoring requires the S1 barrier")
            positions = pd.Index(result.cell_ids)
            for domain in domains:
                report(f"reading {domain} targets seed={seed}")
                rows, target = read_forge_grid_targets(experiment, partition, domain)
                indices = positions.get_indexer(rows.cell_id)
                if np.any(indices < 0):
                    raise ValueError("prediction does not cover current scoring domain")
                metrics, traces = score_forge_grid(result.prediction[indices], target, rows)
                traces.to_parquet(output / f"{domain}_per_trace.parquet", index=False)
                require_finite_metrics(metrics)
                record["metrics"][domain] = metrics
                if result.shadow_predictions:
                    record.setdefault("ema_shadow_metrics", {})[domain] = {
                        decay: score_forge_grid(values[indices], target, rows)[0]
                        for decay, values in result.shadow_predictions.items()
                    }
                    require_finite_metrics(record["ema_shadow_metrics"][domain])
                if domain == "test":
                    record["test_scoring_calls"] += 1
                    write_run_manifest(output / "manifest.json", record)
                    rehearsal.inject("after_test_scored", method, name, attempt, result=result)
            (output / "prediction_result.pt").unlink(missing_ok=True)
            record.update(
                status="completed",
                validation_score=record["metrics"]["validation"]["clean_target"][
                    "masked_trace_relative_mse"
                ],
            )
        except BaseException as error:
            record.update(
                status="diverged" if failure_kind(error) == "diverged" else "failed",
                failure_reason=f"{type(error).__name__}: {error}",
                failure_kind=failure_kind(error),
            )
            if record["status"] != "diverged":
                raise
        finally:
            watchdog.close()
            memory_device = (
                config["device"]
                if isinstance(config["device"], str)
                and config["device"] != "auto"
                and provenance is None
                and saved_result is None
                and not requires_process_restart(RuntimeError(record.get("failure_reason", "")))
                else "cpu"
            )
            record.update(
                ended_at=utc_now(),
                runtime_seconds=time.monotonic() - started,
                **peak_memory(memory_device),
            )
            record.update(rss.stop())
            record["threads_at_end"] = len(list(Path("/proc/self/task").iterdir()))
            if prediction_only:
                record["prediction_execution"] = {
                    "pid": record["pid"],
                    "source_root": record["source_root"],
                    "started_at": record.get("started_at", record.get("start_time")),
                    "ended_at": record.get("ended_at", record.get("end_time")),
                    "cuda_admission": record["cuda_admission"],
                    "external_gpu_utilization_at_admission": record[
                        "external_gpu_utilization_at_admission"
                    ],
                    "peak_cuda_reserved_bytes": record["peak_cuda_reserved_bytes"],
                }

            record["artifacts"] = {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name not in ("manifest.json", "retry_state.json")
            }
            write_run_manifest(output / "manifest.json", record)
            if record["status"] == "completed":
                for path in output.iterdir():
                    if path.is_file():
                        path.chmod(0o444)
            gc.collect()
            if memory_device.startswith("cuda:"):
                torch.cuda.empty_cache()
        return record


def _search_run(candidate, *, repo, experiment, seed):
    return execute_forge_grid_run(repo, experiment, candidate, seed, "S1")


def execute_forge_grid_pilot(repo, experiment, method, candidate_id):
    """Run specified S1 cells through production scoring, without selection or test."""
    seal = json.loads((experiment / "experiment.json").read_text())
    if method not in seal["methods"]:
        raise ValueError("inactive grid method")
    verify_hashes(
        experiment,
        {
            name: digest
            for name, digest in seal["artifacts"].items()
            if not name.startswith("input/")
        },
    )
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    base = yaml.safe_load((experiment / "study" / f"{method}.yaml").read_text())
    candidates = c3_search_candidates(base, grid["methods"][method])
    candidates = Rehearsal(experiment).candidates(candidates, contract, method)
    selected = [item for item in candidates if candidate_id in ("all", item["id"])]
    if not selected:
        raise ValueError("unknown pilot candidate ID")
    return [
        execute_forge_grid_run(repo, experiment, item, contract["trace_split"]["search_seed"], "S1")
        for item in selected
    ]


def run_forge_grid_worker(repo, experiment, method, *, shard="0/1"):
    seal = json.loads((experiment / "experiment.json").read_text())
    if method not in seal["methods"]:
        raise ValueError("inactive grid method")
    # S1 must not hash/read the held-out waveform file during execution.
    verify_hashes(
        experiment,
        {
            name: digest
            for name, digest in seal["artifacts"].items()
            if not name.startswith("input/")
        },
    )
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    base = yaml.safe_load((experiment / "study" / f"{method}.yaml").read_text())
    candidates = c3_search_candidates(base, grid["methods"][method])
    settings = contract.get("method_execution", {}).get(method, {})
    candidates = Rehearsal(experiment).candidates(candidates, contract, method)
    all_candidates = candidates
    candidates = shard_candidates(candidates, shard)
    worker_name = shard_worker_name(method, shard)
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    path = workers / f"{worker_name}.json"
    state = {
        "shard": shard,
        "method_id": method,
        "pid": os.getpid(),
        "process_start_time": process_start_time(os.getpid()),
        "source_root": source_root(),
        "status": "running",
        "s1_completed": (workers / f"{method}_selection.json").exists(),
        "restartable": True,
        "completed_s1": 0,
        "completed_evaluation": 0,
        "child_pids": [],
        "child_processes": [],
    }

    def children_changed(children):
        state.update(child_pids=[child["pid"] for child in children], child_processes=children)
        _write_state(path, state)

    def failed(error):
        state.update(
            status="retry_pending" if failure_kind(error) == "infrastructure" else "failed",
            failure_reason=f"{type(error).__name__}: {error}",
        )
        _write_state(path, state)

    def waiting_for_candidate(details):
        state.update(phase="waiting_for_gpu", resource_wait=details)
        state.pop("active_run", None)
        _write_state(path, state)
        print(f"{method}: waiting for GPU: {details}", flush=True)

    with (workers / f"{worker_name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_state(path, state)
        try:
            without_admission = s1_candidates_without_admission(
                repo, experiment, candidates, "forge"
            )
            search_execute = partial(
                _search_run,
                repo=repo,
                experiment=experiment,
                seed=contract["trace_split"]["search_seed"],
            )
            parallel = settings.get("parallel_candidates", 1)

            def execute(candidate):
                state.update(phase="S1", active_run=candidate["id"])
                state.pop("resource_wait", None)
                _write_state(path, state)
                return search_execute(candidate)

            records = []
            for record in candidate_outcomes(
                execute if parallel == 1 else search_execute,
                memory_fitting_candidates(
                    repo,
                    candidates,
                    contract.get("cuda_execution"),
                    without_admission=without_admission,
                    waiting=waiting_for_candidate,
                ),
                parallel,
                children_changed=children_changed,
                failed=failed,
            ):
                records.append(record)
                state.update(phase="S1", completed_s1=len(records))
                state.pop("resource_wait", None)
                _write_state(path, state)
            if parse_shard(shard)[1] > 1:
                state.update(shard_completed=True, phase="waiting_for_shards")
                _write_state(path, state)
                records = collect_shard_records(experiment, method, all_candidates, shard)
                if records is None:
                    state.update(status="completed", phase="shard_completed", s1_completed=True)
                    _write_state(path, state)
                    return
            winner = select_c3_candidate(
                all_candidates,
                records,
                expected=len(all_candidates) if seal.get("rehearsal") is True else 25,
            )
            write_json(
                workers / f"{method}_selection.json",
                {
                    "rule": "minimum_validation_clean_target_relative_mse_then_configuration_id",
                    "winner": winner,
                    "s1_records": [
                        {key: row.get(key) for key in ("id", "status", "validation_score")}
                        for row in records
                    ],
                    "test_used_for_selection": False,
                },
            )
            state.update(
                s1_completed=True,
                phase="predicting_seeds" if seal.get("rehearsal") is True else "waiting_for_all_S1",
                selected_configuration=winner["id"],
            )
            _write_state(path, state)
            seed_jobs = [
                winner | {"seed": seed} for seed in contract["trace_split"]["evaluation_seeds"]
            ]
            for _ in candidate_outcomes(
                partial(_predict_seed, repo=repo, experiment=experiment),
                seed_jobs,
                2 if base["device"] != "cpu" else 1,
                children_changed=children_changed,
            ):
                pass
            if seal.get("rehearsal") is True:
                state.update(phase="waiting_for_all_S1", active_run=None)
                _write_state(path, state)
            while not forge_s1_barrier_ready(experiment):
                time.sleep(10)
            evaluations = []
            for seed in contract["trace_split"]["evaluation_seeds"]:
                state.update(phase="evaluation", active_run=f"seed_{seed}")
                _write_state(path, state)
                evaluations.append(
                    execute_forge_grid_run(repo, experiment, winner, seed, "evaluation")
                )
                if evaluations[-1]["status"] not in ("completed", "diverged"):
                    raise RuntimeError("post-barrier scoring did not complete")
                state["completed_evaluation"] = sum(
                    row["status"] == "completed" for row in evaluations
                )
                _write_state(path, state)
            summary = summarize_forge_grid_seeds(
                evaluations, expected_seeds=contract["trace_split"]["evaluation_seeds"]
            )
            summary.to_csv(workers / f"{method}_summary.csv", index=False)
            summary.to_csv(experiment / f"{method}_summary.csv", index=False)
            state.update(status="completed", phase="completed", active_run=None, end_time=utc_now())
            _write_state(path, state)
        except BaseException as error:
            state.update(
                status="retry_pending" if failure_kind(error) == "infrastructure" else "failed",
                failure_reason=f"{type(error).__name__}: {error}",
            )
            _write_state(path, state)
            raise


def _predict_seed(candidate, *, repo, experiment):
    return execute_forge_grid_run(
        repo, experiment, candidate, candidate["seed"], "evaluation", prediction_only=True
    )


def summarize_forge_grid_experiment(experiment):
    seal = json.loads((experiment / "experiment.json").read_text())
    summaries, records = [], []
    for method in seal["methods"]:
        table = pd.read_csv(experiment / "workers" / f"{method}_summary.csv")
        table.insert(0, "method_id", method)
        summaries.append(table)
        records.extend(
            json.loads(path.read_text())
            for path in sorted(
                (experiment / "method_runs" / method / "evaluation").glob("*/manifest.json")
            )
        )
    pd.concat(summaries, ignore_index=True).to_csv(experiment / "summary.csv", index=False)
    pd.DataFrame(forge_seed_metric_rows(records)).to_csv(
        experiment / "seed_metrics.csv", index=False
    )


def supervise_forge_grid_experiment(repo, experiment, script, *, only_dead_workers=False):
    seal = json.loads((experiment / "experiment.json").read_text())
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    with (workers / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": str(experiment / "source/src"),
        }
        processes = {}
        contract_path = experiment / "study/config.yaml"
        runtime_contract = (
            yaml.safe_load(contract_path.read_text()) if contract_path.exists() else {}
        )
        jobs = {}
        for method in seal["methods"]:
            count = (
                runtime_contract.get("method_execution", {}).get(method, {}).get("worker_shards", 1)
            )
            for index in range(count):
                shard = f"{index}/{count}"
                jobs[shard_worker_name(method, shard)] = method, shard

        def start_worker(worker_name):
            method, shard = jobs[worker_name]
            with (workers / f"{worker_name}.log").open("a") as log:
                return subprocess.Popen(
                    [
                        *(["nice", "-n", "10"] if method in ("pocs_grid", "drr_grid") else []),
                        sys.executable,
                        str(script),
                        "worker",
                        "--repo",
                        str(repo),
                        "--experiment",
                        str(experiment),
                        "--method",
                        method,
                        "--shard",
                        shard,
                    ],
                    cwd=repo,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

        frozen_script = experiment / "source/scripts" / script.name
        if frozen_script.exists():
            script = frozen_script
        if only_dead_workers:
            processes = start_dead_workers(workers, jobs, start_worker)
        else:
            refuse_live_workers(workers, jobs)
            for worker_name in jobs:
                processes[worker_name] = start_worker(worker_name)

        def restart(method):
            return restart_exited_worker(workers, method, start_worker)

        _write_state(
            workers / "supervisor.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "worker_pids": {method: process.pid for method, process in processes.items()},
                "process_start_time": process_start_time(os.getpid()),
                "source_root": source_root(),
                "pythonpath": os.environ.get("PYTHONPATH"),
                "script": str(script),
                "only_dead_workers": only_dead_workers,
            },
        )
        outcomes, completed = wait_for_workers(
            processes, directory=workers, **({"restart": restart} if contract_path.exists() else {})
        )
        if completed:
            summarize_forge_grid_experiment(experiment)
        _write_state(
            workers / "supervisor.json",
            {
                "status": "completed" if completed else "failed",
                "exit_codes": outcomes,
                "source_root": source_root(),
                "pythonpath": os.environ.get("PYTHONPATH"),
                "script": str(script),
                "workers_still_running": {
                    method: processes[method].pid
                    for method, code in outcomes.items()
                    if code is None
                },
                "failure_policy": "stop_waiting_without_signalling_workers",
                "end_time": utc_now(),
            },
        )
        if not completed:
            raise RuntimeError(f"FORGE grid worker failures: {outcomes}")
