"""Run S1, enforce a five-method barrier, then evaluate three new partitions."""

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

from seis_interp.c3_fair_grid_config import C3_FAIR_GRID_METHODS
from seis_interp.data.c3_experiment_sources import (
    c3_method_experiments,
    publish_c3_replacement_readiness,
)
from seis_interp.data.c3_region_inputs import load_c3_region_inputs, read_c3_region_targets
from seis_interp.data.experiment_import_coverage import s1_candidates_without_admission
from seis_interp.data.experiment_imports import (
    find_prediction_import,
    imported_prediction,
    saved_input_fingerprint,
)
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import peak_memory, utc_now
from seis_interp.data.run_memory import RunRSSSampler
from seis_interp.evaluation.c3_region_results import score_c3_region, summarize_c3_region_seeds
from seis_interp.evaluation.forge_mvp_metrics import validate_predictions
from seis_interp.processing.c3_region_partition import region_scoring_mask
from seis_interp.processing.c3_region_search import (
    c3_search_candidates,
    c3_split_settings,
    resolve_c3_candidate,
    select_c3_candidate,
)
from seis_interp.training.c3_region_methods import run_c3_region_method
from seis_interp.training.candidate_schedule import candidate_outcomes
from seis_interp.training.cuda_run_slot import cuda_run_slot
from seis_interp.training.exact_resume import atomic_torch_save
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
    record_worker_exits,
    report_admission,
    report_admission_wait,
    restart_worker_processes,
    source_root,
    wait_for_workers,
    worker_failure,
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


def _write_status(path: Path, status: dict) -> None:
    write_worker_state(path, status)


@retry_run
def execute_c3_region_run(repo, experiment, candidate, seed, stage, *, prediction_only=False):
    """Finalize a prediction/checkpoint before granting access to its scoring labels."""
    if stage not in ("S1", "evaluation"):
        raise ValueError("only S1 and three-seed evaluation are supported")
    if stage == "evaluation" and not prediction_only and not c3_s1_barrier_ready(experiment):
        method_id = candidate["config"]["method_id"]
        if not (experiment / "workers" / f"{method_id}_selection.json").exists():
            raise S1BarrierError("all five S1 searches must finish before three-seed evaluation")
        while not c3_s1_barrier_ready(experiment):
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
    inputs, partition = load_c3_region_inputs(experiment, seed, domains)
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    kind, _ = c3_split_settings(contract)
    config = resolve_c3_candidate(
        candidate["config"],
        len(inputs.observed_cells),
        contract["matched_training"]["loss_traces_per_update"],
    )
    provenance = None
    input_hash = None
    if contract.get("import_from"):
        input_hash = saved_input_fingerprint(experiment, seed, method, domains)
        provenance = find_prediction_import(
            repo, contract["import_from"], config, input_hash, seed, stage
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
        (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        record = {
            "id": candidate["id"],
            "method_id": method,
            "stage": stage,
            f"{kind}_seed": seed,
            "model_seed": config["model_seed"],
            "axis_a": candidate["axis_a"],
            "axis_b": candidate["axis_b"],
            "status": "running",
            "pid": os.getpid(),
            "source_root": source_root(),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "threads_at_start": len(list(Path("/proc/self/task").iterdir())),
            "start_time": utc_now(),
            "config_sha256": sha256_file(output / "config.resolved.yaml"),
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
        if saved_result is not None:
            record["prediction_finalized_at"] = previous.get("prediction_finalized_at") or utc_now()
            if previous.get("prediction_execution") is not None:
                record["prediction_execution"] = previous["prediction_execution"]
        write_run_manifest(output / "manifest.json", record)
        print(f"{method} {stage} {name} starting at {output}", flush=True)

        watchdog = rehearsal.watchdog(
            output, config, resumable=provenance is None and saved_result is None
        )

        def report(message):
            watchdog.report(message)
            with (output / "execution.log").open("a") as stream:
                stream.write(f"{utc_now()} {message}\n")
            print(f"{method} {stage} {name}: {message}", flush=True)
            rehearsal.report(message, method, name, attempt)

        rss = RunRSSSampler()
        rss.start()
        started = time.monotonic()
        watchdog.start()
        try:
            if saved_result is not None:
                result = saved_result
            elif provenance is not None:
                record["imported_from"] = provenance
                seal = json.loads((experiment / "experiment.json").read_text())
                result = ForgeMethodResult(
                    imported_prediction(provenance, inputs.test_cells),
                    inputs.test_cells.copy(),
                    [],
                    None,
                    {"parameter_count": seal["parameter_counts"][method], "imported": True},
                )
                checkpoint = Path(provenance["source_path"]) / "final.pt"
                if checkpoint.exists():
                    shutil.copyfile(checkpoint, output / "final.pt")
            else:
                if device.startswith("cuda:"):
                    torch.cuda.set_device(device)
                    torch.cuda.reset_peak_memory_stats(device)
                threads = config.get("execution", {}).get("cpu_threads")
                if threads is not None:
                    torch.set_num_threads(threads)
                options = (
                    {}
                    if method in ("pocs_grid", "drr_grid")
                    else {"recovery": rehearsal.recovery(output, method, name, attempt, report)}
                )
                rehearsal.inject("pre_method", method, name, attempt)
                result = run_c3_region_method(inputs, config, report, **options)
            seal = json.loads((experiment / "experiment.json").read_text())
            parameters = seal["parameter_counts"][method]
            if parameters and result.diagnostics["parameter_count"] != parameters:
                raise RuntimeError("runtime parameter count differs from the frozen search")
            record["parameter_count"] = parameters
            rehearsal.inject("after_prediction", method, name, attempt, result=result)
            mask = region_scoring_mask(partition, domains, f"{kind}_role")
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
            if stage == "evaluation" and not c3_s1_barrier_ready(experiment):
                raise RuntimeError("test scoring requires the S1 barrier")
            positions = pd.Index(result.cell_ids)
            for domain in domains:
                report(f"reading {domain} targets seed={seed}")
                rows, target = read_c3_region_targets(repo, experiment, partition, domain)
                indices = positions.get_indexer(rows.cell_id)
                if np.any(indices < 0):
                    raise ValueError("prediction does not cover the selected scoring domain")
                summary, traces = score_c3_region(result.prediction[indices], target, rows)
                traces.to_parquet(output / f"{domain}_per_trace.parquet", index=False)
                require_finite_metrics(summary)
                record["metrics"][domain] = summary
                if result.shadow_predictions:
                    record.setdefault("ema_shadow_metrics", {})[domain] = {
                        decay: score_c3_region(values[indices], target, rows)[0]
                        for decay, values in result.shadow_predictions.items()
                    }
                    require_finite_metrics(record["ema_shadow_metrics"][domain])
                if domain == "test":
                    record["test_scoring_calls"] += 1
                    write_run_manifest(output / "manifest.json", record)
                    rehearsal.inject("after_test_scored", method, name, attempt, result=result)
                del target, traces
            (output / "prediction_result.pt").unlink(missing_ok=True)
            record.update(
                status="completed",
                validation_score=record["metrics"]["validation"]["masked_trace_relative_mse"],
            )
            report(f"completed V={record['validation_score']:.10g}")
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
            record.update(
                end_time=utc_now(),
                runtime_seconds=time.monotonic() - started,
                **peak_memory(
                    config["device"]
                    if isinstance(config["device"], str)
                    and config["device"] != "auto"
                    and provenance is None
                    and saved_result is None
                    and not requires_process_restart(RuntimeError(record.get("failure_reason", "")))
                    else "cpu"
                ),
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
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name not in ("manifest.json", "retry_state.json")
            }
            write_run_manifest(output / "manifest.json", record)
            gc.collect()
            if record["status"] == "completed":
                for path in output.iterdir():
                    if path.is_file():
                        path.chmod(0o444)
            if (
                isinstance(config["device"], str)
                and config["device"].startswith("cuda:")
                and provenance is None
                and saved_result is None
                and not requires_process_restart(RuntimeError(record.get("failure_reason", "")))
            ):
                torch.cuda.empty_cache()
        return record


def c3_s1_barrier_ready(experiment: Path) -> bool:
    return s1_barrier_ready(experiment, c3_method_experiments(experiment))


def execute_c3_region_pilot(repo, experiment, method, candidate_id):
    """Run specified S1 cells through production scoring, without selection or test."""
    seal = json.loads((experiment / "experiment.json").read_text())
    if method not in seal.get("methods", C3_FAIR_GRID_METHODS):
        raise ValueError("pilot method is not active in this experiment")
    verify_hashes(experiment, seal["artifacts"])
    input_spec = yaml.safe_load((experiment / "study/inputs.yaml").read_text())
    verify_hashes(
        repo,
        {
            name: digest
            for name, digest in seal["input_hashes"].items()
            if name != input_spec["amplitudes"]
        },
    )
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    _, specification = c3_split_settings(contract)
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    base = yaml.safe_load((experiment / "study" / f"{method}.yaml").read_text())
    candidates = c3_search_candidates(base, grid["methods"][method])
    candidates = Rehearsal(experiment).candidates(candidates, contract, method)
    selected = [item for item in candidates if candidate_id in ("all", item["id"])]
    if not selected:
        raise ValueError("unknown pilot candidate ID")
    return [
        execute_c3_region_run(repo, experiment, item, specification["search_seed"], "S1")
        for item in selected
    ]


def run_c3_region_worker(repo: Path, experiment: Path, method: str, *, shard="0/1"):
    seal = json.loads((experiment / "experiment.json").read_text())
    if method not in seal.get("methods", C3_FAIR_GRID_METHODS):
        raise ValueError("worker method is not active in this experiment")
    verify_hashes(experiment, seal["artifacts"])
    input_spec = yaml.safe_load((experiment / "study/inputs.yaml").read_text())
    verify_hashes(
        repo,
        {
            name: digest
            for name, digest in seal["input_hashes"].items()
            if name != input_spec["amplitudes"]
        },
    )
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    _, specification = c3_split_settings(contract)
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
    state_path = workers / f"{worker_name}.json"
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

    def execute(candidate, seed, stage, *, prediction_only=False):
        options = {"prediction_only": True} if prediction_only else {}
        resources = contract.get("cuda_execution")
        if resources is None or resources.get("global_admission", False):
            return execute_c3_region_run(repo, experiment, candidate, seed, stage, **options)

        def waiting(details):
            state.update(phase="waiting_for_gpu", resource_wait=details)
            _write_status(state_path, state)

        minimum_memory = resources["minimum_free_memory_gib"][method]
        if isinstance(minimum_memory, dict):
            minimum_memory = minimum_memory[candidate["id"]]
        with cuda_run_slot(
            workers,
            candidate["config"]["device"],
            minimum_memory,
            waiting,
        ):
            state.update(phase=stage)
            state.pop("resource_wait", None)
            _write_status(state_path, state)
            return execute_c3_region_run(repo, experiment, candidate, seed, stage, **options)

    def children_changed(children):
        state.update(child_pids=[child["pid"] for child in children], child_processes=children)
        _write_status(state_path, state)

    def failed(error):
        state.update(
            status="retry_pending" if failure_kind(error) == "infrastructure" else "failed",
            failure_reason=f"{type(error).__name__}: {error}",
        )
        _write_status(state_path, state)

    def waiting_for_candidate(details):
        state.update(phase="waiting_for_gpu", resource_wait=details)
        state.pop("active_run", None)
        _write_status(state_path, state)
        print(f"{method}: waiting for GPU: {details}", flush=True)

    with (workers / f"{worker_name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_status(state_path, state)
        try:
            without_admission = s1_candidates_without_admission(repo, experiment, candidates, "c3")
            records = []
            search_seed = specification["search_seed"]
            parallel = settings.get("parallel_candidates", 1)
            if parallel != 1:
                search_execute = partial(
                    _execute_c3_search_candidate,
                    repo=repo,
                    experiment=experiment,
                    seed=search_seed,
                )
                outcomes = candidate_outcomes(
                    search_execute,
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
                )
            else:

                def sequential_outcomes():
                    for candidate in memory_fitting_candidates(
                        repo,
                        candidates,
                        contract.get("cuda_execution"),
                        without_admission=without_admission,
                        waiting=waiting_for_candidate,
                    ):
                        state.update(phase="S1", active_run=candidate["id"])
                        state.pop("resource_wait", None)
                        _write_status(state_path, state)
                        yield _execute_c3_search_outcome(
                            execute, candidate, search_seed, experiment, method
                        )

                outcomes = sequential_outcomes()
            for record in outcomes:
                records.append(record)
                state.update(phase="S1", active_run=record["id"], completed_s1=len(records))
                state.pop("resource_wait", None)
                _write_status(state_path, state)
            if parse_shard(shard)[1] > 1:
                state.update(shard_completed=True, phase="waiting_for_shards")
                _write_status(state_path, state)
                records = collect_shard_records(experiment, method, all_candidates, shard)
                if records is None:
                    state.update(status="completed", phase="shard_completed", s1_completed=True)
                    _write_status(state_path, state)
                    return
            winner = select_c3_candidate(
                all_candidates,
                records,
                expected=len(all_candidates) if seal.get("rehearsal") is True else 25,
            )
            write_json(
                workers / f"{method}_selection.json",
                {
                    "rule": "minimum_validation_score_then_configuration_id",
                    "winner": winner,
                    "s1_records": [
                        {
                            "id": row["id"],
                            "status": row["status"],
                            "validation_score": row.get("validation_score"),
                        }
                        for row in records
                    ],
                    "test_used_for_selection": False,
                },
            )
            state.update(
                s1_completed=True,
                phase="predicting_seeds" if seal.get("rehearsal") is True else "waiting_for_all_S1",
                selected_configuration=winner["id"],
                active_run=None,
            )
            _write_status(state_path, state)
            seed_jobs = [winner | {"seed": seed} for seed in specification["evaluation_seeds"]]
            resources = contract.get("cuda_execution")
            if resources and not resources.get("global_admission", False):
                for seed in specification["evaluation_seeds"]:
                    execute(winner, seed, "evaluation", prediction_only=True)
            else:
                for _ in candidate_outcomes(
                    partial(_predict_seed, repo=repo, experiment=experiment),
                    seed_jobs,
                    2 if base["device"] != "cpu" else 1,
                    children_changed=children_changed,
                ):
                    pass
            if seal.get("rehearsal") is True:
                state.update(phase="waiting_for_all_S1", active_run=None)
                _write_status(state_path, state)
            while not c3_s1_barrier_ready(experiment):
                time.sleep(10)
            evaluations = []
            for seed in specification["evaluation_seeds"]:
                state.update(phase="evaluation", active_run=f"seed_{seed}")
                _write_status(state_path, state)
                evaluations.append(execute(winner, seed, "evaluation"))
                if evaluations[-1]["status"] not in ("completed", "diverged"):
                    raise RuntimeError("post-barrier scoring did not complete")
                state["completed_evaluation"] = sum(
                    row["status"] == "completed" for row in evaluations
                )
                _write_status(state_path, state)
            summarize_c3_region_seeds(
                evaluations, expected_seeds=specification["evaluation_seeds"]
            ).to_csv(workers / f"{method}_summary.csv", index=False)
            state.update(status="completed", phase="completed", active_run=None, end_time=utc_now())
            _write_status(state_path, state)
        except BaseException as error:
            state.update(
                status="retry_pending" if failure_kind(error) == "infrastructure" else "failed",
                failure_reason=f"{type(error).__name__}: {error}",
                end_time=utc_now(),
            )
            _write_status(state_path, state)
            raise


def _predict_seed(candidate, *, repo, experiment):
    return execute_c3_region_run(
        repo, experiment, candidate, candidate["seed"], "evaluation", prediction_only=True
    )


def _execute_c3_search_candidate(candidate, *, repo, experiment, seed):
    execute = partial(execute_c3_region_run, repo, experiment)
    return _execute_c3_search_outcome(
        execute, candidate, seed, experiment, candidate["config"]["method_id"]
    )


def _execute_c3_search_outcome(execute, candidate, search_seed, experiment, method):
    return execute(candidate, search_seed, "S1")


def _wait_c3_replacement_workers(experiment, processes, sources, *, restart=None):
    """Bridge the original barrier and wait for both new and continuing methods."""
    pending = {}
    while True:
        publish_c3_replacement_readiness(experiment)
        outcomes = {method: process.poll() for method, process in processes.items()}
        if restart is not None:
            restart_worker_processes(processes, outcomes, restart, pending)
        failed = record_worker_exits(experiment / "workers", processes, outcomes)
        if failed or any(code is not None and code != 0 for code in outcomes.values()):
            return outcomes, False
        completed = True
        for method, source in sources.items():
            if method in pending:
                completed = False
                continue
            path = source / "workers" / f"{method}.json"
            if not path.exists():
                completed = False
                continue
            state = json.loads(path.read_text())
            failure = worker_failure(state)
            if failure is not None:
                raise RuntimeError(f"{method} failed: {failure}")
            completed &= state["status"] == "completed"
        if completed and all(code is not None for code in outcomes.values()):
            return outcomes, True
        time.sleep(10)


def _write_c3_experiment_aggregates(experiment, sources):
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    kind, _ = c3_split_settings(contract)
    summaries, seed_metrics = [], []
    for method, source in sources.items():
        summary = pd.read_csv(source / "workers" / f"{method}_summary.csv")
        summary.insert(0, "method_id", method)
        summaries.append(summary)
        for manifest in sorted(
            (source / "method_runs" / method / "evaluation").glob("*/manifest.json")
        ):
            record = json.loads(manifest.read_text())
            for domain, metrics in record["metrics"].items():
                seed_metrics.append(
                    {
                        "method_id": method,
                        f"{kind}_seed": record[f"{kind}_seed"],
                        "configuration_id": record["id"],
                        "domain": domain,
                        **metrics,
                    }
                )
    pd.concat(summaries, ignore_index=True).to_csv(experiment / "summary.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(experiment / "seed_metrics.csv", index=False)


def supervise_c3_region_experiment(
    repo: Path, experiment: Path, script: Path, *, only_dead_workers=False
):
    """Run declared methods and collect their results with any continuing workers."""
    seal = json.loads((experiment / "experiment.json").read_text())
    methods = seal.get("methods", C3_FAIR_GRID_METHODS)
    sources = c3_method_experiments(experiment)
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    with (workers / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = {}
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(experiment / "source/src"),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        contract_path = experiment / "study/config.yaml"
        runtime_contract = (
            yaml.safe_load(contract_path.read_text()) if contract_path.exists() else {}
        )
        jobs = {}
        for method in methods:
            count = (
                runtime_contract.get("method_execution", {}).get(method, {}).get("worker_shards", 1)
            )
            for index in range(count):
                shard = f"{index}/{count}"
                jobs[shard_worker_name(method, shard)] = method, shard

        def start_worker(worker_name):
            method, shard = jobs[worker_name]
            with (workers / f"{worker_name}.log").open("a") as stream:
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
                    stdout=stream,
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

        _write_status(
            workers / "supervisor.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "start_time": utc_now(),
                "worker_pids": {method: process.pid for method, process in processes.items()},
                "process_start_time": process_start_time(os.getpid()),
                "source_root": source_root(),
                "pythonpath": os.environ.get("PYTHONPATH"),
                "script": str(script),
                "only_dead_workers": only_dead_workers,
            },
        )
        if "continuing_experiment" in seal:
            try:
                outcomes, completed = _wait_c3_replacement_workers(
                    experiment, processes, sources, restart=restart
                )
            except BaseException as error:
                _write_status(
                    workers / "supervisor.json",
                    {
                        "status": "failed",
                        "pid": os.getpid(),
                        "failure_reason": f"{type(error).__name__}: {error}",
                        "end_time": utc_now(),
                    },
                )
                raise
        else:
            outcomes, completed = wait_for_workers(
                processes,
                directory=workers,
                **({"restart": restart} if contract_path.exists() else {}),
            )
        if completed:
            _write_c3_experiment_aggregates(experiment, sources)
        _write_status(
            workers / "supervisor.json",
            {
                "status": "completed" if completed else "failed",
                "pid": os.getpid(),
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
            raise RuntimeError(f"region experiment worker failures: {outcomes}")
