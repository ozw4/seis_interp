"""Exercise sealed faults through both production runners with synthetic CPU inputs."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from seis_interp.data import c3_region_inputs as c3_data
from seis_interp.data import forge_grid_inputs as forge_data
from seis_interp.data import forge_trace_experiment as single_data
from seis_interp.evaluation import study067_rehearsal_checks as rehearsal_checks
from seis_interp.pipelines import run_c3_region_experiment as c3
from seis_interp.pipelines import run_forge_grid_experiment as forge
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training import run_recovery
from seis_interp.training.study067_rehearsal_generation import rehearsal_faults
from seis_interp.training.worker_processes import report_admission
from seis_interp.training.worker_restarts import _mark_worker_restart
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study
from tests.fixtures.forge_grid_experiment import tiny_forge_grid_study


def write_plan(study, contract, plan):
    contract["study_id"] += "_rehearsal"
    contract["trace_split"]["evaluation_seeds"] = [990671, 990672, 990673]
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    (study / "rehearsal.yaml").write_text(yaml.safe_dump(plan))


@pytest.fixture(params=["c3", "forge"])
def tiny_rehearsal(tmp_path, monkeypatch, request):
    for module in (c3_data, forge_data, single_data):
        monkeypatch.setattr(
            module,
            "execution_identity",
            lambda _: {"git_commit": "fixture", "dirty_worktree": False},
        )
    monkeypatch.setattr(run_recovery.time, "sleep", lambda _: None)
    if request.param == "c3":
        study, contract, configs = tiny_c3_trace_study(tmp_path)
        config = configs["pocs_grid"]
        prepare, execute = c3_data.prepare_c3_region_experiment, c3.execute_c3_region_run
    else:
        study, contract, config, _ = tiny_forge_grid_study(tmp_path)
        prepare, execute = forge_data.prepare_forge_grid_experiment, forge.execute_forge_grid_run
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    return tmp_path, study, contract, candidate, prepare, execute


@pytest.mark.parametrize("tiny_rehearsal", ["forge"], indirect=True)
@pytest.mark.parametrize("waited", [False, True])
def test_forge_sticky_cuda_restart_preserves_run_evidence(tiny_rehearsal, monkeypatch, waited):
    repo, study, contract, _, prepare, _ = tiny_rehearsal
    method, run = "ccnet5d_grid", "a05_b05"
    worker = f"{method}_shard_1_of_2"
    contract["methods"] = [method]
    contract["method_execution"] = {method: {"parallel_candidates": 1, "worker_shards": 2}}
    grid = {
        "methods": {
            method: {
                "axis_a": [
                    {"id": str(i), "set": {"training.learning_rate": i * 1e-4}} for i in range(1, 6)
                ],
                "axis_b": [
                    {"id": str(i), "set": {"training.weight_decay": i * 1e-5}} for i in range(1, 6)
                ],
            }
        }
    }
    (study / "grid.yaml").write_text(yaml.safe_dump(grid))
    fault = next(fault for fault in rehearsal_faults("forge") if fault["action"] == "sticky_cuda")
    write_plan(study, contract, {"faults": [fault], "candidates": {method: ["a05_b02", run]}})
    experiment = prepare(repo, study)

    def candidates(repo, candidates, resources, **kwargs):
        if waited:
            kwargs["waiting"]({"reason": "no_fitting_candidate"})
        yield from candidates

    def faulted_run(candidate, **kwargs):
        report_admission(
            experiment, method, candidate["id"], "S1", {"device": "cuda:1", "wait_seconds": 0}
        )
        raise run_recovery.RestartRunProcess("fresh process required: device-side assert triggered")

    monkeypatch.setattr(forge, "memory_fitting_candidates", candidates)
    monkeypatch.setattr(forge, "_search_run", faulted_run)
    with pytest.raises(run_recovery.RestartRunProcess):
        forge.run_forge_grid_worker(repo, experiment, method, shard="1/2")
    path = experiment / f"workers/{worker}.json"
    state = json.loads(path.read_text())
    assert state["status"] == "retry_pending"
    assert state["phase"] == "S1"
    assert state["active_run"] == run
    assert "resource_wait" not in state
    assert _mark_worker_restart(experiment / "workers", worker, path, state)
    restart_path = experiment / "workers/rehearsal_restarts.jsonl"
    row = json.loads(restart_path.read_text().splitlines()[0])
    assert row["run"] == ["S1", run, 0, 0]
    assert rehearsal_checks._check_restart_evidence(experiment, {"faults": [fault]}) == restart_path


@pytest.mark.parametrize(
    "action,point,status,attempt",
    [
        ("oom", "pre_method", "completed", 1),
        ("infra", "pre_method", "completed", 1),
        ("resource_wait", "pre_admission", "completed", 0),
        ("nonfinite", "pre_method", "diverged", 0),
        ("nan_prediction", "after_prediction", "diverged", 0),
    ],
)
def test_fault_actions_reach_retry_wait_or_divergence_state(
    tiny_rehearsal, action, point, status, attempt
):
    repo, study, contract, candidate, prepare, execute = tiny_rehearsal
    write_plan(
        study,
        contract,
        {
            "faults": [
                {"method": "pocs_grid", "run": candidate["id"], "point": point, "action": action}
            ],
            "candidates": {"pocs_grid": [candidate["id"]]},
        },
    )
    experiment = prepare(repo, study)
    record = execute(repo, experiment, candidate, contract["trace_split"]["search_seed"], "S1")
    output = experiment / "method_runs/pocs_grid/S1" / candidate["id"]
    assert record["status"] == status and record["attempt"] == attempt
    assert (output / "attempts").exists() == bool(attempt)
    if attempt:
        assert json.loads((output / "attempts/1/manifest.json").read_text())["pid"] == record["pid"]
    assert len(list((experiment / "rehearsal_faults").glob("*.json"))) == 1


@pytest.mark.parametrize("method", ["regsi_real", "ccnet5d_grid", "nersi_profile"])
def test_report_fault_exits_with_running_manifest_then_resumes_latest_same_step_checkpoint(
    tmp_path, monkeypatch, method
):
    monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "fixture"})
    study, contract, configs = tiny_c3_trace_study(tmp_path)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(configs[method], grid["methods"][method])[0]
    write_plan(
        study,
        contract,
        {
            "resume_interval_updates": 2,
            "faults": [
                {
                    "method": method,
                    "run": candidate["id"],
                    "point": "report",
                    "step": 2,
                    "action": "exit",
                }
            ],
        },
    )
    experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    code = """
import json, sys
from pathlib import Path
from seis_interp.training import run_recovery
from seis_interp.pipelines.run_c3_region_experiment import execute_c3_region_run
run_recovery.time.sleep = lambda _: None
execute_c3_region_run(
    Path(sys.argv[1]), Path(sys.argv[2]),
    json.loads(Path(sys.argv[3]).read_text()), 21, 'S1')
"""
    command = [sys.executable, "-c", code, str(tmp_path), str(experiment), str(candidate_path)]
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    assert first.returncode == 137, first.stderr
    output = experiment / "method_runs" / method / "S1" / candidate["id"]
    assert json.loads((output / "manifest.json").read_text())["status"] == "running"
    import torch

    assert torch.load(output / "resume.pt", weights_only=False)["step"] == 2
    second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stderr
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "completed" and record["attempt"] == 1
    assert "resumed step=2 phase=training" in second.stdout
    import pandas as pd

    assert pd.read_csv(output / "history.csv").step.tolist() == [1, 2, 3, 4]
    assert (output / "attempts/1/resume.pt").exists()


def test_real_prepare_is_unchanged_except_disabled_rehearsal_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "fixture"})
    study, _, _ = tiny_c3_trace_study(tmp_path)
    experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
    seal = json.loads((experiment / "experiment.json").read_text())
    assert seal["rehearsal"] is False
    assert seal["s1_runs_per_method"] == 25 and seal["total_runs"] == 140
    assert "rehearsal_plan_sha256" not in seal
    assert not (experiment / "study/rehearsal.yaml").exists()


def test_subset_capacity_import_coverage_and_worker_selection(tiny_rehearsal, monkeypatch):
    from seis_interp.data.experiment_import_coverage import experiment_import_coverage
    from seis_interp.training.rehearsal import Rehearsal

    repo, study, contract, candidate, prepare, _ = tiny_rehearsal
    subsets = {"pocs_grid": ["a01_b01", "a02_b01"]}
    write_plan(study, contract, {"faults": [], "candidates": subsets})
    module = c3_data if prepare is c3_data.prepare_c3_region_experiment else forge_data
    admitted = []
    monkeypatch.setattr(
        module,
        "prepare_admission_capacity",
        lambda resources, candidates: admitted.extend(candidates),
    )
    experiment = prepare(repo, study)
    assert len([item for item in admitted if item["config"]["method_id"] == "pocs_grid"]) == 2
    coverage = experiment_import_coverage(repo, experiment, "c3" if module is c3_data else "forge")
    assert coverage["methods"]["pocs_grid"]["s1_total"] == 2
    seal = json.loads((experiment / "experiment.json").read_text())
    counts = seal["s1_runs_per_method"]
    assert counts == 2 if module is forge_data else counts["pocs_grid"] == 2
    options = Rehearsal(experiment).candidates(
        [candidate | {"id": name} for name in ["a01_b01", "a02_b01", "a03_b01"]],
        contract,
        "pocs_grid",
    )
    assert len(options) == 2


def test_post_scoring_exit_reuses_prediction_and_preserves_scoring_bytes(
    tiny_rehearsal, monkeypatch
):
    repo, study, contract, candidate, prepare, execute = tiny_rehearsal
    write_plan(
        study,
        contract,
        {
            "faults": [
                {
                    "method": "pocs_grid",
                    "run": "seed_990672",
                    "point": "after_test_scored",
                    "action": "exit",
                }
            ]
        },
    )
    experiment = prepare(repo, study)
    pipeline = c3 if prepare is c3_data.prepare_c3_region_experiment else forge
    barrier = "c3_s1_barrier_ready" if pipeline is c3 else "forge_s1_barrier_ready"
    monkeypatch.setattr(pipeline, barrier, lambda _: True)
    predicted = execute(repo, experiment, candidate, 990672, "evaluation", prediction_only=True)
    prediction_execution = predicted["prediction_execution"].copy()

    candidate_path = repo / "scoring_candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    dataset = "c3_region" if pipeline is c3 else "forge_grid"
    code = f"""
import json, sys
from pathlib import Path
from seis_interp.pipelines import run_{dataset}_experiment as pipeline
pipeline.{barrier} = lambda _: True
pipeline.execute_{dataset}_run(
    Path(sys.argv[1]), Path(sys.argv[2]),
    json.loads(Path(sys.argv[3]).read_text()), 990672, 'evaluation')
"""
    lost = subprocess.run(
        [sys.executable, "-c", code, str(repo), str(experiment), str(candidate_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert lost.returncode == 137, lost.stderr
    output = experiment / "method_runs/pocs_grid/evaluation/seed_990672"
    previous = (output / "test_per_trace.parquet").read_bytes()
    predictor = "run_c3_region_method" if pipeline is c3 else "run_forge_c3_method"
    monkeypatch.setattr(
        pipeline, predictor, lambda *args, **kwargs: pytest.fail("prediction repeated")
    )
    record = execute(repo, experiment, candidate, 990672, "evaluation")
    assert record["status"] == "completed" and record["attempt"] == 1
    assert record["prediction_execution"] == prediction_execution
    assert (output / "attempts/1/test_per_trace.parquet").read_bytes() == previous
    assert (output / "test_per_trace.parquet").read_bytes() == previous
    assert np.isfinite(np.load(output / "prediction.npy")).all()


def test_sleep_fault_leaves_watchdog_exit75_then_resumes_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "fixture"})
    study, contract, configs = tiny_c3_trace_study(tmp_path)
    method = "nersi_profile"
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(configs[method], grid["methods"][method])[0]
    write_plan(
        study,
        contract,
        {
            "resume_interval_updates": 2,
            "watchdog": {"floor_seconds": 0.05, "expected_report_seconds": 0.01},
            "faults": [
                {
                    "point": "report",
                    "method": method,
                    "run": candidate["id"],
                    "step": 2,
                    "action": "sleep",
                    "seconds": 3,
                }
            ],
        },
    )
    experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    code = """
import json, sys
from pathlib import Path
from seis_interp.training import run_recovery
from seis_interp.pipelines.run_c3_region_experiment import execute_c3_region_run
run_recovery.time.sleep = lambda _: None if _ < 3 else __import__('threading').Event().wait(_)
execute_c3_region_run(
    Path(sys.argv[1]), Path(sys.argv[2]),
    json.loads(Path(sys.argv[3]).read_text()), 21, 'S1')
"""
    command = [sys.executable, "-c", code, str(tmp_path), str(experiment), str(candidate_path)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode == 75, first.stderr
    output = experiment / "method_runs/nersi_profile/S1/a01_b01"
    dog = json.loads((output / "watchdog.json").read_text())
    assert dog["exit_code"] == 75
    # Use production-scale timing after proving the deliberately short test watchdog.
    # The consumed fault never sleeps again. Same CPU fixture finishes before the 1s polling tick.
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr
    assert (output / "attempts/1/watchdog.json").exists()
    assert json.loads((output / "manifest.json").read_text())["attempt"] == 1
    assert "resumed step=2" in second.stdout


def test_ccnet_report_oom_cleans_prefetch_and_retries_in_the_same_cpu_process(
    tmp_path, monkeypatch
):
    import threading

    monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "fixture"})
    monkeypatch.setattr(run_recovery.time, "sleep", lambda _: None)
    study, contract, configs = tiny_c3_trace_study(tmp_path)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(configs["ccnet5d_grid"], grid["methods"]["ccnet5d_grid"])[0]
    candidate["config"]["training"]["report_interval"] = 2
    write_plan(
        study,
        contract,
        {
            "resume_interval_updates": 2,
            "faults": [
                {
                    "method": "ccnet5d_grid",
                    "run": candidate["id"],
                    "point": "report",
                    "step": 2,
                    "action": "oom",
                }
            ],
        },
    )
    experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
    before = {thread.ident for thread in threading.enumerate()}
    record = c3.execute_c3_region_run(tmp_path, experiment, candidate, 21, "S1")
    output = experiment / "method_runs/ccnet5d_grid/S1/a01_b01"
    previous = json.loads((output / "attempts/1/manifest.json").read_text())
    assert record["status"] == "completed" and record["attempt"] == 1
    assert previous["pid"] == record["pid"] == os.getpid()
    assert {thread.ident for thread in threading.enumerate()} == before
    assert "resumed step=2" in (output / "execution.log").read_text()
    import pandas as pd

    history = pd.read_csv(output / "history.csv")
    assert history.step.tolist() == [1, 2, 3, 4]
    assert history.loss.iloc[-1] == pytest.approx(record["diagnostics"]["final_training_loss"])


def test_rehearsal_worker_distinguishes_seed_computation_from_barrier_wait(
    tiny_rehearsal, monkeypatch
):
    repo, study, contract, _, prepare, _ = tiny_rehearsal
    write_plan(study, contract, {"faults": [], "candidates": {"pocs_grid": ["a01_b01", "a02_b01"]}})
    experiment = prepare(repo, study)
    pipeline = c3 if prepare is c3_data.prepare_c3_region_experiment else forge
    prefix = "c3" if pipeline is c3 else "forge"
    monkeypatch.setattr(pipeline, f"{prefix}_s1_barrier_ready", lambda _: True)
    worker = c3.run_c3_region_worker if pipeline is c3 else forge.run_forge_grid_worker
    worker(repo, experiment, "pocs_grid")
    events = [
        json.loads(line)
        for line in (experiment / "workers/rehearsal_events.jsonl").read_text().splitlines()
    ]
    phases = [event.get("phase") for event in events]
    assert "predicting_seeds" in phases
    assert phases.index("predicting_seeds") < phases.index("waiting_for_all_S1")
    assert events[-1]["status"] == "completed"


@pytest.mark.parametrize("parallel", [1, 2])
def test_s1_memory_fit_wait_is_published_by_each_worker(tiny_rehearsal, monkeypatch, parallel):
    repo, study, contract, candidate, prepare, execute = tiny_rehearsal
    contract.setdefault("method_execution", {})["pocs_grid"] = {"parallel_candidates": parallel}
    write_plan(study, contract, {"candidates": {"pocs_grid": [candidate["id"]]}, "faults": []})
    experiment = prepare(repo, study)
    pipeline = c3 if execute is c3.execute_c3_region_run else forge
    worker = c3.run_c3_region_worker if pipeline is c3 else forge.run_forge_grid_worker
    seen = []

    def candidates(*args, waiting, **kwargs):
        waiting({"reason": "no_fitting_candidate", "remaining": [candidate["id"]]})
        state = json.loads((experiment / "workers/pocs_grid.json").read_text())
        assert state["phase"] == "waiting_for_gpu"
        assert state["resource_wait"]["remaining"] == [candidate["id"]]
        seen.append(True)
        raise RuntimeError("stop after checking waiting state")
        yield

    monkeypatch.setattr(pipeline, "memory_fitting_candidates", candidates)
    with pytest.raises(RuntimeError, match="stop after checking waiting state"):
        worker(repo, experiment, "pocs_grid")
    assert seen == [True]
