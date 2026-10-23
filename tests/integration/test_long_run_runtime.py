"""Recovery and the scoring boundary in both production experiment runners."""

import json
from copy import deepcopy

import pytest
import torch
import yaml

from seis_interp.data import c3_region_inputs as c3_data
from seis_interp.data import forge_grid_inputs as forge_data
from seis_interp.data import forge_trace_experiment as single_data
from seis_interp.pipelines import run_c3_region_experiment as c3
from seis_interp.pipelines import run_forge_grid_experiment as forge
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training import run_recovery
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study
from tests.fixtures.forge_grid_experiment import tiny_forge_grid_study


@pytest.fixture(params=["c3", "forge"])
def prepared(tmp_path, monkeypatch, request):
    monkeypatch.setattr(run_recovery.time, "sleep", lambda _: None)
    for module in (single_data, c3_data, forge_data):
        monkeypatch.setattr(module, "execution_identity", lambda _: {"git_commit": "testcommit"})
    if request.param == "c3":
        study, contract, configs = tiny_c3_trace_study(tmp_path)
        experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
        pipeline, execute = c3, c3.execute_c3_region_run
        train, reader, barrier = (
            "run_c3_region_method",
            "read_c3_region_targets",
            "c3_s1_barrier_ready",
        )
        config = configs["pocs_grid"]
    else:
        study, contract, config, _ = tiny_forge_grid_study(tmp_path)
        experiment = forge_data.prepare_forge_grid_experiment(tmp_path, study)
        pipeline, execute = forge, forge.execute_forge_grid_run
        train, reader, barrier = (
            "run_forge_c3_method",
            "read_forge_grid_targets",
            "forge_s1_barrier_ready",
        )
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    yield tmp_path, experiment, contract, pipeline, execute, candidate, train, reader, barrier


def test_two_infrastructure_retries_preserve_failed_attempts(prepared, monkeypatch):
    repo, experiment, contract, pipeline, execute, candidate, train, *_ = prepared
    original = getattr(pipeline, train)
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise torch.OutOfMemoryError("out of memory")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, train, flaky)
    record = execute(repo, experiment, candidate, contract["trace_split"]["search_seed"], "S1")
    output = experiment / "method_runs/pocs_grid/S1/a01_b01"
    assert record["status"] == "completed" and record["attempt"] == 2
    assert len(calls) == 3
    for number in (1, 2):
        failed = json.loads((output / f"attempts/{number}/manifest.json").read_text())
        assert failed["failure_kind"] == "infrastructure"


def test_retry_limit_is_two(prepared, monkeypatch):
    repo, experiment, contract, pipeline, execute, candidate, train, *_ = prepared
    calls = []

    def oom(*args, **kwargs):
        calls.append(1)
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(pipeline, train, oom)
    with pytest.raises(RuntimeError, match="retry limit exhausted"):
        execute(repo, experiment, candidate, contract["trace_split"]["search_seed"], "S1")
    assert len(calls) == 3
    output = experiment / "method_runs/pocs_grid/S1/a01_b01"
    assert {path.name for path in (output / "attempts").iterdir()} == {"1", "2", "3"}
    assert json.loads((output / "manifest.json").read_text())["retry_exhausted"]
    with pytest.raises(RuntimeError, match="retry limit exhausted"):
        execute(repo, experiment, candidate, contract["trace_split"]["search_seed"], "S1")
    assert len(calls) == 3


def test_nonfinite_loss_is_diverged_and_never_retried(prepared, monkeypatch):
    repo, experiment, contract, pipeline, execute, candidate, train, *_ = prepared
    calls = []

    def nonfinite(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("non-finite training loss")

    monkeypatch.setattr(pipeline, train, nonfinite)
    args = repo, experiment, candidate, contract["trace_split"]["search_seed"], "S1"
    assert execute(*args)["status"] == "diverged"
    assert execute(*args)["status"] == "diverged"
    assert len(calls) == 1
    assert not (experiment / "method_runs/pocs_grid/S1/a01_b01/attempts").exists()


def test_seed_prediction_can_finish_before_barrier_without_opening_any_targets(
    prepared, monkeypatch
):
    repo, experiment, _, pipeline, execute, candidate, train, reader, barrier = prepared
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    (workers / "pocs_grid_selection.json").write_text(json.dumps({"winner": candidate}))
    ready = False
    monkeypatch.setattr(pipeline, barrier, lambda _: ready)
    original_train, original_read = getattr(pipeline, train), getattr(pipeline, reader)
    trained, reads = [], []

    def observed_only(*args, **kwargs):
        assert not ready
        trained.append(1)
        return original_train(*args, **kwargs)

    def protected(*args):
        assert ready, "no validation or test target read is allowed before the barrier"
        reads.append(args[-1])
        return original_read(*args)

    monkeypatch.setattr(pipeline, train, observed_only)
    monkeypatch.setattr(pipeline, reader, protected)
    record = execute(repo, experiment, candidate, 7311, "evaluation", prediction_only=True)
    assert record["status"] == "awaiting_barrier" and record["test_scoring_calls"] == 0
    assert reads == [] and len(trained) == 1
    ready = True
    record = execute(repo, experiment, candidate, 7311, "evaluation")
    assert record["status"] == "completed" and record["test_scoring_calls"] == 1
    assert reads == ["validation", "test"] and len(trained) == 1
    execute(repo, experiment, candidate, 7311, "evaluation")
    assert reads == ["validation", "test"]


@pytest.mark.parametrize(
    "method,predictor",
    [
        ("ccnet5d_grid", "predict_ccnet5d_volume"),
        ("regsi_real", "predict_relational_trace_graph"),
        ("nersi_profile", "shadow_prediction_placeholder"),
    ],
)
def test_prediction_failure_resumes_without_another_optimizer_update(
    tmp_path, monkeypatch, method, predictor
):
    from seis_interp.training import forge_mvp_methods as methods

    monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    study, contract, configs = tiny_c3_trace_study(tmp_path)
    experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(configs[method], grid["methods"][method])[0]
    updates = []
    original_step = torch.optim.AdamW.step

    def count_update(self, *args, **kwargs):
        updates.append(1)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", count_update)
    failures = []
    if method == "nersi_profile":
        original_forward = methods.Nersi.forward

        def forward(self, *args, **kwargs):
            if not self.training and not failures:
                failures.append(1)
                raise torch.OutOfMemoryError("out of memory during prediction")
            return original_forward(self, *args, **kwargs)

        monkeypatch.setattr(methods.Nersi, "forward", forward)
    else:
        original_predict = getattr(methods, predictor)

        def fail_prediction(*args, **kwargs):
            if not failures:
                failures.append(1)
                raise torch.OutOfMemoryError("out of memory during prediction")
            return original_predict(*args, **kwargs)

        monkeypatch.setattr(methods, predictor, fail_prediction)
    monkeypatch.setattr(run_recovery.time, "sleep", lambda _: None)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        record = c3.execute_c3_region_run(
            tmp_path, experiment, deepcopy(candidate), contract["trace_split"]["search_seed"], "S1"
        )
    finally:
        torch.set_num_threads(previous)
    assert failures == [1] and record["status"] == "completed"
    assert len(updates) == configs[method]["training"]["max_steps"]
    output = experiment / "method_runs" / method / "S1/a01_b01"
    assert (
        torch.load(output / "attempts/1/resume.pt", weights_only=False)["phase"]
        == "before_prediction"
    )


def test_barrier_waits_for_pending_retry(prepared):
    import os

    from seis_interp.training.worker_processes import process_start_time

    _, experiment, _, pipeline, _, _, _, _, barrier = prepared
    seal = json.loads((experiment / "experiment.json").read_text())
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    for method in seal.get("methods", c3.C3_FAIR_GRID_METHODS):
        (workers / f"{method}.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "s1_completed": True,
                    "pid": os.getpid(),
                    "process_start_time": process_start_time(os.getpid()),
                }
            )
        )
    path = workers / "pocs_grid.json"
    saved = path.read_text()
    path.write_text(json.dumps({"status": "retry_pending", "s1_completed": True}))
    assert not getattr(pipeline, barrier)(experiment)
    path.write_text(saved)
    assert getattr(pipeline, barrier)(experiment)
    assert (workers / "s1_barrier.json").exists()
    path.write_text(json.dumps({"status": "retry_pending", "s1_completed": False}))
    assert getattr(pipeline, barrier)(experiment)


def test_disjoint_workers_collect_all_25_before_selecting(prepared):
    import os

    from seis_interp.training.worker_processes import process_start_time

    repo, experiment, _, pipeline, _, _, _, _, _ = prepared
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    seal = json.loads((experiment / "experiment.json").read_text())
    for method in seal.get("methods", c3.C3_FAIR_GRID_METHODS):
        if method != "pocs_grid":
            (workers / f"{method}.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "s1_completed": True,
                        "pid": os.getpid(),
                        "process_start_time": process_start_time(os.getpid()),
                    }
                )
            )
    run_worker = pipeline.run_c3_region_worker if pipeline == c3 else pipeline.run_forge_grid_worker
    run_worker(repo, experiment, "pocs_grid", shard="1/2")
    assert len(list((experiment / "method_runs/pocs_grid/S1").glob("*/manifest.json"))) == 12
    assert not (workers / "pocs_grid_selection.json").exists()
    run_worker(repo, experiment, "pocs_grid", shard="0/2")
    assert len(list((experiment / "method_runs/pocs_grid/S1").glob("*/manifest.json"))) == 25
    selection = json.loads((workers / "pocs_grid_selection.json").read_text())
    assert len(selection["s1_records"]) == 25
    assert json.loads((workers / "pocs_grid.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("failure", ["trace_value", "prediction", "metrics"])
def test_nonfinite_outputs_are_final_divergence_in_both_runners(prepared, monkeypatch, failure):
    import numpy as np

    repo, experiment, contract, pipeline, execute, candidate, train, *_ = prepared
    original = getattr(pipeline, train)
    calls = []

    def diverge(*args, **kwargs):
        calls.append(1)
        if failure == "trace_value":
            raise ValueError("selected prediction and target traces must be finite")
        result = original(*args, **kwargs)
        if failure == "prediction":
            result.prediction[:] = np.nan
        return result

    monkeypatch.setattr(pipeline, train, diverge)
    if failure == "metrics":
        scoring = "score_c3_region" if pipeline == c3 else "score_forge_grid"
        original_score = getattr(pipeline, scoring)

        def score(*args, **kwargs):
            metrics, traces = original_score(*args, **kwargs)
            population = metrics if pipeline == c3 else metrics["clean_target"]
            population["masked_trace_relative_mse"] = float("nan")
            return metrics, traces

        monkeypatch.setattr(pipeline, scoring, score)
    args = repo, experiment, candidate, contract["trace_split"]["search_seed"], "S1"
    assert execute(*args)["status"] == "diverged"
    assert execute(*args)["status"] == "diverged"
    assert len(calls) == 1
    assert not (experiment / "method_runs/pocs_grid/S1/a01_b01/attempts").exists()


def test_awaiting_barrier_scoring_sleeps_and_never_retrains(prepared, monkeypatch):
    repo, experiment, _, pipeline, execute, candidate, train, reader, barrier = prepared
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    (workers / "pocs_grid_selection.json").write_text(json.dumps({"winner": candidate}))
    assert (
        execute(repo, experiment, candidate, 7311, "evaluation", prediction_only=True)["status"]
        == "awaiting_barrier"
    )
    ready = iter([False, False, True, True])
    sleeps = []
    monkeypatch.setattr(pipeline, barrier, lambda _: next(ready))
    monkeypatch.setattr(run_recovery.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        pipeline, train, lambda *a, **kw: pytest.fail("retrained while waiting for barrier")
    )
    result = execute(repo, experiment, candidate, 7311, "evaluation")
    assert result["status"] == "completed" and result["test_scoring_calls"] == 1
    assert sleeps == [10]


def test_s1_and_seed_divergence_do_not_halt_worker_or_fake_three_seed_statistics(
    prepared, monkeypatch
):
    import os

    import pandas as pd

    from seis_interp.training.worker_processes import process_start_time

    repo, experiment, _, pipeline, _, _, train, *_ = prepared
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    methods = json.loads((experiment / "experiment.json").read_text()).get(
        "methods", c3.C3_FAIR_GRID_METHODS
    )
    for method in methods:
        if method != "pocs_grid":
            (workers / f"{method}.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "s1_completed": True,
                        "pid": os.getpid(),
                        "process_start_time": process_start_time(os.getpid()),
                    }
                )
            )
    calls = []
    original = getattr(pipeline, train)

    def diverge_some(*args, **kwargs):
        calls.append(1)
        if len(calls) in (1, 26):
            raise ValueError("selected prediction and target traces must be finite")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, train, diverge_some)
    worker = pipeline.run_c3_region_worker if pipeline == c3 else pipeline.run_forge_grid_worker
    worker(repo, experiment, "pocs_grid")
    state = json.loads((workers / "pocs_grid.json").read_text())
    assert state["status"] == "completed" and state["completed_evaluation"] == 2
    selection = json.loads((workers / "pocs_grid_selection.json").read_text())
    assert selection["winner"]["id"] != "a01_b01"
    assert len(selection["s1_records"]) == 25
    summary = pd.read_csv(workers / "pocs_grid_summary.csv")
    assert set(summary.seed_count) == {2} and summary["mean"].isna().all()
    assert summary.diverged_seeds.notna().all()
