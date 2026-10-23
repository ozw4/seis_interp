import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import yaml

from seis_interp.data import c3_region_inputs as c3_data
from seis_interp.data import forge_grid_inputs as forge_data
from seis_interp.data import forge_trace_experiment as single_data
from seis_interp.pipelines import run_c3_region_experiment as c3_pipeline
from seis_interp.pipelines import run_forge_grid_experiment as forge_pipeline
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training.worker_processes import process_start_time
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study
from tests.fixtures.forge_grid_experiment import tiny_forge_grid_study


@pytest.mark.parametrize("dataset", ["c3", "forge"])
@pytest.mark.parametrize("minimum", [90, 94])
def test_prepare_rejects_impossible_budget_before_writing_run(
    tmp_path, monkeypatch, dataset, minimum
):
    monkeypatch.setattr(single_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    if dataset == "c3":
        study, contract, configs = tiny_c3_trace_study(tmp_path)
        config = configs["pocs_grid"]
        prepare = c3_data.prepare_c3_region_experiment
    else:
        study, contract, config, _ = tiny_forge_grid_study(tmp_path)
        prepare = forge_data.prepare_forge_grid_experiment
    config["device"] = "cuda:0"
    (study / "pocs_grid.yaml").write_text(yaml.safe_dump(config))
    contract["cuda_execution"] = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"pocs_grid": minimum},
        "device_total_memory_gib": {"cuda:0": 93},
        "admission_margin_gib": 2,
    }
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    destination = tmp_path / "never_created"
    with pytest.raises(ValueError, match="GiB exceeds.*91.000"):
        prepare(tmp_path, study, destination=destination)
    assert not destination.exists()


@pytest.mark.parametrize("dataset", ["c3", "forge"])
@pytest.mark.parametrize("priority", [["typo"], ["a01_b01", "a01_b01"]])
def test_prepare_validates_candidate_priority_before_creating_experiment(
    tmp_path, monkeypatch, dataset, priority
):
    monkeypatch.setattr(single_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    if dataset == "c3":
        study, contract, _ = tiny_c3_trace_study(tmp_path)
        prepare = c3_data.prepare_c3_region_experiment
    else:
        study, contract, _, _ = tiny_forge_grid_study(tmp_path)
        prepare = forge_data.prepare_forge_grid_experiment
    contract["method_execution"] = {"pocs_grid": {"candidate_priority": priority}}
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    destination = tmp_path / "never_created"
    with pytest.raises(ValueError, match="candidate_priority"):
        prepare(tmp_path, study, destination=destination)
    assert not destination.exists()


def test_c3_interrupted_admission_can_resume_without_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    study, _, configs = tiny_c3_trace_study(tmp_path)
    experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(configs["pocs_grid"], grid["methods"]["pocs_grid"])[0]
    admission = c3_pipeline.candidate_admission

    @contextmanager
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("waiting")
        yield

    monkeypatch.setattr(c3_pipeline, "candidate_admission", interrupted)
    with pytest.raises(KeyboardInterrupt, match="waiting"):
        c3_pipeline.execute_c3_region_run(tmp_path, experiment, candidate, 21, "S1")
    assert not (experiment / "method_runs/pocs_grid/S1/a01_b01").exists()
    monkeypatch.setattr(c3_pipeline, "candidate_admission", admission)
    record = c3_pipeline.execute_c3_region_run(tmp_path, experiment, candidate, 21, "S1")
    assert record["status"] == "completed"


@pytest.mark.parametrize(
    "pipeline,supervise",
    [
        (forge_pipeline, forge_pipeline.supervise_forge_grid_experiment),
        (c3_pipeline, c3_pipeline.supervise_c3_region_experiment),
    ],
)
@pytest.mark.parametrize("prior_status", [None, "running", "completed", "failed"])
@pytest.mark.parametrize("exit_code", [2, -9])
def test_supervisors_set_allocator_and_record_failure_without_waiting_or_signals(
    tmp_path, monkeypatch, pipeline, supervise, prior_status, exit_code
):
    (tmp_path / "experiment.json").write_text(json.dumps({"methods": ["first", "failed"]}))
    environments = []
    failed_path = tmp_path / "workers/failed.json"
    failed_path.parent.mkdir()
    if prior_status is not None:
        failed_path.write_text(
            json.dumps({"status": prior_status, "failure_reason": "original failure"})
        )

    def popen(*args, **kwargs):
        environments.append(kwargs["env"])
        code = None if len(environments) == 1 else exit_code
        return SimpleNamespace(pid=len(environments), poll=lambda: code)

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="worker failures"):
        supervise(tmp_path, tmp_path, tmp_path / "script.py")
    assert all(env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True" for env in environments)
    state = json.loads((tmp_path / "workers/supervisor.json").read_text())
    assert state["status"] == "failed"
    assert state["exit_codes"] == {"first": None, "failed": exit_code}
    assert state["workers_still_running"] == {"first": 1}
    assert state["failure_policy"] == "stop_waiting_without_signalling_workers"
    failed = json.loads(failed_path.read_text())
    assert failed["status"] == "failed"
    if prior_status == "failed":
        assert failed["failure_reason"] == "original failure"
    else:
        assert f"code {exit_code}" in failed["failure_reason"]
        if prior_status is None:
            assert "missing worker state" in failed["failure_reason"]
    barrier = (
        c3_pipeline.c3_s1_barrier_ready
        if pipeline == c3_pipeline
        else forge_pipeline.forge_s1_barrier_ready
    )
    with pytest.raises(RuntimeError, match="failed failed"):
        barrier(tmp_path)


@pytest.mark.parametrize("dataset", ["c3", "forge"])
@pytest.mark.parametrize("stage", ["S1", "evaluation"])
def test_runner_clears_admission_wait_before_executing_candidate(
    tmp_path, monkeypatch, dataset, stage
):
    monkeypatch.setattr(single_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    if dataset == "c3":
        monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
        study, contract, configs = tiny_c3_trace_study(tmp_path)
        config = configs["pocs_grid"]
        experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
        pipeline, execute = c3_pipeline, c3_pipeline.execute_c3_region_run
    else:
        study, contract, config, _ = tiny_forge_grid_study(tmp_path)
        monkeypatch.setattr(
            forge_data, "execution_identity", lambda _: {"git_commit": "testcommit"}
        )
        experiment = forge_data.prepare_forge_grid_experiment(tmp_path, study)
        pipeline, execute = forge_pipeline, forge_pipeline.execute_forge_grid_run
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    methods = json.loads((experiment / "experiment.json").read_text()).get(
        "methods", list(grid["methods"])
    )
    (experiment / "workers").mkdir()
    for method in methods:
        (experiment / f"workers/{method}.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "phase": stage,
                    "s1_completed": True,
                    "pid": os.getpid(),
                    "process_start_time": process_start_time(os.getpid()),
                }
            )
        )
    path = experiment / "workers/pocs_grid.json"

    @contextmanager
    def admission(repo, config, resources, candidate_id, waiting, *, admitted, reservation=None):
        waiting({"reason": "insufficient_free_memory"})
        assert json.loads(path.read_text())["phase"] == "waiting_for_gpu"
        admitted({"device": "cpu", "wait_seconds": 2.5})
        state = json.loads(path.read_text())
        assert state["phase"] == stage
        assert "resource_wait" not in state
        assert state["last_admission"]["wait_seconds"] == 2.5
        yield "cpu"

    monkeypatch.setattr(pipeline, "candidate_admission", admission)
    seed = contract["trace_split"]["search_seed"] if stage == "S1" else 7311
    assert execute(tmp_path, experiment, candidate, seed, stage)["status"] == "completed"


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_production_runner_stores_only_metrics_for_ema_shadows(tmp_path, monkeypatch, dataset):
    monkeypatch.setattr(single_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    if dataset == "c3":
        monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
        study, contract, configs = tiny_c3_trace_study(tmp_path)
        config = configs["pocs_grid"]
        experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
        pipeline, execute = c3_pipeline, c3_pipeline.execute_c3_region_run
        name = "run_c3_region_method"
    else:
        monkeypatch.setattr(
            forge_data, "execution_identity", lambda _: {"git_commit": "testcommit"}
        )
        study, contract, config, _ = tiny_forge_grid_study(tmp_path)
        experiment = forge_data.prepare_forge_grid_experiment(tmp_path, study)
        pipeline, execute = forge_pipeline, forge_pipeline.execute_forge_grid_run
        name = "run_forge_c3_method"
    original = getattr(pipeline, name)

    def with_shadows(*args, **kwargs):
        result = original(*args, **kwargs)
        result.shadow_predictions = {"0.999": result.prediction, "0.9995": result.prediction * 0.5}
        return result

    monkeypatch.setattr(pipeline, name, with_shadows)
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    candidate = c3_search_candidates(config, grid["methods"]["pocs_grid"])[0]
    record = execute(tmp_path, experiment, candidate, contract["trace_split"]["search_seed"], "S1")
    assert record["test_scoring_calls"] == 0
    assert set(record["ema_shadow_metrics"]["validation"]) == {"0.999", "0.9995"}
    assert record["ema_shadow_metrics"]["validation"]["0.999"] == record["metrics"]["validation"]
    assert set(name for name in record["artifacts"] if name.endswith(".npy")) == {
        "prediction.npy",
        "cell_ids.npy",
    }


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_pilot_entrypoint_uses_production_s1_and_never_enters_seed_evaluation(
    tmp_path,
    monkeypatch,
    dataset,
):
    monkeypatch.setattr(single_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    if dataset == "c3":
        monkeypatch.setattr(c3_data, "execution_identity", lambda _: {"git_commit": "testcommit"})
        study, contract, _ = tiny_c3_trace_study(tmp_path)
        experiment = c3_data.prepare_c3_region_experiment(tmp_path, study)
        pipeline, pilot, name = (
            c3_pipeline,
            c3_pipeline.execute_c3_region_pilot,
            "execute_c3_region_run",
        )
    else:
        monkeypatch.setattr(
            forge_data, "execution_identity", lambda _: {"git_commit": "testcommit"}
        )
        study, contract, _, _ = tiny_forge_grid_study(tmp_path)
        experiment = forge_data.prepare_forge_grid_experiment(tmp_path, study)
        pipeline, pilot, name = (
            forge_pipeline,
            forge_pipeline.execute_forge_grid_pilot,
            "execute_forge_grid_run",
        )
    calls = []

    def recorded(repo, experiment, candidate, seed, stage):
        calls.append((candidate["id"], seed, stage))
        return {"status": "completed"}

    monkeypatch.setattr(pipeline, name, recorded)
    pilot(tmp_path, experiment, "pocs_grid", "a02_b04")
    assert calls == [("a02_b04", contract["trace_split"]["search_seed"], "S1")]
    calls.clear()
    pilot(tmp_path, experiment, "pocs_grid", "all")
    assert len(calls) == 25 and {stage for _, _, stage in calls} == {"S1"}
    with pytest.raises(ValueError, match="unknown pilot candidate"):
        pilot(tmp_path, experiment, "pocs_grid", "invalid")


@pytest.mark.parametrize(
    "pipeline,supervise",
    [
        (forge_pipeline, forge_pipeline.supervise_forge_grid_experiment),
        (c3_pipeline, c3_pipeline.supervise_c3_region_experiment),
    ],
)
def test_supervisor_relaunch_refuses_live_workers_before_spawning_or_rewriting(
    tmp_path, monkeypatch, pipeline, supervise
):
    (tmp_path / "experiment.json").write_text(json.dumps({"methods": ["pocs_grid", "drr_grid"]}))
    workers = tmp_path / "workers"
    workers.mkdir()
    path = workers / "drr_grid.json"
    original = json.dumps(
        {
            "status": "running",
            "s1_completed": True,
            "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
        }
    )
    path.write_text(original)
    credits = workers / "drr_grid_retries.json"
    credits.write_text(json.dumps({"retries": 1}))
    monkeypatch.setattr(
        pipeline.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("duplicate worker started"),
    )
    with pytest.raises(RuntimeError, match="refusing duplicate live worker"):
        supervise(tmp_path, tmp_path, tmp_path / "script.py")
    assert path.read_text() == original
    assert json.loads(credits.read_text()) == {"retries": 1}
    assert not (workers / "supervisor.json").exists()
