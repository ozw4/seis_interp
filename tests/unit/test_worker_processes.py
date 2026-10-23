import fcntl
import json
import os
from types import SimpleNamespace

import pytest

from seis_interp.pipelines.run_c3_region_experiment import c3_s1_barrier_ready
from seis_interp.pipelines.run_forge_grid_experiment import forge_s1_barrier_ready
from seis_interp.training import worker_processes as workers


def test_process_identity_and_proc_stat_fields():
    assert workers.process_is_alive(os.getpid(), workers.process_start_time(os.getpid()))
    assert not workers.process_is_alive(os.getpid(), -1)
    assert not workers.process_is_alive(999999999, None)


@pytest.mark.parametrize("barrier", [forge_s1_barrier_ready, c3_s1_barrier_ready])
@pytest.mark.parametrize("status", ["running", "queued"])
@pytest.mark.parametrize("stat", [None, ("Z", 10), ("R", 11)])
def test_barriers_fail_on_missing_zombie_or_reused_worker(
    tmp_path, monkeypatch, barrier, status, stat
):
    (tmp_path / "experiment.json").write_text(json.dumps({"methods": ["pocs_grid"]}))
    path = tmp_path / "workers/pocs_grid.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps({"status": status, "pid": 123, "process_start_time": 10, "s1_completed": True})
    )
    monkeypatch.setattr(workers, "process_stat", lambda _: stat)
    with pytest.raises(RuntimeError, match="pocs_grid failed.*missing, zombie"):
        barrier(tmp_path)


def test_poll_checks_all_workers_and_returns_without_waiting_or_signalling(monkeypatch):
    calls = []

    def process(name, code):
        def poll():
            calls.append(name)
            return code

        return SimpleNamespace(poll=poll)

    monkeypatch.setattr(workers.time, "sleep", lambda _: pytest.fail("slept after a failure"))
    outcomes, completed = workers.wait_for_workers(
        {"first": process("first", None), "failed": process("failed", 2)}
    )
    assert calls == ["first", "failed"]
    assert outcomes == {"first": None, "failed": 2} and not completed


def test_duplicate_worker_exit_leaves_live_foreign_state_untouched(tmp_path):
    path = tmp_path / "pocs_grid.json"
    state = {
        "status": "running",
        "pid": os.getpid(),
        "process_start_time": workers.process_start_time(os.getpid()),
        "s1_completed": True,
    }
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    temporary = path.with_suffix(".tmp")
    temporary.write_text("worker publication in progress")

    workers.record_worker_exits(
        tmp_path, {"pocs_grid": SimpleNamespace(pid=999999999)}, {"pocs_grid": 1}
    )

    assert path.read_bytes() == before
    assert temporary.read_text() == "worker publication in progress"
    assert workers.worker_failure(json.loads(path.read_text())) is None


@pytest.mark.parametrize("stat", [None, ("Z", 10), ("X", 10), ("R", 11)])
@pytest.mark.parametrize("child_pid", [123, 456])
def test_worker_exit_records_failure_for_dead_identity_using_distinct_temp_file(
    tmp_path, monkeypatch, stat, child_pid
):
    path = tmp_path / "pocs_grid.json"
    path.write_text(json.dumps({"status": "running", "pid": 123, "process_start_time": 10}))
    temporary = path.with_suffix(".tmp")
    temporary.write_text("worker publication in progress")
    monkeypatch.setattr(workers, "process_stat", lambda _: stat)

    assert workers.record_worker_exits(
        tmp_path, {"pocs_grid": SimpleNamespace(pid=child_pid)}, {"pocs_grid": 1}
    )

    state = json.loads(path.read_text())
    assert state["status"] == "failed" and state["pid"] == child_pid
    assert state["exit_code"] == 1
    assert "worker exited with code 1" in state["failure_reason"]
    assert temporary.read_text() == "worker publication in progress"
    assert list(tmp_path.glob("*.tmp")) == [temporary]


def test_worker_exit_does_not_publish_while_a_replacement_worker_is_starting(tmp_path):
    path = tmp_path / "pocs_grid.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workers.record_worker_exits(
            tmp_path, {"pocs_grid": SimpleNamespace(pid=999999999)}, {"pocs_grid": 1}
        )
        assert not path.exists()


@pytest.mark.parametrize("stage", ["S1", "evaluation"])
def test_admission_restores_running_phase_and_keeps_summary_on_later_state_writes(tmp_path, stage):
    path = tmp_path / "workers/ccnet5d_grid.json"
    path.parent.mkdir()
    state = {"status": "running", "phase": stage, "pid": os.getpid()}
    workers.write_worker_state(path, state)
    workers.report_admission_wait(tmp_path, "ccnet5d_grid", "a01_b01", {"reason": "busy"})
    assert json.loads(path.read_text())["phase"] == "waiting_for_gpu"
    workers.report_admission(
        tmp_path,
        "ccnet5d_grid",
        "a01_b01",
        stage,
        {"device": "cuda:1", "wait_seconds": 3.5},
    )
    admitted = json.loads(path.read_text())
    assert admitted["phase"] == stage
    assert admitted["active_run"] == "a01_b01"
    assert "resource_wait" not in admitted
    assert admitted["last_admission"] == {"run": "a01_b01", "device": "cuda:1", "wait_seconds": 3.5}
    workers.write_worker_state(path, state)
    assert json.loads(path.read_text())["last_admission"] == admitted["last_admission"]
    workers.report_admission(
        tmp_path,
        "ccnet5d_grid",
        "a02_b01",
        stage,
        {"device": "cuda:0", "wait_seconds": 1.0},
    )
    workers.write_worker_state(path, state)
    assert json.loads(path.read_text())["last_admission"]["wait_seconds"] == 1.0
