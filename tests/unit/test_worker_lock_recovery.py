"""A dead parent's inherited flock is a bounded pending restart, not a failure."""

import fcntl
import json
import os
import time
from types import SimpleNamespace

import pytest

from seis_interp.evaluation import study067_rehearsal_checks as checks
from seis_interp.pipelines import run_c3_region_experiment as c3
from seis_interp.pipelines import run_forge_grid_experiment as forge
from seis_interp.training import worker_processes as workers
from seis_interp.training import worker_restarts as restarts
from tests.fixtures.worker_locks import orphaned_lock_holder


@pytest.mark.parametrize("only_dead", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
def test_orphaned_worker_lock_restarts_once_after_pending_window(
    tmp_path, monkeypatch, only_dead, replacement
):
    directory = tmp_path / "workers"
    (tmp_path / "experiment.json").write_text(json.dumps({"rehearsal": True}))
    name = "regsi_real"
    starts, attempts = [], []

    def start(method):
        starts.append(method)
        state = json.loads((directory / f"{method}.json").read_text())
        workers.write_worker_state(directory / f"{method}.json", state | {"status": "completed"})
        return SimpleNamespace(pid=os.getpid(), poll=lambda: 0)

    def restart(method):
        try:
            return restarts.restart_exited_worker(directory, method, start)
        except workers.WorkerLockBusy:
            attempts.append(method)
            assert not starts
            assert not (directory / f"{name}_retries.json").exists()
            assert not (directory / "rehearsal_restarts.jsonl").exists()
            raise

    with orphaned_lock_holder(directory, name) as holder:
        holder.exit_path.touch()
        assert holder.process.wait(timeout=5) == 137
        assert not workers.process_is_alive(holder.process.pid, holder.state["process_start_time"])
        assert holder.child_pid in workers.lock_holder_pids(directory / f"{name}.lock")
        processes = (
            restarts.start_dead_workers(directory, [name], start)
            if only_dead
            else {name: holder.process}
        )
        if replacement:
            monkeypatch.setattr(c3, "publish_c3_replacement_readiness", lambda _: None)
            sleep = time.sleep
            monkeypatch.setattr(c3.time, "sleep", lambda _: sleep(0.02))
            result = c3._wait_c3_replacement_workers(
                tmp_path, processes, {name: tmp_path}, restart=restart
            )
        else:
            result = workers.wait_for_workers(
                processes, poll=0.02, directory=directory, restart=restart
            )
    assert result == ({name: 0}, True)
    assert attempts and starts == [name]
    assert json.loads((directory / f"{name}_retries.json").read_text())["retries"] == 1
    # A delay between exit and restart is accepted by the unchanged rehearsal checker.
    assert checks._check_restart_evidence(
        tmp_path, {"faults": [{"method": name, "run": "a01_b05", "action": "exit"}]}
    )


@pytest.mark.parametrize("only_dead", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
def test_orphaned_worker_lock_grace_expires_with_holder_evidence_without_signals(
    tmp_path, monkeypatch, only_dead, replacement
):
    directory = tmp_path / "workers"
    name = "ccnet5d_grid_shard_1_of_2"
    monkeypatch.setattr(workers, "WORKER_LOCK_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("sent a process signal"))
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("sent a process group signal"))

    def start(_):
        pytest.fail("restarted while the orphan still held the lock")

    with orphaned_lock_holder(directory, name) as holder:
        holder.exit_path.touch()
        assert holder.process.wait(timeout=5) == 137
        processes = (
            restarts.start_dead_workers(directory, [name], start)
            if only_dead
            else {name: holder.process}
        )

        def restart(method):
            return restarts.restart_exited_worker(directory, method, start)

        if replacement:
            monkeypatch.setattr(c3, "publish_c3_replacement_readiness", lambda _: None)
            sleep = time.sleep
            monkeypatch.setattr(c3.time, "sleep", lambda _: sleep(0.01))
            result = c3._wait_c3_replacement_workers(tmp_path, processes, {}, restart=restart)
        else:
            result = workers.wait_for_workers(
                processes, poll=0.01, directory=directory, restart=restart
            )
        assert result == ({name: 1 if only_dead else 137}, False)
        state = json.loads((directory / f"{name}.json").read_text())
        assert state["status"] == "failed"
        assert "lock grace expired after 0.05s" in state["failure_reason"]
        assert str(directory / f"{name}.lock") in state["failure_reason"]
        assert str(holder.child_pid) in state["failure_reason"]
        assert workers.process_is_alive(
            holder.child_pid, workers.process_start_time(holder.child_pid)
        )
        with (
            (directory / f"{name}.lock").open("a") as lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (directory / f"{name}_retries.json").exists()
    # An operator can adopt again after the orphan exits; the pending window spent no credit.
    assert restarts.restart_exited_worker(directory, name, lambda _: "fresh") == "fresh"
    assert json.loads((directory / f"{name}_retries.json").read_text())["retries"] == 1


@pytest.mark.parametrize("status,reason", [("completed", ""), ("failed", "nonfinite loss")])
def test_busy_lock_does_not_make_a_terminal_worker_retryable(tmp_path, status, reason):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"status": status, "failure_reason": reason}))
    before = path.read_bytes()
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert (
            restarts.restart_exited_worker(tmp_path, "m", lambda _: pytest.fail("restart")) is None
        )
    assert path.read_bytes() == before
    assert not (tmp_path / "m_retries.json").exists()


@pytest.mark.parametrize("pipeline", [c3, forge])
@pytest.mark.parametrize("only_dead", [False, True])
@pytest.mark.parametrize("expires", [False, True])
def test_supervisors_handle_pending_ccnet_shard_without_spending_extra_credits(
    tmp_path, monkeypatch, pipeline, only_dead, expires
):
    directory = tmp_path / "workers"
    directory.mkdir()
    (tmp_path / "experiment.json").write_text(json.dumps({"methods": ["ccnet5d_grid"]}))
    study = tmp_path / "study"
    study.mkdir()
    (study / "config.yaml").write_text("method_execution:\n  ccnet5d_grid:\n    worker_shards: 2\n")
    shard = "ccnet5d_grid_shard_1_of_2"
    state = {
        "pid": 999999999,
        "process_start_time": 1,
        "status": "running",
        "phase": "S1",
        "active_run": "a01_b05",
        "completed_s1": 1,
        "completed_evaluation": 0,
    }
    (directory / f"{shard}.json").write_text(json.dumps(state))
    (directory / "ccnet5d_grid.json").write_text(json.dumps({"status": "completed"}))
    run = [state[key] for key in ("phase", "active_run", "completed_s1", "completed_evaluation")]
    counter = directory / f"{shard}_retries.json"
    counter.write_text(json.dumps({"run": run, "retries": 1}))
    starts = []
    now = 0.0
    monkeypatch.setattr(workers.time, "monotonic", lambda: now)
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("sent a process signal"))
    lock = (directory / f"{shard}.lock").open("a")
    if only_dead:
        fcntl.flock(lock, fcntl.LOCK_EX)

    def sleep(_):
        nonlocal now
        restarted = starts.count(shard) == (1 if only_dead else 2)
        assert json.loads(counter.read_text())["retries"] == (2 if restarted else 1)
        now += 60
        if not expires:
            fcntl.flock(lock, fcntl.LOCK_UN)

    def popen(command, **kwargs):
        name = "ccnet5d_grid" if command[-1] == "0/2" else shard
        starts.append(name)
        if name == shard and not only_dead and starts.count(shard) == 1:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return SimpleNamespace(pid=state["pid"], poll=lambda: 137)
        path = directory / f"{name}.json"
        path.write_text(json.dumps({"status": "completed"}))
        return SimpleNamespace(pid=12345, poll=lambda: 0)

    monkeypatch.setattr(workers.time, "sleep", sleep)
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    if pipeline is c3:
        supervise = c3.supervise_c3_region_experiment
        monkeypatch.setattr(c3, "_write_c3_experiment_aggregates", lambda *args: None)
    else:
        supervise = forge.supervise_forge_grid_experiment
        monkeypatch.setattr(forge, "summarize_forge_grid_experiment", lambda *args: None)
    try:
        if expires:
            with pytest.raises(RuntimeError, match="worker failures"):
                supervise(tmp_path, tmp_path, tmp_path / "worker.py", only_dead_workers=only_dead)
        else:
            supervise(tmp_path, tmp_path, tmp_path / "worker.py", only_dead_workers=only_dead)
        supervisor = json.loads((directory / "supervisor.json").read_text())
        assert supervisor["status"] == ("failed" if expires else "completed")
        assert json.loads(counter.read_text())["retries"] == (1 if expires else 2)
        assert starts.count(shard) == (0 if only_dead else 1) + (0 if expires else 1)
        if expires:
            failure = json.loads((directory / f"{shard}.json").read_text())
            assert "lock grace expired after 120s" in failure["failure_reason"]
    finally:
        lock.close()
