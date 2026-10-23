import pytest

from seis_interp.forge_mvp_config import validate_classical_execution
from seis_interp.training.candidate_schedule import candidate_outcomes


def _pool_candidate_with_failure(candidate):
    import time
    from pathlib import Path

    directory = Path(candidate["directory"])
    (directory / candidate["id"]).touch()
    deadline = time.monotonic() + 15
    expected = "running" if candidate["id"] == "fail" else "failure_reported"
    while not (directory / expected).exists():
        if time.monotonic() > deadline:
            raise TimeoutError(expected)
        time.sleep(0.01)
    if candidate["id"] == "fail":
        raise ValueError("first candidate failed")
    return candidate["id"]


def test_real_cpu_pool_does_not_start_queued_candidates_after_failure(tmp_path):
    candidates = [
        {"id": name, "directory": str(tmp_path), "config": {"device": "cpu"}}
        for name in ["fail", "running", "pending1", "pending2", "pending3"]
    ]
    with pytest.raises(ValueError, match="first candidate failed"):
        list(
            candidate_outcomes(
                _pool_candidate_with_failure,
                candidates,
                2,
                failed=lambda _: (tmp_path / "failure_reported").touch(),
            )
        )
    assert {path.name for path in tmp_path.iterdir()} == {"fail", "running", "failure_reported"}


def _successful_candidate(candidate):
    from pathlib import Path

    (Path(candidate["directory"]) / str(candidate["id"])).touch()
    return candidate["id"]


def test_isolated_jobs_submit_successful_candidates_incrementally(tmp_path):
    candidates = [{"id": i, "directory": str(tmp_path)} for i in range(5)]
    results = candidate_outcomes(_successful_candidate, candidates, 2)
    first = next(results)
    assert not (tmp_path / "2").exists()
    assert {first, *results} == set(range(5))


def _watchdog_candidate(candidate):
    import os
    import time
    from pathlib import Path

    from seis_interp.training.run_recovery import ProgressWatchdog

    directory = Path(candidate["directory"])
    if candidate["id"] == "stalled":
        attempt = directory / "stalled_attempts"
        number = int(attempt.read_text()) + 1 if attempt.exists() else 1
        attempt.write_text(str(number))
        if number == 1:
            watchdog = ProgressWatchdog(directory, 0.01, floor_seconds=0.01)
            watchdog.start()
            time.sleep(2)
        return os.getpid()
    started = os.getpid()
    while not (directory / "watchdog.json").exists():
        time.sleep(0.01)
    (directory / "healthy_finished").write_text(str(started))
    return started


def test_watchdog_restarts_only_stalled_job_without_signalling_sibling(tmp_path, monkeypatch):
    import multiprocessing.process
    import os

    def forbidden(*args, **kwargs):
        pytest.fail("no process may signal a sibling")

    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(multiprocessing.process.BaseProcess, "terminate", forbidden)
    jobs = [{"id": name, "directory": str(tmp_path)} for name in ("stalled", "healthy")]
    changes = []
    results = list(
        candidate_outcomes(_watchdog_candidate, jobs, 2, children_changed=changes.append)
    )
    assert len(results) == 2
    assert (tmp_path / "healthy_finished").exists()
    assert (tmp_path / "stalled_attempts").read_text() == "2"
    healthy_pid = int((tmp_path / "healthy_finished").read_text())
    assert healthy_pid in results
    assert changes[-1] == []


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_classical_candidate_thread_count_must_be_a_positive_integer(threads):
    with pytest.raises(ValueError, match="execution settings"):
        validate_classical_execution(
            {
                "method_id": "pocs_grid",
                "device": "cpu",
                "execution": {"backend": "batched", "cpu_threads": threads},
            }
        )


def test_classical_candidate_thread_counts_are_optional_and_explicit():
    for method in ("pocs_grid", "drr_grid"):
        for execution in ({"backend": "batched"}, {"backend": "batched", "cpu_threads": 2}):
            validate_classical_execution(
                {"method_id": method, "device": "cpu", "execution": execution}
            )


def test_parallelism_rejects_invalid_process_counts():
    candidate = {"id": "a01_b01", "config": {"device": "cuda:0"}}
    with pytest.raises(ValueError, match="positive integer"):
        list(candidate_outcomes(None, [candidate], True))


def _sticky_candidate(candidate):
    import os
    from pathlib import Path

    from seis_interp.training.run_recovery import retry_run

    directory = Path(candidate["directory"])

    @retry_run
    def run(*args, **kwargs):
        original = directory / "poisoned_pid"
        if not original.exists():
            original.write_text(str(os.getpid()))
            raise RuntimeError("CUDA error: unspecified launch failure")
        return os.getpid()

    return run(directory, directory, {"id": "a", "config": {"method_id": "m"}}, 1, "S1")


def test_sticky_cuda_retry_executes_in_a_new_process(tmp_path):
    jobs = [{"directory": str(tmp_path)}]
    [pid] = candidate_outcomes(_sticky_candidate, jobs, 2)
    assert pid != int((tmp_path / "poisoned_pid").read_text())
    assert (tmp_path / "method_runs/m/S1/a/attempts/1/manifest.json").exists()
