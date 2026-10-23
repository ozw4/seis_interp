import json
from types import SimpleNamespace

import pytest
import torch

from seis_interp.training import exact_resume, gpu_admission, run_recovery
from seis_interp.training.worker_processes import wait_for_workers
from seis_interp.training.worker_restarts import restart_exited_worker
from seis_interp.training.worker_shards import parse_shard, shard_candidates


def test_checkpoint_is_atomic_and_due_by_updates_or_elapsed_time(tmp_path, monkeypatch):
    now = 10.0
    monkeypatch.setattr(exact_resume.time, "monotonic", lambda: now)
    path = tmp_path / "resume.pt"
    recovery = exact_resume.ExactResume(path)
    assert not recovery.due(2499)
    assert recovery.due(2500)
    now += 1800
    assert recovery.due(1)
    exact_resume.atomic_torch_save({"value": torch.tensor([1])}, path)
    previous = path.read_bytes()

    def fail(*args):
        raise OSError("disk write failed")

    monkeypatch.setattr(exact_resume.torch, "save", fail)
    with pytest.raises(OSError, match="disk write"):
        exact_resume.atomic_torch_save({"value": 2}, path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_shards_are_disjoint_and_cover_every_candidate():
    candidates = [{"id": f"a{i:02}"} for i in range(25)]
    shards = [shard_candidates(candidates, f"{i}/3") for i in range(3)]
    flattened = [item["id"] for shard in shards for item in shard]
    assert len(flattened) == len(set(flattened)) == 25
    assert set(flattened) == {item["id"] for item in candidates}
    for value in ("1/1", "-1/2", "0/0", "1", "bad"):
        with pytest.raises(ValueError, match="shard must"):
            parse_shard(value)


def test_next_candidate_is_the_highest_priority_that_fits_memory(tmp_path, monkeypatch):
    candidates = [
        {"id": name, "config": {"device": "cuda:0", "method_id": "regsi_real"}}
        for name in ("large", "medium", "small")
    ]
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"regsi_real": {"large": 12, "medium": 6, "small": 2}},
    }
    free = 7 * 2**30
    monkeypatch.setattr(
        gpu_admission, "gpu_memory_info", lambda devices: {d: (free, 20 * 2**30) for d in devices}
    )
    ordered = gpu_admission.memory_fitting_candidates(tmp_path, candidates, resources)
    assert next(ordered)["id"] == "medium"
    gpu_admission.Path(next((tmp_path / "runs/.gpu_admission").glob("slot_*.json"))).unlink()
    free = 15 * 2**30
    assert next(ordered)["id"] == "large"
    assert next(ordered)["id"] == "small"


def test_disk_admission_and_gpu_utilisation_do_not_initialize_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_recovery.shutil, "disk_usage", lambda _: SimpleNamespace(free=4 * 2**30)
    )
    with pytest.raises(OSError, match="insufficient free disk"):
        run_recovery.check_free_disk(tmp_path)
    assert run_recovery.check_free_disk(tmp_path, 3) == 4 * 2**30
    queries = []

    def query(command, **kwargs):
        queries.append(command)
        return "0, GPU-123, 73, 1200, 8000\n"

    monkeypatch.setattr(run_recovery.subprocess, "check_output", query)
    assert run_recovery.gpu_utilization("cuda:0") == ["0, GPU-123, 73, 1200, 8000"]
    assert "utilization.gpu" in queries[0][1]
    assert run_recovery.gpu_utilization("cpu") == []


def test_watchdog_exits_only_its_own_worker_at_three_intervals(tmp_path, monkeypatch):
    now = 0.0
    monkeypatch.setattr(run_recovery.time, "monotonic", lambda: now)
    exits = []
    watchdog = run_recovery.ProgressWatchdog(
        tmp_path, 10, floor_seconds=30, exit_process=exits.append
    )
    waits = []

    def wait(_):
        nonlocal now
        now = 30 if not waits else 31
        waits.append(now)
        return False

    monkeypatch.setattr(watchdog.stopped, "wait", wait)
    watchdog._watch()
    assert waits == [30, 31] and exits == [75]
    assert json.loads((tmp_path / "watchdog.json").read_text())["threshold_seconds"] == 30


def test_supervisor_waits_for_a_pending_retry_without_signals(tmp_path):
    starts = []
    processes = {"regsi_real": SimpleNamespace(poll=lambda: 75)}

    def start(method):
        starts.append(method)
        (tmp_path / f"{method}.json").write_text(json.dumps({"status": "completed"}))
        return SimpleNamespace(pid=12345, poll=lambda: 0)

    outcomes, completed = wait_for_workers(
        processes,
        poll=0,
        directory=tmp_path,
        restart=lambda method: restart_exited_worker(tmp_path, method, start),
    )
    assert starts == ["regsi_real"] and completed and outcomes == {"regsi_real": 0}
    assert json.loads((tmp_path / "regsi_real_retries.json").read_text())["retries"] == 1


def test_worker_restarts_are_bounded_and_nonfinite_is_not_retried(tmp_path):
    path = tmp_path / "regsi_real.json"

    def start(_):
        return "new worker"

    for _ in range(2):
        assert restart_exited_worker(tmp_path, "regsi_real", start) == "new worker"
    assert restart_exited_worker(tmp_path, "regsi_real", start) is None
    path.write_text(json.dumps({"status": "failed", "failure_reason": "nonfinite loss"}))
    assert restart_exited_worker(tmp_path, "regsi_real", start) is None


@pytest.mark.parametrize("dataset", ["c3_region", "forge_grid"])
@pytest.mark.parametrize("method", ["pocs_grid", "drr_grid", "regsi_real"])
def test_worker_cli_passes_the_shard_and_nices_cpu_methods(tmp_path, monkeypatch, dataset, method):
    import importlib
    import sys

    script = importlib.import_module(f"scripts.run_{dataset}_experiment")
    pipeline = importlib.import_module(f"seis_interp.pipelines.run_{dataset}_experiment")
    nice, launched = [], []
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(script.os, "getpriority", lambda *args: 0)
    monkeypatch.setattr(script.os, "nice", nice.append)
    monkeypatch.setattr(
        pipeline, f"run_{dataset}_worker", lambda *args, **kwargs: launched.append((args, kwargs))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "worker",
            "--repo",
            str(tmp_path),
            "--experiment",
            str(tmp_path),
            "--method",
            method,
            "--shard",
            "1/2",
        ],
    )
    script.main()
    assert launched == [((tmp_path, tmp_path, method), {"shard": "1/2"})]
    assert nice == ([] if method == "regsi_real" else [10])
