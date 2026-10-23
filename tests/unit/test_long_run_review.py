"""CPU regressions for long-run recovery review findings."""

import fcntl
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from seis_interp.data import experiment_import_coverage as coverage
from seis_interp.evaluation.c3_region_results import summarize_c3_region_seeds
from seis_interp.evaluation.forge_grid_results import summarize_forge_grid_seeds
from seis_interp.training import exact_resume, gpu_admission, run_recovery
from seis_interp.training.worker_processes import process_start_time
from seis_interp.training.worker_restarts import restart_exited_worker


@pytest.mark.parametrize("method", ["pocs_grid", "drr_grid"])
def test_nonresumable_methods_have_no_progress_watchdog(tmp_path, monkeypatch, method):
    now = 0.0
    monkeypatch.setattr(run_recovery.time, "monotonic", lambda: now)
    dog = run_recovery.run_watchdog(tmp_path, {"method_id": method})
    dog.start()
    now += 10 * 24 * 3600
    dog.report("DRR blocks 1/4")
    assert not dog.enabled and dog.thread.ident is None
    dog.close()
    assert not (tmp_path / "watchdog.json").exists()


@pytest.mark.parametrize("method", ["pocs_grid", "drr_grid"])
def test_classical_wall_limit_does_not_reset_on_reports(tmp_path, monkeypatch, method):
    now = 0.0
    monkeypatch.setattr(run_recovery.time, "monotonic", lambda: now)
    dog = run_recovery.run_watchdog(
        tmp_path,
        {
            "method_id": method,
            "execution": {"wall_clock_limit_seconds": 7 * 24 * 3600},
        },
    )
    now = 6 * 24 * 3600
    dog.report("DRR blocks 1/4")
    now = 7 * 24 * 3600 + 1
    exits = []
    dog.exit_process = exits.append
    monkeypatch.setattr(dog.stopped, "wait", lambda _: False)
    dog._watch()
    assert exits == [75]


def test_neural_watchdog_uses_floor_and_high_quantile_including_slow_phases(tmp_path, monkeypatch):
    now = 0.0
    monkeypatch.setattr(run_recovery.time, "monotonic", lambda: now)
    dog = run_recovery.ProgressWatchdog(tmp_path, 1)
    for interval, message in [
        (1, "step 100/500"),
        (30, "step 200/500"),
        (900, "checkpoint saved"),
        (1000, "prediction complete"),
    ]:
        now += interval
        dog.report(message)
    expected = max(1800, 3 * np.quantile([1, 30, 900, 1000], 0.95))
    assert dog.threshold == expected
    now += 1
    dog.report("step 300/500")
    assert dog.threshold > 1800
    assert run_recovery.ProgressWatchdog(tmp_path, 1).threshold == 1800


@pytest.mark.parametrize(
    "message",
    ["selected prediction and target traces must be finite", "non-finite metric", "nonfinite loss"],
)
def test_nonfinite_value_errors_are_divergence(message):
    assert run_recovery.failure_kind(ValueError(message)) == "diverged"
    assert run_recovery.failure_kind(FloatingPointError("overflow")) == "diverged"


def test_retry_budget_counts_errors_before_manifest_rewrite_with_backoff(tmp_path, monkeypatch):
    now = 10.0
    sleeps, calls = [], []
    monkeypatch.setattr(run_recovery.time, "time", lambda: now)

    def sleep(seconds):
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(run_recovery.time, "sleep", sleep)
    output = tmp_path / "method_runs/m/evaluation/seed_1"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps({"status": "awaiting_barrier", "artifacts": {}})
    )

    @run_recovery.retry_run
    def broken(*args, **kwargs):
        calls.append(1)
        raise OSError("connection reset before manifest rewrite")

    with pytest.raises(RuntimeError, match="retry limit exhausted"):
        broken(tmp_path, tmp_path, {"id": "a", "config": {"method_id": "m"}}, 1, "evaluation")
    assert len(calls) == 3 and sleeps == [5, 10]
    assert json.loads((output / "retry_state.json").read_text())["failed_attempts"] == 3
    assert len(list((output / "attempts").iterdir())) == 3


def test_low_disk_waits_without_consuming_attempts(tmp_path, monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(run_recovery.time, "sleep", sleeps.append)

    @run_recovery.retry_run
    def admitted(*args, **kwargs):
        calls.append(1)
        if len(calls) < 4:
            run_recovery.check_free_disk(tmp_path)
        return "done"

    monkeypatch.setattr(run_recovery.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    assert (
        admitted(tmp_path, tmp_path, {"id": "a", "config": {"method_id": "m"}}, 1, "S1") == "done"
    )
    assert sleeps == [5, 5, 5]
    assert not (tmp_path / "method_runs/m/S1/a").exists()


def test_peer_barrier_failure_does_not_poison_a_seed_attempt(tmp_path):
    @run_recovery.retry_run
    def waiting(*args, **kwargs):
        raise run_recovery.S1BarrierError("other failed: worker exited with code 75")

    with pytest.raises(run_recovery.S1BarrierError):
        waiting(tmp_path, tmp_path, {"id": "a", "config": {"method_id": "m"}}, 1, "evaluation")
    assert not (tmp_path / "method_runs/m/evaluation/seed_1").exists()


def test_sticky_cuda_error_requires_fresh_process_before_retry(tmp_path, monkeypatch):
    calls = []

    @run_recovery.retry_run
    def sticky(*args, **kwargs):
        calls.append(os.getpid())
        raise RuntimeError("CUDA error: an illegal memory access")

    monkeypatch.setattr(
        run_recovery.torch.cuda, "empty_cache", lambda: pytest.fail("poisoned context reused")
    )
    with pytest.raises(run_recovery.RestartRunProcess, match="fresh process required"):
        sticky(tmp_path, tmp_path, {"id": "a", "config": {"method_id": "m"}}, 1, "S1")
    assert len(calls) == 1
    assert (tmp_path / "method_runs/m/S1/a/attempts/1/manifest.json").exists()


def test_restart_budget_changes_with_run_and_preserves_s1(tmp_path):
    path = tmp_path / "m.json"
    for run in ("a", "b", "seed_7321"):
        path.write_text(
            json.dumps({"status": "retry_pending", "s1_completed": True, "active_run": run})
        )
        for _ in range(2):
            assert restart_exited_worker(tmp_path, "m", lambda _: "fresh") == "fresh"
        assert restart_exited_worker(tmp_path, "m", lambda _: "fresh") is None
        assert json.loads(path.read_text())["s1_completed"]


def test_live_worker_restart_preserves_state_and_credits(tmp_path):
    path = tmp_path / "m.json"
    original = json.dumps(
        {
            "status": "running",
            "s1_completed": True,
            "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
        }
    )
    path.write_text(original)
    assert restart_exited_worker(tmp_path, "m", lambda _: pytest.fail("duplicate")) is None
    assert path.read_text() == original
    assert not (tmp_path / "m_retries.json").exists()
    path.write_text(json.dumps({"status": "running"}))
    with (tmp_path / "m.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert (
            restart_exited_worker(tmp_path, "m", lambda _: pytest.fail("locked duplicate")) is None
        )
    assert not (tmp_path / "m_retries.json").exists()


def test_memory_choice_reserves_before_another_worker_selects(tmp_path, monkeypatch):
    candidates = [
        {"id": name, "config": {"device": "cuda:0", "method_id": "m"}}
        for name in ("large", "small")
    ]
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"m": {"large": 12, "small": 4}},
    }
    monkeypatch.setattr(
        gpu_admission.torch.cuda, "mem_get_info", lambda _: (20 * 2**30, 24 * 2**30)
    )
    first = next(gpu_admission.memory_fitting_candidates(tmp_path, candidates, resources))
    second = next(gpu_admission.memory_fitting_candidates(tmp_path, candidates, resources))
    assert first["id"] == "large" and second["id"] == "small"
    assert len(gpu_admission.live_admission_slots(tmp_path / "runs/.gpu_admission")) == 2


@pytest.mark.parametrize("status", ["completed", "diverged", "imported"])
def test_finalized_and_imported_s1_cells_bypass_all_cuda_queries(tmp_path, monkeypatch, status):
    study = tmp_path / "study"
    study.mkdir()
    (study / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "trace_split": {"search_seed": 1},
                "matched_training": {"loss_traces_per_update": 1},
                "import_from": ["source"],
            }
        )
    )
    candidate = {"id": "large", "config": {"device": "cuda:0", "method_id": "m"}}
    if status != "imported":
        output = tmp_path / "method_runs/m/S1/large"
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps({"status": status}))
    else:
        monkeypatch.setattr(
            coverage,
            "load_c3_region_inputs",
            lambda *args: (SimpleNamespace(observed_cells=[1]), None),
        )
        monkeypatch.setattr(coverage, "saved_input_fingerprint", lambda *args: "fingerprint")
        monkeypatch.setattr(
            coverage, "find_prediction_import", lambda *args: {"source_path": "pilot"}
        )
    ready = coverage.s1_candidates_without_admission(tmp_path, tmp_path, [candidate], "c3")
    monkeypatch.setattr(
        gpu_admission.torch.cuda, "mem_get_info", lambda _: pytest.fail("CUDA queried")
    )
    monkeypatch.setattr(
        gpu_admission, "live_admission_slots", lambda _: pytest.fail("slots queried")
    )
    assert list(
        gpu_admission.memory_fitting_candidates(
            tmp_path, [candidate], {"global_admission": True}, without_admission=ready
        )
    ) == [candidate]


@pytest.mark.parametrize(
    "summarize,forge", [(summarize_c3_region_seeds, False), (summarize_forge_grid_seeds, True)]
)
def test_seed_summary_reports_actual_completed_count_and_divergence(summarize, forge):
    metrics = {
        key: 1.0
        for key in (
            "masked_trace_relative_mse",
            "global_nmse",
            "snr_db",
            "mean_trace_snr_db",
            "rmse",
        )
    }
    metrics["correlation"] = {key: 0.5 for key in ("p05", "p25", "median", "p75", "p95")}
    if forge:
        metrics = {p: deepcopy(metrics) for p in ("all_eligible", "clean_target")}
    records = [
        {
            "trace_seed": seed,
            "id": "a",
            "method_id": "m",
            "status": "completed",
            "metrics": {d: deepcopy(metrics) for d in ("validation", "test")},
        }
        for seed in (7321, 7322, 7323)
    ]
    complete = summarize(records)
    assert set(complete.seed_count) == {3} and complete["mean"].notna().all()
    records[1].update(status="diverged", metrics={})
    summary = summarize(records)
    assert set(summary.seed_count) == {2} and summary["mean"].isna().all()
    assert set(summary.diverged_seeds) == {"7322"}
    missing = summarize(records[:2], expected_seeds=[7321, 7322, 7323])
    assert set(missing.missing_seeds) == {"7323"} and set(missing.seed_count) == {1}
    records[1]["status"] = "awaiting_barrier"
    with pytest.raises(ValueError, match="post-barrier"):
        summarize(records)


def test_resume_loads_cpu_steps_and_snapshots_but_moves_live_device_state(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    recovery = exact_resume.ExactResume(tmp_path / "resume.pt")
    recovery.save(
        1,
        model,
        optimizer,
        {
            "objects": {
                "ema": {
                    "_state": {"weight": model.weight.detach().clone()},
                    "snapshots": {1: {"weight": model.weight.detach().clone()}},
                }
            }
        },
        force=True,
    )
    original = torch.load
    maps = []

    def load(*args, **kwargs):
        maps.append(kwargs["map_location"])
        return original(*args, **kwargs)

    monkeypatch.setattr(exact_resume.torch, "load", load)
    state = recovery.load(model, optimizer)
    assert maps == ["cpu"]
    assert all(item["step"].is_cpu for item in optimizer.state.values())
    assert state["objects"]["ema"]["snapshots"][1]["weight"].is_cpu
    moved = exact_resume._device_training_state(state, "meta")
    assert moved["objects"]["ema"]["_state"]["weight"].device.type == "meta"
    assert moved["objects"]["ema"]["snapshots"][1]["weight"].is_cpu


def test_admission_consumes_choice_reservation_without_double_counting(tmp_path, monkeypatch):
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"m": 12},
    }
    candidate = {"id": "a", "config": {"device": "cuda:0", "method_id": "m"}}
    monkeypatch.setattr(
        gpu_admission.torch.cuda, "mem_get_info", lambda _: (12 * 2**30, 24 * 2**30)
    )
    selected = next(gpu_admission.memory_fitting_candidates(tmp_path, [candidate], resources))
    monkeypatch.setattr(
        gpu_admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=24 * 2**30),
    )
    monkeypatch.setattr(gpu_admission.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        gpu_admission.torch.cuda, "set_per_process_memory_fraction", lambda *args: None
    )
    monkeypatch.setattr(gpu_admission.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(gpu_admission.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(gpu_admission.torch.cuda, "max_memory_reserved", lambda _: 0)
    with gpu_admission.candidate_admission(
        tmp_path,
        candidate["config"],
        resources,
        "a",
        pytest.fail,
        reservation=selected["admission_reservation"],
    ) as device:
        assert device == "cuda:0"
        slots = gpu_admission.live_admission_slots(tmp_path / "runs/.gpu_admission")
        assert len(slots) == 1
    assert gpu_admission.live_admission_slots(tmp_path / "runs/.gpu_admission") == []
