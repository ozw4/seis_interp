"""Regression tests for reservation deadlock, atomic publication and recovery visibility."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from seis_interp.training import gpu_admission as admission
from seis_interp.training import run_recovery
from seis_interp.training.worker_processes import process_start_time
from seis_interp.training.worker_restarts import start_dead_workers


def test_two_19gib_reservations_do_not_deadlock_when_free_drops_40_to_37(tmp_path, monkeypatch):
    """Both holders used to wait forever at 37 - 19 < 19 despite either fitting."""
    free = 40 * 2**30
    resources = {
        "global_admission": True,
        "devices": ["cuda:1"],
        "minimum_free_memory_gib": {"regsi_real": 19, "ccnet5d_grid": 19},
    }
    monkeypatch.setattr(admission.torch.cuda, "mem_get_info", lambda _: (free, 48 * 2**30))
    monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: 0)
    selected = []
    for method in ("regsi_real", "ccnet5d_grid"):
        candidate = {"id": "a01_b01", "config": {"method_id": method, "device": "auto"}}
        selected.append(next(admission.memory_fitting_candidates(tmp_path, [candidate], resources)))
    directory = tmp_path / "runs/.gpu_admission"
    slots = admission.live_admission_slots(directory)
    assert len(slots) == 2
    first, second = [
        json.loads(Path(item["admission_reservation"]).read_text()) for item in selected
    ]
    free = 37 * 2**30
    # The older reservation can progress; the younger still honors its predecessor.
    assert (
        admission.choose_admission_device(
            ["cuda:1"], {"cuda:1": free}, admission.admission_predecessors(slots, first), 19 * 2**30
        )
        == "cuda:1"
    )
    assert (
        admission.choose_admission_device(
            ["cuda:1"],
            {"cuda:1": free},
            admission.admission_predecessors(slots, second),
            19 * 2**30,
        )
        is None
    )
    monkeypatch.setattr(
        admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=48 * 2**30),
    )
    monkeypatch.setattr(admission.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "set_per_process_memory_fraction", lambda *_: None)
    monkeypatch.setattr(admission.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: 0)
    for item in selected:
        with admission.candidate_admission(
            tmp_path,
            item["config"],
            resources,
            item["id"],
            pytest.fail,
            reservation=item["admission_reservation"],
        ) as device:
            assert device == "cuda:1"
    assert admission.live_admission_slots(directory) == []


def test_fifo_never_ignores_an_active_jobs_regrowth_reservation():
    own = {"reservation": "older", "reserved_at_ns": 10, "device": "cuda:0"}
    active = {"device": "cuda:0", "required_bytes": 20, "current_reserved_bytes": 0}
    younger = {"reservation": "younger", "reserved_at_ns": 20, "device": "cuda:0"}
    assert admission.admission_predecessors([own, active, younger], own) == [active]
    assert admission.choose_admission_device(["cuda:0"], {"cuda:0": 37}, [active], 19) is None


def test_slot_publication_is_atomic_and_reads_tolerate_partial_legacy_files(tmp_path, monkeypatch):
    path = tmp_path / "slot_current.json"
    slot = {
        "pid": os.getpid(),
        "process_start_time": process_start_time(os.getpid()),
        "device": "cuda:0",
        "required_bytes": 20,
        "current_reserved_bytes": 0,
    }
    admission.write_admission_slot(path, slot)
    previous = path.read_bytes()

    def interrupted(temporary, value):
        temporary.write_text('{"pid":')
        assert path.read_bytes() == previous
        assert admission.live_admission_slots(tmp_path) == [slot]
        raise OSError("interrupted write")

    monkeypatch.setattr(admission, "write_json", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        admission.write_admission_slot(path, slot | {"required_bytes": 25})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".slot_*"))
    for name, contents in (
        ("partial", '{"pid":'),
        ("empty", ""),
        ("list", "[]"),
        ("missing", "{}"),
    ):
        (tmp_path / f"slot_{name}.json").write_text(contents)
    assert admission.live_admission_slots(tmp_path) == [slot]


def test_resource_wait_logs_first_and_each_minute_and_publishes_phase_without_attempts(
    tmp_path, monkeypatch, capsys
):
    now, calls = [0.0], []
    workers = tmp_path / "workers"
    workers.mkdir()
    path = workers / "m.json"
    path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": "running",
                "phase": "S1",
                "active_run": "old",
                "s1_completed": True,
            }
        )
    )
    monkeypatch.setattr(run_recovery.time, "monotonic", lambda: now[0])

    def sleep(seconds):
        state = json.loads(path.read_text())
        assert state["phase"] == "waiting_for_resource"
        assert "disk" in state["resource_wait"]["reason"]
        assert state["active_run"] == "a"
        now[0] += seconds

    monkeypatch.setattr(run_recovery.time, "sleep", sleep)

    @run_recovery.retry_run
    def waiting(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 26:
            raise run_recovery.ResourceWait("insufficient free disk")
        return "done"

    assert waiting(tmp_path, tmp_path, {"id": "a", "config": {"method_id": "m"}}, 1, "S1") == "done"
    assert capsys.readouterr().out.count("ResourceWait: insufficient free disk") == 3
    state = json.loads(path.read_text())
    assert state["phase"] == "S1" and state["s1_completed"]
    assert "resource_wait" not in state
    assert not (tmp_path / "method_runs/m/S1/a").exists()


def test_only_dead_mode_preserves_live_and_completed_workers_and_restarts_retry_pending(tmp_path):
    state = {
        "pid": os.getpid(),
        "process_start_time": process_start_time(os.getpid()),
        "status": "running",
        "s1_completed": True,
        "phase": "waiting_for_all_S1",
    }
    live = tmp_path / "live.json"
    live.write_text(json.dumps(state))
    completed = tmp_path / "completed.json"
    completed.write_text(json.dumps({"status": "completed"}))
    retry = tmp_path / "dead.json"
    retry.write_text(
        json.dumps({"pid": 999999999, "status": "retry_pending", "s1_completed": True})
    )
    originals = {path: path.read_bytes() for path in (live, completed)}
    starts = []

    def start(name):
        starts.append(name)
        return SimpleNamespace(pid=234, poll=lambda: None)

    processes = start_dead_workers(tmp_path, ["live", "completed", "dead"], start)
    assert starts == ["dead"]
    assert processes["live"].poll() is None and processes["completed"].poll() == 0
    assert json.loads(retry.read_text())["s1_completed"] is True
    assert {path: path.read_bytes() for path in originals} == originals
    assert not (tmp_path / "live_retries.json").exists()


@pytest.fixture
def fake_cuda(monkeypatch):
    free = {"cuda:0": 10 * 2**30, "cuda:1": 49 * 2**30}
    monkeypatch.setattr(
        admission.torch.cuda, "mem_get_info", lambda device: (free[device], 93 * 2**30)
    )
    monkeypatch.setattr(
        admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=93 * 2**30),
    )
    for name in ("set_device", "set_per_process_memory_fraction", "reset_peak_memory_stats"):
        monkeypatch.setattr(admission.torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: 0)
    return free


def test_older_reservation_cannot_take_younger_jobs_other_device(tmp_path, fake_cuda):
    resources = {
        "global_admission": True,
        "devices": ["cuda:0", "cuda:1"],
        "minimum_free_memory_gib": {"nersi_profile": 6, "regsi_real": 48},
    }
    # Reserve the small job first on GPU 0, then give GPU 1 enough space for ReGSI.
    fake_cuda["cuda:1"] = 0
    small = {"id": "small", "config": {"device": "auto", "method_id": "nersi_profile"}}
    large = {"id": "large", "config": {"device": "auto", "method_id": "regsi_real"}}
    small = next(admission.memory_fitting_candidates(tmp_path, [small], resources))
    fake_cuda["cuda:1"] = 49 * 2**30
    large = next(admission.memory_fitting_candidates(tmp_path, [large], resources))
    for item, expected in ((small, "cuda:0"), (large, "cuda:1")):
        with admission.candidate_admission(
            tmp_path,
            item["config"],
            resources,
            item["id"],
            pytest.fail,
            reservation=item["admission_reservation"],
        ) as device:
            assert device == expected


@pytest.mark.parametrize(
    "error",
    [OSError("transient"), run_recovery.ResourceWait("disk"), RuntimeError("out of memory")],
)
def test_retry_keeps_large_reservation_during_backoff_and_cleans_on_success(
    tmp_path, monkeypatch, fake_cuda, error
):
    fake_cuda["cuda:1"] = 70 * 2**30
    resources = {
        "global_admission": True,
        "devices": ["cuda:1"],
        "minimum_free_memory_gib": {"m": 66},
    }
    item = {"id": "large", "config": {"device": "auto", "method_id": "m"}}
    item = next(admission.memory_fitting_candidates(tmp_path, [item], resources))
    reservation = Path(item["admission_reservation"])
    original = json.loads(reservation.read_text())
    sleeps = []

    def sleep(seconds):
        slot = admission.live_admission_slots(reservation.parent)[0]
        assert slot["reservation"] == original["reservation"]
        assert slot["reserved_at_ns"] == original["reserved_at_ns"]
        assert slot["retry_hold_until"] <= admission.time.monotonic() + 300
        assert (
            admission.choose_admission_device(
                ["cuda:1"], {"cuda:1": 70 * 2**30}, [slot], 48 * 2**30
            )
            is None
        )
        sleeps.append(seconds)

    monkeypatch.setattr(run_recovery.time, "sleep", sleep)
    monkeypatch.setattr(run_recovery.torch.cuda, "is_initialized", lambda: False)
    calls = []

    @run_recovery.retry_run
    def execute(repo, experiment, candidate, seed, stage):
        calls.append(1)
        with admission.candidate_admission(
            repo,
            candidate["config"],
            resources,
            candidate["id"],
            pytest.fail,
            reservation=candidate["admission_reservation"],
        ):
            if len(calls) == 1:
                raise error
        return "done"

    assert execute(tmp_path, tmp_path, item, 1, "S1") == "done"
    assert sleeps and len(calls) == 2
    assert not reservation.exists()


def test_retry_reservation_keeps_instantaneous_memory_before_cache_cleanup(
    tmp_path, monkeypatch, fake_cuda
):
    fake_cuda["cuda:1"] = 70 * 2**30
    resources = {
        "global_admission": True,
        "devices": ["cuda:1"],
        "minimum_free_memory_gib": {"m": 66},
    }
    item = {"id": "large", "config": {"device": "auto", "method_id": "m"}}
    item = next(admission.memory_fitting_candidates(tmp_path, [item], resources))
    reservation = Path(item["admission_reservation"])
    with (
        pytest.raises(OSError, match="transient"),
        admission.candidate_admission(
            tmp_path, item["config"], resources, item["id"], pytest.fail, reservation=reservation
        ),
    ):
        monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: int(57.5 * 2**30))
        monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: 58 * 2**30)
        raise OSError("transient")
    slot = json.loads(reservation.read_text())
    assert slot["current_reserved_bytes"] == 0
    assert slot["current_reserved_now_bytes"] == int(57.5 * 2**30)
    assert slot["peak_reserved_bytes"] == 58 * 2**30


def test_retry_hold_expires_without_renewal_on_repeated_resource_wait(
    tmp_path, monkeypatch, fake_cuda
):
    now = [100.0]
    monkeypatch.setattr(admission.time, "monotonic", lambda: now[0])
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"m": 6},
    }
    item = next(
        admission.memory_fitting_candidates(
            tmp_path, [{"id": "a", "config": {"device": "auto", "method_id": "m"}}], resources
        )
    )
    path = Path(item["admission_reservation"])
    for stamp in (100.0, 399.0):
        now[0] = stamp
        with (
            pytest.raises(run_recovery.ResourceWait),
            admission.candidate_admission(
                tmp_path, item["config"], resources, "a", pytest.fail, reservation=path
            ),
        ):
            raise run_recovery.ResourceWait("disk")
        assert json.loads(path.read_text())["retry_hold_until"] == 400.0
    now[0] = 400.0
    assert admission.live_admission_slots(path.parent) == []
    assert not path.exists()


def test_candidate_wait_reports_remaining_memory_and_throttles_each_minute(
    tmp_path, monkeypatch, fake_cuda
):
    now, waits = [0.0], []
    fake_cuda["cuda:0"] = 0
    monkeypatch.setattr(admission.time, "monotonic", lambda: now[0])

    def sleep(seconds):
        now[0] += seconds
        if now[0] >= 125:
            fake_cuda["cuda:0"] = 10 * 2**30

    monkeypatch.setattr(admission.time, "sleep", sleep)
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"m": 6},
    }
    item = {"id": "a", "config": {"device": "auto", "method_id": "m"}}
    result = next(
        admission.memory_fitting_candidates(tmp_path, [item], resources, waiting=waits.append)
    )
    assert result["id"] == "a" and len(waits) == 3
    assert waits[0] == {
        "reason": "no_fitting_candidate",
        "remaining": ["a"],
        "required_bytes": {"a": 6 * 2**30},
        "free_memory_bytes": {"cuda:0": 0},
        "excluded_devices": [],
    }


@pytest.mark.parametrize(
    "error", [RuntimeError("invalid input"), RuntimeError("CUDA error: device-side assert")]
)
def test_fatal_and_process_restart_errors_release_reservation(tmp_path, fake_cuda, error):
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"m": 6},
    }
    item = next(
        admission.memory_fitting_candidates(
            tmp_path, [{"id": "a", "config": {"device": "auto", "method_id": "m"}}], resources
        )
    )
    path = Path(item["admission_reservation"])
    with (
        pytest.raises(RuntimeError),
        admission.candidate_admission(
            tmp_path, item["config"], resources, "a", pytest.fail, reservation=path
        ),
    ):
        raise error
    assert not path.exists()


def test_retry_exhaustion_releases_retained_reservation(tmp_path, monkeypatch, fake_cuda):
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"m": 6},
    }
    item = next(
        admission.memory_fitting_candidates(
            tmp_path, [{"id": "a", "config": {"device": "auto", "method_id": "m"}}], resources
        )
    )
    path = Path(item["admission_reservation"])
    monkeypatch.setattr(run_recovery.time, "sleep", lambda _: None)
    monkeypatch.setattr(run_recovery.torch.cuda, "is_initialized", lambda: False)

    @run_recovery.retry_run
    def execute(repo, experiment, candidate, seed, stage):
        with admission.candidate_admission(
            repo,
            candidate["config"],
            resources,
            "a",
            pytest.fail,
            reservation=candidate["admission_reservation"],
        ):
            raise OSError("unavailable")

    with pytest.raises(RuntimeError, match="automatic retry limit exhausted"):
        execute(tmp_path, tmp_path, item, 1, "S1")
    assert not path.exists()
    assert len(list((tmp_path / "method_runs/m/S1/a/attempts").iterdir())) == 3
