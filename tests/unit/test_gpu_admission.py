import fcntl
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from seis_interp.training import gpu_admission as admission
from seis_interp.training.worker_processes import process_is_alive, process_start_time
from tests.fixtures.worker_locks import orphaned_lock_holder


def test_regrowth_is_reserved_once_even_after_current_reserved_drops_below_peak():
    slots = [
        {
            "device": "cuda:0",
            "required_bytes": 60,
            "current_reserved_bytes": 10,
            "peak_reserved_bytes": 60,
        }
    ]
    free = {"cuda:0": 100, "cuda:1": 70}
    assert admission.choose_admission_device(["cuda:0", "cuda:1"], free, slots, 60) == "cuda:1"
    assert admission.choose_admission_device(["cuda:0"], free, slots, 60) is None
    slots[0]["current_reserved_bytes"] = 60
    assert admission.choose_admission_device(["cuda:0"], free, slots, 100) == "cuda:0"
    slots[0]["current_reserved_bytes"] = 10
    assert admission.choose_admission_device(["cuda:0"], free, slots, 60) is None
    slots[0]["current_reserved_bytes"] = 90
    assert admission.choose_admission_device(["cuda:0"], free, slots, 101) is None


def test_conservative_reservation_minimum_expires_reports_after_thirty_seconds():
    reports = deque()
    assert admission._recent_reserved_minimum(reports, 100, 4) == 4
    assert admission._recent_reserved_minimum(reports, 129, 10) == 4
    assert admission._recent_reserved_minimum(reports, 130, 10) == 4
    assert admission._recent_reserved_minimum(reports, 131, 10) == 10
    assert admission._recent_reserved_minimum(reports, 132, 2) == 2
    assert admission._recent_reserved_minimum(reports, 162, 10) == 2
    assert admission._recent_reserved_minimum(reports, 163, 10) == 10


def test_cpu_and_legacy_admission_never_query_cuda_or_write_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(admission.torch.cuda, "mem_get_info", lambda _: pytest.fail("CUDA queried"))
    for device, resources in [("cpu", {"global_admission": True}), ("cuda:0", None)]:
        with admission.candidate_admission(
            tmp_path, {"device": device}, resources, "a01_b01", pytest.fail
        ) as result:
            assert result == device
    assert not (tmp_path / "runs").exists()


def test_startup_lock_timeout_keeps_pending_reservation_and_cleans_up(tmp_path, monkeypatch):
    fractions = []
    monkeypatch.setattr(
        admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=120 * 2**30),
    )
    monkeypatch.setattr(
        admission.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: fractions.append((fraction, device)),
    )
    monkeypatch.setattr(admission.torch.cuda, "mem_get_info", lambda _: (100 * 2**30, 120 * 2**30))
    monkeypatch.setattr(admission.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: 2**30)
    monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: 2**30)
    with (
        admission.gpu_admission_slot(
            tmp_path, ["cuda:0"], 10, pytest.fail, timeout=0.02, poll=0.005
        ),
        (tmp_path / "admission.lock").open("a") as other,
    ):
        with pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 2
        while True:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        slot = admission.live_admission_slots(tmp_path)[0]
        assert slot["pid"] == os.getpid()
        assert slot["process_start_time"] == admission.process_start_time(os.getpid())
        assert fractions == [(10 / 120, "cuda:0")]
        assert slot["peak_reserved_bytes"] == 2**30
        assert slot["current_reserved_bytes"] == 2**30
        assert slot["current_reserved_now_bytes"] == 2**30
        assert (
            admission.choose_admission_device(["cuda:0"], {"cuda:0": 10 * 2**30}, [slot], 2 * 2**30)
            is None
        )
    assert not list(tmp_path.glob("slot_*.json"))


def test_stale_slot_is_removed_without_signalling_any_process(tmp_path):
    path = tmp_path / "slot_dead.json"
    path.write_text(json.dumps({"pid": 999999999}))
    assert admission.live_admission_slots(tmp_path) == []
    assert not path.exists()


def test_fork_child_cannot_retain_startup_lock_or_dead_parents_slot(tmp_path):
    with (
        orphaned_lock_holder(tmp_path, "admitted", admission=True) as holder,
        (tmp_path / "admission.lock").open("a") as lock,
    ):
        # Closing the child's inherited descriptor must not unlock its live parent.
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        slots = admission.live_admission_slots(tmp_path)
        assert len(slots) == 1 and slots[0]["pid"] == holder.process.pid
        holder.exit_path.touch()
        assert holder.process.wait(timeout=5) == 137
        assert process_is_alive(holder.child_pid, process_start_time(holder.child_pid))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert admission.live_admission_slots(tmp_path) == []
        assert not list(tmp_path.glob("slot_*.json"))


@pytest.mark.parametrize("start_offset,state", [(0, "Z"), (1, "R")])
def test_zombie_and_reused_pid_slots_are_removed(tmp_path, monkeypatch, start_offset, state):
    from seis_interp.training import worker_processes

    monkeypatch.setattr(worker_processes, "process_stat", lambda _: (state, 100))
    path = tmp_path / "slot_stale.json"
    path.write_text(json.dumps({"pid": os.getpid(), "process_start_time": 100 + start_offset}))
    assert admission.live_admission_slots(tmp_path) == []
    assert not path.exists()


@pytest.mark.parametrize("required,allowance", [(92, 2), (153, 2), (90, 4)])
def test_impossible_budget_fails_before_admission_directory_or_memory_wait(
    tmp_path, monkeypatch, required, allowance
):
    monkeypatch.setattr(
        admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=93 * 2**30),
    )
    monkeypatch.setattr(admission.torch.cuda, "mem_get_info", lambda _: pytest.fail("wait entered"))
    with (
        pytest.raises(ValueError, match="exceeds largest allowed device"),
        admission.gpu_admission_slot(
            tmp_path / "slots",
            ["cuda:0"],
            required,
            pytest.fail,
            context_allowance_gib=allowance,
        ),
    ):
        pytest.fail("admitted")
    assert not (tmp_path / "slots").exists()


def test_prepare_capacity_uses_allowed_devices_and_explicit_margin_without_cuda(monkeypatch):
    monkeypatch.setattr(
        admission.torch.cuda, "get_device_properties", lambda _: pytest.fail("CUDA queried")
    )
    item = {"id": "a01_b01", "config": {"device": "cuda:0", "method_id": "ccnet5d_grid"}}
    resources = {
        "global_admission": True,
        "devices": ["cuda:0", "cuda:1"],
        "minimum_free_memory_gib": {"ccnet5d_grid": 9},
        "admission_margin_gib": 2,
        "device_total_memory_gib": {"cuda:0": 10, "cuda:1": 20},
    }
    with pytest.raises(ValueError, match="11.000 GiB exceeds.*8.000"):
        admission.prepare_admission_capacity(resources, [item])
    item["config"]["device"] = "auto"
    admission.prepare_admission_capacity(resources, [item])
    resources["minimum_free_memory_gib"]["ccnet5d_grid"] = None
    with pytest.raises(ValueError, match="measure with the probe first"):
        admission.prepare_admission_capacity(resources, [item])


def test_prepare_capacity_can_query_nvidia_smi_without_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []

    def totals(command, **kwargs):
        calls.append(command)
        return "10240\n"

    monkeypatch.setattr(admission.subprocess, "check_output", totals)
    item = {"id": "a01_b01", "config": {"device": "auto", "method_id": "ccnet5d_grid"}}
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "minimum_free_memory_gib": {"ccnet5d_grid": 11},
    }
    with pytest.raises(ValueError, match="exceeds largest"):
        admission.prepare_admission_capacity(resources, [item])
    assert calls[0][0] == "nvidia-smi"


@pytest.mark.parametrize("allowance", [None, 4])
def test_prepare_rejects_near_total_budget_with_context_allowance(allowance):
    item = {"id": "a01_b01", "config": {"device": "auto", "method_id": "ccnet5d_grid"}}
    resources = {
        "global_admission": True,
        "devices": ["cuda:0"],
        "device_total_memory_gib": {"cuda:0": 93},
        "minimum_free_memory_gib": {"ccnet5d_grid": 92},
    }
    if allowance is not None:
        resources["context_allowance_gib"] = allowance
    with pytest.raises(ValueError, match="context allowance"):
        admission.prepare_admission_capacity(resources, [item])
    resources["minimum_free_memory_gib"]["ccnet5d_grid"] = 93 - (allowance or 2)
    admission.prepare_admission_capacity(resources, [item])


def test_monitor_publishes_current_reserved_after_cache_shrinks(tmp_path, monkeypatch):
    reserved = [10 * 2**30]
    monkeypatch.setattr(
        admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=20 * 2**30),
    )
    monkeypatch.setattr(admission.torch.cuda, "mem_get_info", lambda _: (18 * 2**30, 20 * 2**30))
    monkeypatch.setattr(admission.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "set_per_process_memory_fraction", lambda *_: None)
    monkeypatch.setattr(admission.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: 10 * 2**30)
    monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: reserved[0])
    with admission.gpu_admission_slot(tmp_path, ["cuda:0"], 10, pytest.fail, poll=0.005):
        reserved[0] = 4 * 2**30
        deadline = time.monotonic() + 2
        while True:
            slot = admission.live_admission_slots(tmp_path)[0]
            if slot["current_reserved_bytes"] == reserved[0]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert slot["peak_reserved_bytes"] == 10 * 2**30
        assert slot["current_reserved_now_bytes"] == 4 * 2**30
        assert (
            admission.choose_admission_device(["cuda:0"], {"cuda:0": 8 * 2**30}, [slot], 3 * 2**30)
            is None
        )


def test_monitor_reserves_regrowth_before_cache_release_is_reported(tmp_path, monkeypatch):
    reserved = [4 * 2**30]
    monkeypatch.setattr(
        admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=20 * 2**30),
    )
    monkeypatch.setattr(admission.torch.cuda, "mem_get_info", lambda _: (18 * 2**30, 20 * 2**30))
    monkeypatch.setattr(admission.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "set_per_process_memory_fraction", lambda *_: None)
    monkeypatch.setattr(admission.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: reserved[0])
    monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: reserved[0])
    with admission.gpu_admission_slot(tmp_path, ["cuda:0"], 10, pytest.fail, poll=0.005):
        reserved[0] = 10 * 2**30
        deadline = time.monotonic() + 2
        while True:
            slot = admission.live_admission_slots(tmp_path)[0]
            if slot["peak_reserved_bytes"] == reserved[0]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert slot["current_reserved_now_bytes"] == 10 * 2**30
        # Use the published slot from before the cap releases cached memory.
        reserved[0] = 4 * 2**30
        assert slot["current_reserved_bytes"] == reserved[0]
        assert (
            admission.choose_admission_device(["cuda:0"], {"cuda:0": 8 * 2**30}, [slot], 3 * 2**30)
            is None
        )


@pytest.mark.parametrize(
    "stat,excluded", [(None, False), (("Z", 10), False), (("R", 11), False), (("S", 10), True)]
)
def test_device_exclusion_checks_liveness_zombies_and_start_time(monkeypatch, stat, excluded):
    from seis_interp.training import worker_processes

    monkeypatch.setattr(worker_processes, "process_stat", lambda _: stat)
    rules = {"cuda:0": [{"pid": 123, "process_start_time": 10}]}
    assert admission.excluded_admission_devices(rules) == ({"cuda:0"} if excluded else set())


def test_admission_waits_until_all_pinned_processes_exit_without_querying_excluded_gpu(
    tmp_path, monkeypatch
):
    from seis_interp.training import worker_processes

    live = {123, 456}
    monkeypatch.setattr(
        worker_processes, "process_stat", lambda pid: ("S", 10) if pid in live else None
    )
    monkeypatch.setattr(
        admission.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=20 * 2**30),
    )
    monkeypatch.setattr(
        admission.torch.cuda,
        "mem_get_info",
        lambda _: (18 * 2**30, 20 * 2**30) if not live else pytest.fail("excluded GPU queried"),
    )
    monkeypatch.setattr(admission.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "set_per_process_memory_fraction", lambda *_: None)
    monkeypatch.setattr(admission.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(admission.torch.cuda, "max_memory_reserved", lambda _: 0)
    monkeypatch.setattr(admission.torch.cuda, "memory_reserved", lambda _: 0)
    waits = []

    def waiting(details):
        waits.append(details)
        live.remove(min(live))

    rules = {"cuda:0": [{"pid": pid, "process_start_time": 10} for pid in sorted(live)]}
    with admission.gpu_admission_slot(
        tmp_path,
        ["cuda:0"],
        10,
        waiting,
        poll=0.001,
        excluded_devices_while_pids_alive=rules,
    ) as device:
        assert device == "cuda:0"
    assert len(waits) == 2
    assert all(wait["excluded_devices"] == ["cuda:0"] for wait in waits)


def test_candidate_forwards_configured_capacity_and_process_exclusions(tmp_path, monkeypatch):
    rules = {"cuda:0": [{"pid": 123, "process_start_time": 10}]}
    resources = {
        "global_admission": True,
        "devices": ["cuda:0", "cuda:1"],
        "minimum_free_memory_gib": {"ccnet5d_grid": 10},
        "context_allowance_gib": 4,
        "excluded_devices_while_pids_alive": rules,
    }

    @contextmanager
    def slot(directory, devices, minimum, waiting, **kwargs):
        assert kwargs["context_allowance_gib"] == 4
        assert kwargs["excluded_devices_while_pids_alive"] == rules
        yield "cuda:1"

    monkeypatch.setattr(admission, "gpu_admission_slot", slot)
    with admission.candidate_admission(
        tmp_path, {"device": "auto", "method_id": "ccnet5d_grid"}, resources, "a01_b01", pytest.fail
    ) as device:
        assert device == "cuda:1"


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_common_grid_configs_pin_legacy_worker_identities_and_context_allowance(dataset):
    path = (
        Path(__file__).resolve().parents[2]
        / "studies/study_067_forge_c3_common_grid"
        / dataset
        / "config.yaml"
    )
    resources = yaml.safe_load(path.read_text())["cuda_execution"]
    assert resources["context_allowance_gib"] == 2
    assert resources["excluded_devices_while_pids_alive"]["cuda:0"] == [
        {"pid": 3958956, "process_start_time": 1111834778},
        {"pid": 381291, "process_start_time": 1113638455},
    ]
