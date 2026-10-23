import fcntl

import pytest

from seis_interp.training import cuda_run_slot as slots


def test_cpu_run_does_not_query_cuda_or_create_a_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(slots.torch.cuda, "mem_get_info", lambda _: pytest.fail("CUDA queried"))
    with slots.cuda_run_slot(tmp_path / "workers", "cpu", 40, pytest.fail):
        assert not (tmp_path / "workers").exists()


def test_insufficient_memory_releases_lock_before_waiting(tmp_path, monkeypatch):
    available = iter([2**30, 5 * 2**30])
    monkeypatch.setattr(slots.torch.cuda, "mem_get_info", lambda _: (next(available), 10 * 2**30))
    waits = []

    def sleep(seconds):
        assert seconds == 10
        with (tmp_path / "cuda_0.lock").open("a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    monkeypatch.setattr(slots.time, "sleep", sleep)
    with (
        slots.cuda_run_slot(tmp_path, "cuda:0", 4, waits.append),
        (tmp_path / "cuda_0.lock").open("a") as other,
        pytest.raises(BlockingIOError),
    ):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert waits == [
        {
            "device": "cuda:0",
            "reason": "insufficient_free_memory",
            "free_memory_bytes": 2**30,
            "required_free_memory_bytes": 4 * 2**30,
        }
    ]
    with (tmp_path / "cuda_0.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_device_lock_released_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(slots.torch.cuda, "mem_get_info", lambda _: (10 * 2**30, 10 * 2**30))
    with (
        pytest.raises(RuntimeError, match="training failed"),
        slots.cuda_run_slot(tmp_path, "cuda:1", 4, pytest.fail),
    ):
        raise RuntimeError("training failed")
    with (tmp_path / "cuda_1.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_busy_device_waits_before_querying_memory(tmp_path, monkeypatch):
    waits = []
    with (tmp_path / "cuda_0.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)

        def waiting(details):
            waits.append(details)
            fcntl.flock(other, fcntl.LOCK_UN)

        monkeypatch.setattr(slots.torch.cuda, "mem_get_info", lambda _: (5 * 2**30, 10 * 2**30))
        with slots.cuda_run_slot(tmp_path, "cuda:0", 4, waiting):
            pass
    assert waits == [{"device": "cuda:0", "reason": "another_experiment_run"}]
