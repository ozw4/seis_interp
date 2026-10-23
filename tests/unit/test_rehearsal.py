"""CPU isolation guards and deterministic rehearsal hook contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from seis_interp.data.experiment_imports import method_config_hash
from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.c3_region_search import select_c3_candidate
from seis_interp.training import rehearsal
from seis_interp.training.exact_resume import ExactResume
from seis_interp.training.run_recovery import ResourceWait, failure_kind


def sealed_rehearsal(root, faults=(), **plan_values):
    folder = root / "study"
    folder.mkdir(parents=True)
    plan = {"faults": list(faults), **plan_values}
    (folder / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "tiny_rehearsal",
                "trace_split": {"evaluation_seeds": [990671, 990672, 990673]},
            }
        )
    )
    (folder / "rehearsal.yaml").write_text(yaml.safe_dump(plan))
    (root / "experiment.json").write_text(
        json.dumps(
            {"rehearsal": True, "rehearsal_plan_sha256": sha256_file(folder / "rehearsal.yaml")}
        )
    )
    return rehearsal.Rehearsal(root)


@pytest.mark.parametrize(
    "seeds",
    [
        [7321, 7322, 7323],
        [7311, 7312, 7313],
        [990000, 990001, 991000],
        [990671] * 3,
        [True, 990672, 990673],
    ],
)
def test_rehearsal_guard_rejects_unsafe_seeds(tmp_path, seeds):
    (tmp_path / "rehearsal.yaml").write_text("faults: []\n")
    with pytest.raises(ValueError, match="seeds"):
        rehearsal.load_rehearsal_plan(
            tmp_path, {"study_id": "tiny_rehearsal", "trace_split": {"evaluation_seeds": seeds}}
        )


@pytest.mark.parametrize(
    "contract,plan,error",
    [
        (
            {"study_id": "real", "trace_split": {"evaluation_seeds": [7321, 7322, 7323]}},
            True,
            "ending",
        ),
        (
            {"study_id": "real", "trace_split": {"evaluation_seeds": [990671, 990672, 990673]}},
            False,
            "reserved",
        ),
        ({"study_id": "real", "import_from": ["runs/example_rehearsal/run1"]}, False, "import"),
        ({"study_id": "tiny_rehearsal"}, False, "requires rehearsal.yaml"),
    ],
)
def test_rehearsal_guard_rejects_wrong_study_and_imports(tmp_path, contract, plan, error):
    if plan:
        (tmp_path / "rehearsal.yaml").write_text("faults: []\n")
    with pytest.raises(ValueError, match=error):
        rehearsal.load_rehearsal_plan(tmp_path, contract)


@pytest.mark.parametrize(
    "plan",
    [
        {"unknown": True},
        {"resume_interval_updates": 0},
        {"watchdog": {"floor_seconds": -1}},
        {"watchdog": {"unknown": 3}},
        {"candidates": {"m": []}},
        {"candidates": {"m": ["a", "a"]}},
        {"faults": [{"point": "bad", "action": "exit"}]},
        {
            "faults": [
                {
                    "point": "report",
                    "action": "sleep",
                    "method": "m",
                    "run": "a",
                    "seconds": float("inf"),
                }
            ]
        },
        {
            "faults": [
                {"point": "report", "action": "exit", "method": "m", "run": "a", "attempt": -1}
            ]
        },
    ],
)
def test_invalid_plan_is_rejected(plan):
    with pytest.raises(ValueError):
        rehearsal.validate_rehearsal_plan(plan)


def test_hook_and_overrides_require_the_sealed_true_flag(tmp_path, monkeypatch):
    hook = sealed_rehearsal(
        tmp_path,
        [{"point": "report", "method": "m", "run": "a", "action": "exit", "step": 1}],
        resume_interval_updates=1,
        candidates={"m": ["a"]},
    )
    (tmp_path / "experiment.json").write_text('{"rehearsal": false}')
    monkeypatch.setattr(rehearsal.os, "_exit", lambda _: pytest.fail("real experiment injected"))
    hook = rehearsal.Rehearsal(tmp_path)
    hook.inject("report", "m", "a", 0, step=1)
    assert hook.plan is None
    assert hook.recovery(tmp_path, "m", "a", 0, print).updates == 2500
    assert hook.watchdog(tmp_path, {"method_id": "m"}, resumable=True).floor == 1800
    assert not (tmp_path / "rehearsal_faults").exists()


def test_hook_rejects_modified_frozen_plan(tmp_path):
    sealed_rehearsal(tmp_path)
    (tmp_path / "study/rehearsal.yaml").write_text("faults: []\n# edited\n")
    with pytest.raises(ValueError, match="hash differs"):
        rehearsal.Rehearsal(tmp_path)


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_barrier_hold_waits_for_every_neural_seed_prediction(tmp_path, monkeypatch, dataset):
    from seis_interp.training.s1_barrier import s1_barrier_ready
    from seis_interp.training.study067_rehearsal_generation import rehearsal_faults

    fault = next(f for f in rehearsal_faults(dataset) if f["action"] == "barrier_hold")
    hook = sealed_rehearsal(tmp_path, [fault])
    methods = ["regsi_real", "ccnet5d_grid", "nersi_profile"]
    seal = json.loads((tmp_path / "experiment.json").read_text())
    (tmp_path / "experiment.json").write_text(
        json.dumps(seal | {"methods": methods + ["drr_grid"]})
    )
    now = [0.0]
    sleeps = []
    workers = tmp_path / "workers"
    workers.mkdir()
    sources = {method: tmp_path for method in methods + ["drr_grid"]}
    for method in sources:
        (workers / f"{method}.json").write_text(
            json.dumps(
                {"status": "running", "restartable": True, "s1_completed": method != "drr_grid"}
            )
        )

    def finalize_next(seconds):
        assert not s1_barrier_ready(tmp_path, sources)
        now[0] += seconds
        sleeps.append(seconds)
        method = methods[len(sleeps) - 1]
        output = tmp_path / "method_runs" / method / "evaluation/seed_990671"
        output.mkdir(parents=True)
        np.save(output / "prediction.npy", np.zeros((1, 1)))
        (output / "manifest.json").write_text(
            json.dumps(
                {"status": "awaiting_barrier", "prediction_finalized_at": "2026-10-22T00:00:01Z"}
            )
        )

    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(rehearsal.time, "sleep", finalize_next)
    hook.inject("pre_method", "drr_grid", "a01_b01", 0)
    assert sleeps == [10, 10, 10]
    event = json.loads((tmp_path / "rehearsal_faults/000.json").read_text())
    assert set(event["seed_predictions"]) == set(methods)
    assert event["released_at"] >= event["fired_at"]
    assert not (tmp_path / "workers/s1_barrier.json").exists()
    (workers / "drr_grid.json").write_text('{"status": "completed", "s1_completed": true}')
    assert s1_barrier_ready(tmp_path, sources)


@pytest.mark.parametrize("incomplete", ["manifest", "status", "timestamp", "prediction"])
def test_barrier_hold_times_out_and_does_not_skip_existing_marker_on_retry(
    tmp_path, monkeypatch, incomplete
):
    fault = {
        "method": "drr_grid",
        "run": "a01_b01",
        "point": "pre_method",
        "action": "barrier_hold",
        "seconds": 1,
    }
    hook = sealed_rehearsal(tmp_path, [fault])
    seal = json.loads((tmp_path / "experiment.json").read_text())
    (tmp_path / "experiment.json").write_text(
        json.dumps(seal | {"methods": ["ccnet5d_grid", "drr_grid"]})
    )
    output = tmp_path / "method_runs/ccnet5d_grid/evaluation/seed_990671"
    output.mkdir(parents=True)
    record = {"status": "awaiting_barrier", "prediction_finalized_at": "2026-10-22T00:00:01Z"}
    if incomplete == "status":
        record["status"] = "running"
    if incomplete == "timestamp":
        record.pop("prediction_finalized_at")
    (output / "manifest.json").write_text("{" if incomplete == "manifest" else json.dumps(record))
    if incomplete != "prediction":
        np.save(output / "prediction.npy", np.zeros((1, 1)))
    now = [0.0]
    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        rehearsal.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    for attempt in (0, 1):
        with pytest.raises(TimeoutError, match="awaiting seed predictions: ccnet5d_grid"):
            hook.inject("pre_method", "drr_grid", "a01_b01", attempt)
        assert "released_at" not in json.loads((tmp_path / "rehearsal_faults/000.json").read_text())
    (output / "manifest.json").write_text(
        json.dumps(
            {"status": "awaiting_barrier", "prediction_finalized_at": "2026-10-22T00:00:01Z"}
        )
    )
    np.save(output / "prediction.npy", np.zeros((1, 1)))
    hook.inject("pre_method", "drr_grid", "a01_b01", 2)
    assert "released_at" in json.loads((tmp_path / "rehearsal_faults/000.json").read_text())


@pytest.mark.parametrize(
    "point,method,seconds",
    [("report", "drr_grid", 1), ("pre_method", "regsi_real", 1), ("pre_method", "drr_grid", 0)],
)
def test_barrier_hold_plan_requires_drr_pre_method_and_bounded_timeout(point, method, seconds):
    with pytest.raises(ValueError, match="barrier_hold|seconds"):
        rehearsal.validate_rehearsal_plan(
            {
                "faults": [
                    {
                        "method": method,
                        "run": "a01_b01",
                        "point": point,
                        "action": "barrier_hold",
                        "seconds": seconds,
                    }
                ]
            }
        )


def test_subset_is_applied_after_priority_and_default_selection_stays_25():
    candidates = [{"id": name} for name in ("a", "b", "c")]
    contract = {"method_execution": {"m": {"candidate_priority": ["c", "b"]}}}
    chosen = rehearsal.rehearsal_candidates(
        candidates, contract, "m", {"candidates": {"m": ["a", "b"]}}
    )
    assert [item["id"] for item in chosen] == ["b", "a"]
    records = [
        {"id": "b", "status": "diverged"},
        {"id": "a", "status": "completed", "validation_score": 1.0},
    ]
    with pytest.raises(ValueError, match="25"):
        select_c3_candidate(chosen, records)
    assert select_c3_candidate(chosen, records, expected=2)["id"] == "a"
    with pytest.raises(ValueError, match="unknown rehearsal"):
        rehearsal.rehearsal_candidates(
            candidates, contract, "m", {"candidates": {"m": ["missing"]}}
        )


@pytest.mark.parametrize(
    "action,error,kind",
    [
        ("oom", torch.cuda.OutOfMemoryError, "infrastructure"),
        ("infra", OSError, "infrastructure"),
        ("resource_wait", ResourceWait, "infrastructure"),
        ("nonfinite", FloatingPointError, "diverged"),
    ],
)
def test_fault_filters_attempt_step_phase_and_fires_once(tmp_path, action, error, kind):
    hook = sealed_rehearsal(
        tmp_path,
        [
            {
                "point": "report",
                "method": "m",
                "run": "a",
                "action": action,
                "step": 2,
                "phase": "training",
            }
        ],
    )
    for args in [
        ("report", "m", "a", 1, 2, "training"),
        ("report", "m", "a", 0, 1, "training"),
        ("report", "m", "b", 0, 2, "training"),
        ("report", "m", "a", 0, 2, "other"),
    ]:
        hook.inject(*args)
    assert not (tmp_path / "rehearsal_faults").exists()
    with pytest.raises(error) as caught:
        hook.inject("report", "m", "a", 0, 2, "training")
    assert failure_kind(caught.value) == kind
    rehearsal.Rehearsal(tmp_path).inject("report", "m", "a", 0, 2, "training")
    assert len(list((tmp_path / "rehearsal_faults").glob("*.json"))) == 1


def test_nan_prediction_sleep_and_exit_actions(tmp_path, monkeypatch):
    faults = [
        {"point": "after_prediction", "method": "m", "run": "a", "action": action, **extra}
        for action, extra in [("nan_prediction", {}), ("sleep", {"seconds": 3}), ("exit", {})]
    ]
    hook = sealed_rehearsal(tmp_path, faults)
    sleeps, exits = [], []
    monkeypatch.setattr(rehearsal.time, "sleep", sleeps.append)
    monkeypatch.setattr(rehearsal.os, "_exit", exits.append)
    result = SimpleNamespace(prediction=np.ones((2, 3)))
    hook.inject("after_prediction", "m", "a", 0, result=result)
    assert np.isnan(result.prediction[0, 0])
    assert sleeps == [3] and exits == [137]


def test_sticky_cuda_action_is_device_assert_then_synchronize_without_a_real_gpu(
    tmp_path, monkeypatch
):
    hook = sealed_rehearsal(
        tmp_path, [{"point": "report", "method": "m", "run": "a", "action": "sticky_cuda"}]
    )
    calls = []

    class FakeCUDA:
        def __setitem__(self, index, value):
            calls.append((index, value))

    monkeypatch.setattr(torch, "zeros", lambda count, device: FakeCUDA())
    monkeypatch.setattr(torch, "tensor", lambda values, device: values)

    def synchronize():
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with pytest.raises(RuntimeError, match="device-side"):
        hook.inject("report", "m", "a", 0)
    assert calls == [([2], 1)]


def test_checkpoint_hook_observes_replaced_file_and_resume_digest(tmp_path):
    logs, hooks = [], []
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = tmp_path / "resume.pt"

    def checkpoint(step, phase):
        saved = torch.load(path, weights_only=False)
        hooks.append((saved["step"], saved["phase"], step, phase))

    recovery = ExactResume(path, updates=2, reporter=logs.append, checkpoint=checkpoint)
    recovery.save(2, model, optimizer, {"history": [{"step": 2}]})
    assert hooks == [(2, "training", 2, "training")]
    state = recovery.load(model, optimizer)
    assert state["step"] == 2
    assert "resumed step=2 phase=training" in logs[0]
    fields = dict(item.split("=", 1) for item in logs[0].split() if "=" in item)
    assert fields["state_sha256_saved"] == fields["state_sha256_loaded"]
    assert fields["adamw_step_device"] == "cpu"


def test_execution_watchdog_keys_do_not_break_import_identity():
    base = {
        "method_id": "regsi_real",
        "execution": {"cpu_threads": 1},
        "training": {"max_steps": 50000},
    }
    altered = base | {
        "execution": {
            "expected_report_seconds": 60,
            "watchdog_floor_seconds": 300,
            "wall_clock_limit_seconds": 3600,
        }
    }
    assert method_config_hash(base) == method_config_hash(altered)


def test_rehearsal_watchdog_overrides_do_not_enable_classical_watchdogs(tmp_path):
    hook = sealed_rehearsal(
        tmp_path,
        watchdog={"floor_seconds": 300, "expected_report_seconds": 60},
        resume_interval_updates=100,
    )
    assert hook.recovery(tmp_path, "m", "a", 0, print).updates == 100
    dog = hook.watchdog(tmp_path, {"method_id": "regsi_real"}, resumable=True)
    assert dog.threshold == 300
    assert not hook.watchdog(tmp_path, {"method_id": "drr_grid"}, resumable=True).enabled


@pytest.mark.parametrize("field", ["search_seed", "model_seed", "mask_seed"])
def test_nonrehearsal_reserved_seed_guard_also_covers_non_evaluation_seeds(tmp_path, field):
    contract = {"study_id": "real", "trace_split": {"evaluation_seeds": [7321, 7322, 7323]}}
    contract[field] = 990671
    with pytest.raises(ValueError, match="reserved rehearsal seeds"):
        rehearsal.load_rehearsal_plan(tmp_path, contract)
