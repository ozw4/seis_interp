"""Acceptance checks must accept structural evidence and reject broken evidence."""

import json
from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.evaluation import study067_rehearsal_checks as checks
from seis_interp.training.gpu_admission import _recent_reserved_minimum


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_events(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def seal(experiment, faults, dataset):
    folder = experiment / "study"
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "fixture_rehearsal",
                "trace_split": {"evaluation_seeds": [990671, 990672, 990673]},
            }
        )
    )
    (folder / "rehearsal.yaml").write_text(yaml.safe_dump({"faults": faults}))
    write_json(
        experiment / "experiment.json",
        {
            "rehearsal": True,
            "rehearsal_plan_sha256": sha256_file(folder / "rehearsal.yaml"),
            "methods": ["ccnet5d_grid"],
            "dataset_id": dataset,
        },
    )


def gpu_record(method, run, device, stage="S1"):
    return {
        "method_id": method,
        "id": run,
        "stage": stage,
        "trace_seed": 990671,
        "status": "completed",
        "started_at": "2026-10-22T00:00:00Z",
        "ended_at": "2026-10-22T00:00:10Z",
        "cuda_admission": {"device": device, "required_bytes": 20 * 2**30},
        "external_gpu_utilization_at_admission": [],
        "diagnostics": {
            "cuda_reserved_after_warmup_bytes": 4,
            "cuda_reserved_before_prediction_bytes": 5,
            "cuda_reserved_after_prediction_bytes": 5,
            "cuda_max_reserved_after_warmup_bytes": 6,
            "cuda_max_reserved_before_prediction_bytes": 6,
            "cuda_max_reserved_after_prediction_bytes": 6,
        },
    }


@pytest.fixture
def gpu_evidence(tmp_path):
    c3, forge = tmp_path / "c3", tmp_path / "forge"
    write_json(
        c3 / "method_runs/regsi_real/S1/a01_b05/manifest.json",
        gpu_record("regsi_real", "a01_b05", "cuda:0"),
    )
    for name, device in (("a05_b02", "cuda:0"), ("a05_b05", "cuda:1")):
        write_json(
            forge / f"method_runs/ccnet5d_grid/S1/{name}/manifest.json",
            gpu_record("ccnet5d_grid", name, device),
        )
    for seed, device in ((990671, "cuda:0"), (990672, "cuda:1")):
        record = gpu_record("ccnet5d_grid", "a05_b05", device, "evaluation")
        prediction = {
            key: record[key]
            for key in (
                "started_at",
                "ended_at",
                "cuda_admission",
                "external_gpu_utilization_at_admission",
            )
        }
        record.update(
            trace_seed=seed,
            cuda_admission={},
            prediction_execution=prediction,
            started_at="2026-10-22T00:00:20Z",
            ended_at="2026-10-22T00:00:30Z",
        )
        write_json(forge / f"method_runs/ccnet5d_grid/evaluation/seed_{seed}/manifest.json", record)
    for experiment in (c3, forge):
        write_events(
            experiment / "workers/rehearsal_events.jsonl",
            [{"last_admission": {"device": "cuda:0"}, "child_processes": [{"pid": 1}, {"pid": 2}]}],
        )
    sample = {
        "gpus": [
            {
                "device": "cuda:0",
                "uuid": "GPU-a",
                "total_bytes": 100 * 2**30,
                "used_bytes": 10 * 2**30,
            }
        ],
        "slots_before": [],
        "slots_after": [
            {
                "pid": 1,
                "slot_file": "slot_own.json",
                "device": "cuda:0",
                "required_bytes": 20 * 2**30,
                "current_reserved_bytes": 10 * 2**30,
                "current_reserved_now_bytes": 10 * 2**30,
            }
        ],
        "applications": [{"pid": 1, "uuid": "GPU-a", "used_bytes": 10 * 2**30}],
        "experiments": {str(c3): {"states": {"regsi_real": {"pid": 1}}}},
    }
    return [c3, forge], [sample]


def test_admission_checker_uses_preserved_prediction_intervals(gpu_evidence):
    experiments, samples = gpu_evidence
    assert checks._check_admission(experiments, samples * 2)
    path = experiments[1] / "method_runs/ccnet5d_grid/evaluation/seed_990672/manifest.json"
    record = json.loads(path.read_text())
    record.pop("prediction_execution")
    write_json(path, record)
    with pytest.raises(ValueError, match="seed children did not overlap"):
        checks._check_admission(experiments, samples * 2)


@pytest.mark.parametrize(
    "broken", [None, "missing_method", "late_seed", "late_release", "missing_held_method"]
)
def test_barrier_checker_requires_neural_predictions_and_hold_release_evidence(tmp_path, broken):
    fault = {
        "method": "drr_grid",
        "run": "a01_b01",
        "point": "pre_method",
        "action": "barrier_hold",
        "seconds": 21600,
    }
    seal(tmp_path, [fault], "seg_c3_na")
    if broken == "missing_method":
        path = tmp_path / "experiment.json"
        write_json(path, json.loads(path.read_text()) | {"methods": ["ccnet5d_grid", "regsi_real"]})
    write_json(tmp_path / "workers/s1_barrier.json", {"passed_at": "2026-10-22T00:01:00Z"})
    write_json(
        tmp_path / "method_runs/ccnet5d_grid/S1/a05_b02/manifest.json",
        gpu_record("ccnet5d_grid", "a05_b02", "cuda:0") | {"test_scoring_calls": 0},
    )
    finalized = "2026-10-22T00:01:01Z" if broken == "late_seed" else "2026-10-22T00:00:20Z"
    write_json(
        tmp_path / "method_runs/ccnet5d_grid/evaluation/seed_990671/manifest.json",
        {"stage": "evaluation", "method_id": "ccnet5d_grid", "prediction_finalized_at": finalized},
    )
    write_json(
        tmp_path / "rehearsal_faults/000.json",
        {
            "released_at": "2026-10-22T00:01:00Z"
            if broken == "late_release"
            else "2026-10-22T00:00:30Z",
            "seed_predictions": {}
            if broken == "missing_held_method"
            else {"ccnet5d_grid": finalized},
        },
    )
    write_events(tmp_path / "workers/rehearsal_events.jsonl", [])
    sample = {
        "monotonic": 100,
        "clock_ticks": 100,
        "experiments": {
            str(tmp_path): {
                "barrier_sha256": sha256_file(tmp_path / "workers/s1_barrier.json"),
                "states": {"ccnet5d_grid": {"pid": 1, "phase": "waiting_for_all_S1"}},
                "processes": [{"pid": 1, "start_ticks": 10, "cpu_ticks": 100}],
            }
        },
    }
    samples = [sample, sample | {"monotonic": 110}]
    if broken:
        with pytest.raises(ValueError, match="pre-barrier neural"):
            checks._check_barrier([tmp_path], samples)
    else:
        assert checks._check_barrier([tmp_path], samples)


def test_admission_checker_rejects_overbooked_memory_and_unexpected_oom(gpu_evidence):
    experiments, samples = gpu_evidence
    samples[0]["slots_after"][0]["required_bytes"] = 99 * 2**30
    with pytest.raises(ValueError, match="exceeds total"):
        checks._check_admission(experiments, samples * 2)
    samples[0]["slots_after"][0]["required_bytes"] = 20 * 2**30
    path = experiments[0] / "method_runs/regsi_real/S1/a01_b05/manifest.json"
    record = json.loads(path.read_text()) | {"failure_reason": "unplanned out of memory"}
    write_json(path, record)
    with pytest.raises(ValueError, match="unexpected OOM"):
        checks._check_admission(experiments, samples * 2)


def test_warmup_checker_rejects_growth_and_missing_diagnostics(gpu_evidence):
    experiments, _ = gpu_evidence
    assert checks._check_warmup(experiments)
    path = experiments[0] / "method_runs/regsi_real/S1/a01_b05/manifest.json"
    record = json.loads(path.read_text())
    record["diagnostics"]["cuda_reserved_after_prediction_bytes"] = 6
    write_json(path, record)
    with pytest.raises(ValueError, match="more than 2%"):
        checks._check_warmup(experiments)


def test_restart_checker_rejects_restarted_siblings(tmp_path):
    fault = {"method": "m", "run": "a", "point": "report", "action": "exit"}
    events = [
        {"worker": "m", "method_id": "m", "pid": 1},
        {"worker": "m", "method_id": "m", "pid": 2},
    ]
    write_events(tmp_path / "workers/rehearsal_events.jsonl", events)
    write_events(
        tmp_path / "workers/rehearsal_restarts.jsonl",
        [{"worker": "m", "run": ["S1", "a", 0, 0], "retries": 1, "previous_pid": 1}],
    )
    assert checks._check_restart_evidence(tmp_path, {"faults": [fault]})
    events += [
        {"worker": "sibling", "method_id": "sibling", "pid": 3},
        {"worker": "sibling", "method_id": "sibling", "pid": 4},
    ]
    write_events(tmp_path / "workers/rehearsal_events.jsonl", events)
    with pytest.raises(ValueError, match="without a recorded restart"):
        checks._check_restart_evidence(tmp_path, {"faults": [fault]})


@pytest.mark.parametrize("relogged_loss", [0.5001, 0.8])
def test_resume_checker_compares_digests_steps_reports_and_twins(tmp_path, relogged_loss):
    experiments, twins, determinism_twins = [], [], []
    for name, dataset, candidate in (
        ("c3", "seg_c3_na", "a05_b02"),
        ("forge", "forge_2017", "a05_b05"),
    ):
        experiment, twin = tmp_path / name, tmp_path / f"{name}_twin"
        faults = [
            {
                "method": "ccnet5d_grid",
                "run": candidate,
                "point": "report",
                "action": "oom",
                "step": 250,
            }
        ]
        seal(experiment, faults, dataset)
        seal(twin, [], dataset)
        output = experiment / "method_runs/ccnet5d_grid/S1" / candidate
        output.mkdir(parents=True)
        pd.DataFrame({"step": range(1, 301), "loss": [0.5] * 300}).to_csv(
            output / "history.csv", index=False
        )
        np.save(output / "prediction.npy", np.ones((2, 3)))
        write_json(
            output / "manifest.json",
            {
                "attempt": 1,
                "pid": 123,
                "status": "completed",
                "metrics": {
                    "validation": {"masked_trace_relative_mse": 0.5, "mean_trace_snr_db": 3.0}
                },
            },
        )
        write_json(output / "attempts/1/manifest.json", {"pid": 123})
        (output / "attempts/1/execution.log").write_text("step 250/300: loss=0.5\n")
        (output / "execution.log").write_text(
            "resumed step=200 phase=training state_sha256_saved=abc "
            "state_sha256_loaded=abc adamw_step_device=cpu\n"
            + f"step 250/300: loss={relogged_loss}\n"
        )
        write_events(
            experiment / "workers/rehearsal_events.jsonl",
            [
                {
                    "worker": "ccnet5d_grid",
                    "method_id": "ccnet5d_grid",
                    "pid": 123,
                    "status": "completed",
                }
            ],
        )
        twin_output = twin / "method_runs/ccnet5d_grid/S1" / candidate
        twin_output.mkdir(parents=True)
        for path in (output, twin_output):
            (path / "config.resolved.yaml").write_text(
                yaml.safe_dump({"execution": {"cudnn_benchmark": True}})
            )
        for filename in ("history.csv", "prediction.npy", "manifest.json"):
            (twin_output / filename).write_bytes((output / filename).read_bytes())
        for suffix in ("a", "b"):
            deterministic = tmp_path / f"{name}_determinism_{suffix}"
            seal(deterministic, [], dataset)
            det_output = deterministic / "method_runs/ccnet5d_grid/S1" / candidate
            det_output.mkdir(parents=True)
            (det_output / "config.resolved.yaml").write_text(
                "execution: {cudnn_benchmark: false}\n"
            )
            for filename in ("history.csv", "prediction.npy"):
                (det_output / filename).write_bytes((output / filename).read_bytes())
            determinism_twins.append(deterministic)
        experiments.append(experiment)
        twins.append(twin)
    assert checks._check_resume(experiments, twins, determinism_twins)
    with pytest.raises(ValueError, match="four separate benchmark-off"):
        checks._check_resume(experiments, twins)
    with pytest.raises(ValueError, match="four separate benchmark-off"):
        checks._check_resume(experiments, twins, determinism_twins[:2] * 2)
    det = determinism_twins[0] / "method_runs/ccnet5d_grid/S1/a05_b02"
    (det / "config.resolved.yaml").write_text("execution: {cudnn_benchmark: true}\n")
    with pytest.raises(ValueError, match="must disable benchmark"):
        checks._check_resume(experiments, twins, determinism_twins)
    (det / "config.resolved.yaml").write_text("execution: {cudnn_benchmark: false}\n")
    log = experiments[0] / "method_runs/ccnet5d_grid/S1/a05_b02/execution.log"
    log.write_text(log.read_text().replace("loaded=abc", "loaded=bad"))
    with pytest.raises(ValueError, match="state-exact resume"):
        checks._check_resume(experiments, twins, determinism_twins)


def test_import_checker_compares_checkpoint_and_metric_without_disclosing_values(tmp_path):
    source, replay = tmp_path / "original", tmp_path / "replay"
    metrics = {"validation": {"masked_trace_relative_mse": 0.123456789}}
    write_json(source / "manifest.json", {"metrics": metrics})
    (source / "final.pt").write_bytes(b"opaque checkpoint bytes")
    output = replay / "method_runs/m/S1/a"
    write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "imported_from": {"source_path": str(source)},
            "cuda_admission": {},
            "metrics": metrics,
        },
    )
    (output / "final.pt").write_bytes((source / "final.pt").read_bytes())
    write_json(
        replay / "import_coverage.json", {"methods": {"m": {"s1_sources": {"a": str(source)}}}}
    )
    assert checks._check_imports([replay])
    (output / "final.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint differs") as error:
        checks._check_imports([replay])
    assert "0.123456789" not in str(error.value)


def test_capacity_uses_own_reported_memory_despite_host_pid_mismatch(gpu_evidence):
    experiments, samples = gpu_evidence
    sample = samples[0]
    sample["gpus"][0].update(total_bytes=93 * 2**30, used_bytes=70 * 2**30)
    sample["slots_after"][0].update(
        required_bytes=66 * 2**30, current_reserved_now_bytes=57 * 2**30
    )
    sample["applications"][0].update(pid=987654321, used_bytes=57 * 2**30)
    # Another study's slot is external, even though it uses the same admission directory.
    sample["slots_after"].append(
        {
            "pid": 2,
            "slot_file": "slot_other_study.json",
            "device": "cuda:0",
            "required_bytes": 80 * 2**30,
            "current_reserved_bytes": 10 * 2**30,
        }
    )
    assert checks._check_admission(experiments, samples * 2)
    sample["gpus"][0]["used_bytes"] = 90 * 2**30
    with pytest.raises(ValueError, match="exceeds total"):
        checks._check_admission(experiments, samples * 2)


def test_capacity_requires_successful_gpu_evidence(gpu_evidence):
    experiments, _ = gpu_evidence
    with pytest.raises(ValueError, match="missing GPU samples"):
        checks._check_admission(experiments, [{"gpus": [], "gpu_error": "timeout"}])


@pytest.mark.parametrize("packed,rate", [(False, 16), (False, 32), (True, 4), (True, 8)])
def test_capacity_accepts_reviewers_two_and_three_job_monitor_lag_ramps(gpu_evidence, packed, rate):
    """Replay x2rev/lag_race{,_packed}.py with a deterministic 1-second monitor."""
    experiments, samples = gpu_evidence
    base = samples[0]
    gib = 2**30

    def slot(name, required, reserved):
        return {
            "slot_file": f"slot_{name}.json",
            "pid": 1,
            "device": "cuda:0",
            "required_bytes": required * gib,
            "current_reserved_bytes": 0,  # The 30-second minimum still lags.
            "current_reserved_now_bytes": int(reserved * gib),
        }

    ccnet = slot("ccnet", 66, 61)
    ccnet["pid"] = 101
    nersi = slot("nersi", 6, 4)
    nersi["pid"] = 102
    base["experiments"][str(experiments[0])]["states"].update(
        ccnet={"pid": 101}, nersi={"pid": 102}
    )
    steady = [ccnet, nersi] if packed else [ccnet]
    replay = []
    old_excesses = 0
    for instant in np.arange(0.05, 14 / rate + 0.2, 0.05):
        sample = deepcopy(base)
        actual = min(14, instant * rate)
        before = min(14, int(instant) * rate)
        after = min(14, int(instant + 1.5) * rate)
        sample["slots_before"] = steady + [slot("regsi", 16, before)]
        sample["slots_after"] = steady + [slot("regsi", 16, after)]
        used = int(
            (61 + (4 if packed else 0) + actual + 0.6 * len(sample["slots_before"]) + 1.5) * gib
        )
        sample["gpus"][0].update(total_bytes=95830 * 2**20, used_bytes=used)
        old_owned = sum(s["current_reserved_now_bytes"] + gib for s in sample["slots_before"])
        required = sum(s["required_bytes"] for s in sample["slots_before"])
        old_excesses += required + max(0, used - old_owned) > 95830 * 2**20 - 2 * gib
        replay.append(sample)
    assert old_excesses > 0  # Each reviewer scenario exposes the original race.
    assert checks._check_admission(experiments, replay)
    # A real over-commit still fails after the ramp, with fresh slot evidence.
    overcommitted = deepcopy(replay[-1])
    overcommitted["gpus"][0]["used_bytes"] += 20 * gib
    with pytest.raises(ValueError, match="exceeds total"):
        checks._check_admission(experiments, [overcommitted, deepcopy(overcommitted)])


def test_capacity_uses_per_slot_max_in_both_directions_and_counts_slots_once(gpu_evidence):
    experiments, samples = gpu_evidence
    sample = samples[0]
    first = sample["slots_after"][0]
    first.update(required_bytes=45 * 2**30, current_reserved_now_bytes=40 * 2**30)
    second = first | {"slot_file": "slot_second.json", "pid": 2}
    sample["slots_before"] = [first.copy(), second | {"current_reserved_now_bytes": 0}]
    sample["slots_after"] = [first | {"current_reserved_now_bytes": 0}, second]
    sample["experiments"][str(experiments[0])]["states"]["second"] = {"pid": 2}
    sample["gpus"][0].update(total_bytes=100 * 2**30, used_bytes=82 * 2**30)
    assert checks._check_admission(experiments, samples * 2)


def test_capacity_requires_consecutive_excesses_on_the_same_device(gpu_evidence):
    experiments, samples = gpu_evidence
    healthy = samples[0]
    excess = deepcopy(healthy)
    excess["slots_after"][0]["required_bytes"] = 99 * 2**30
    assert checks._check_admission(experiments, [healthy, excess, healthy, excess])
    assert checks._check_admission(experiments, [excess, {"gpus": []}, excess])
    other_device = deepcopy(excess)
    other_device["gpus"][0]["device"] = "cuda:1"
    other_device["slots_after"][0]["device"] = "cuda:1"
    assert checks._check_admission(experiments, [excess, other_device, excess])
    with pytest.raises(ValueError, match="consecutive samples"):
        checks._check_admission(experiments, [healthy, excess, deepcopy(excess)])


def test_capacity_ramp_uses_instantaneous_reservation_during_thirty_second_minimum(gpu_evidence):
    """Regression for scratchpad/x1verify/ramp_capacity.py's FORGE a05 ramp."""
    experiments, samples = gpu_evidence
    reports = deque([(0.0, 0)])
    for second in range(1, 21):
        reserved_now = 0 if second < 10 else int(57.5 * 2**30)
        minimum = _recent_reserved_minimum(reports, float(second), reserved_now)
    assert minimum == 0
    sample = samples[0]
    sample["gpus"][0].update(total_bytes=95830 * 2**20, used_bytes=58 * 2**30)
    sample["slots_after"][0].update(
        required_bytes=66 * 2**30,
        current_reserved_bytes=minimum,
        current_reserved_now_bytes=reserved_now,
        peak_reserved_bytes=reserved_now,
    )
    assert checks._check_admission(experiments, samples * 2)
    sample["slots_after"][0]["current_reserved_now_bytes"] = 0
    with pytest.raises(ValueError, match="exceeds total"):
        checks._check_admission(experiments, samples * 2)
    sample["slots_after"][0].pop("current_reserved_now_bytes")
    with pytest.raises(KeyError, match="current_reserved_now_bytes"):
        checks._check_admission(experiments, samples * 2)


def test_capacity_context_allowance_is_per_process_and_does_not_hide_external_use(gpu_evidence):
    experiments, samples = gpu_evidence
    sample = samples[0]
    sample["gpus"][0].update(total_bytes=93 * 2**30, used_bytes=59 * 2**30)
    sample["slots_after"][0].update(
        required_bytes=91 * 2**30, current_reserved_now_bytes=58 * 2**30
    )
    assert checks._check_admission(experiments, samples * 2)
    # A second slot for the same allocator must not double-count its memory/context.
    sample["slots_after"].append(
        sample["slots_after"][0] | {"slot_file": "slot_same_pid.json", "required_bytes": 0}
    )
    sample["gpus"][0]["used_bytes"] += 1
    with pytest.raises(ValueError, match="exceeds total"):
        checks._check_admission(experiments, samples * 2)


@pytest.mark.parametrize(
    "changed", ["relmse", "snr", "loss", "steps", "shape", "nan", "metric_nan"]
)
def test_benchmarked_ccnet_twins_compare_run_summaries_and_loss_correlation(tmp_path, changed):
    left, right = tmp_path / "left", tmp_path / "right"
    for path, delta in ((left, 0.0), (right, 1e-4)):
        path.mkdir()
        (path / "config.resolved.yaml").write_text("execution: {cudnn_benchmark: true}\n")
        pd.DataFrame({"step": range(1, 301), "loss": np.linspace(1.0, 0.5, 300) + delta}).to_csv(
            path / "history.csv", index=False
        )
        # bf16 ULPs and zero-crossing differences exceed element-wise tolerances.
        np.save(path / "prediction.npy", np.array([[0, 0.0123, -9.27]]) + 50 * delta)
        summary = {"masked_trace_relative_mse": 0.5 + delta, "mean_trace_snr_db": 3.0 + delta}
        validation = {"clean_target": summary} if path == right else summary
        write_json(
            path / "manifest.json", {"status": "completed", "metrics": {"validation": validation}}
        )
    checks._check_ccnet_twin(left, right)
    if changed in ("relmse", "snr", "metric_nan"):
        record = json.loads((right / "manifest.json").read_text())
        metric = "mean_trace_snr_db" if changed == "snr" else "masked_trace_relative_mse"
        record["metrics"]["validation"]["clean_target"][metric] = (
            np.nan if changed == "metric_nan" else 9.0
        )
        write_json(right / "manifest.json", record)
    elif changed in ("shape", "nan"):
        prediction = np.ones((1, 4)) if changed == "shape" else np.ones((1, 3))
        if changed == "nan":
            prediction[0, 0] = np.nan
        np.save(right / "prediction.npy", prediction)
    else:
        history = pd.read_csv(right / "history.csv")
        if changed == "loss":
            history["loss"] = history.loss.to_numpy()[::-1]
        else:
            history.loc[0, "step"] = 2
        history.to_csv(right / "history.csv", index=False)
    with pytest.raises(ValueError, match="exceeds tolerance|correlation|invalid"):
        checks._check_ccnet_twin(left, right)


@pytest.mark.parametrize("filename", ["history.csv", "prediction.npy"])
def test_benchmark_off_ccnet_twins_require_bitwise_history_and_prediction(tmp_path, filename):
    left, right = tmp_path / "left", tmp_path / "right"
    for path in (left, right):
        path.mkdir()
        (path / "config.resolved.yaml").write_text("execution: {cudnn_benchmark: false}\n")
        pd.DataFrame({"step": range(1, 301), "loss": np.linspace(1.0, 0.5, 300)}).to_csv(
            path / "history.csv", index=False
        )
        np.save(path / "prediction.npy", np.ones((2, 3)))
    checks._check_ccnet_twin(left, right)
    if filename == "history.csv":
        history = pd.read_csv(right / filename)
        history.loc[0, "loss"] += 1e-12
        history.to_csv(right / filename, index=False)
    else:
        prediction = np.load(right / filename)
        prediction[0, 0] += 1e-12
        np.save(right / filename, prediction)
    with pytest.raises(ValueError, match="differs in bytes"):
        checks._check_ccnet_twin(left, right)


@pytest.mark.parametrize(
    "metric,delta,accepted",
    [
        ("masked_trace_relative_mse", 1.0, True),
        ("masked_trace_relative_mse", 1.001, False),
        ("masked_trace_relative_mse", -1.001, False),
        ("mean_trace_snr_db", 0.05, True),
        ("mean_trace_snr_db", 0.0501, False),
        ("mean_trace_snr_db", -0.0501, False),
    ],
)
def test_benchmarked_ccnet_summary_tolerances(tmp_path, metric, delta, accepted):
    left, right = tmp_path / "left", tmp_path / "right"
    for path in (left, right):
        summary = {"masked_trace_relative_mse": 100.0, "mean_trace_snr_db": 0.0}
        if path == right:
            summary[metric] += delta
        write_json(
            path / "manifest.json", {"status": "completed", "metrics": {"validation": summary}}
        )
    loss = np.linspace(1.0, 0.5, 300)
    if accepted:
        checks._check_ccnet_summaries(left, right, loss, loss)
    else:
        with pytest.raises(ValueError, match="exceeds tolerance"):
            checks._check_ccnet_summaries(left, right, loss, loss)


@pytest.mark.parametrize("leaked", [False, True])
def test_oom_threads_are_compared_with_pre_oom_baseline(tmp_path, leaked):
    fault = {"method": "ccnet5d_grid", "run": "a05_b02", "point": "report", "action": "oom"}
    seal(tmp_path, [fault], "seg_c3_na")
    write_json(tmp_path / "rehearsal_faults/000.json", {"attempt": 0})
    output = tmp_path / "method_runs/ccnet5d_grid/S1/a05_b02"
    write_json(
        output / "attempts/1/manifest.json",
        {"pid": 1, "threads_at_start": 4, "peak_cuda_reserved_bytes": 100},
    )
    count = 5 if leaked else 4
    write_json(
        output / "manifest.json",
        {
            "pid": 1,
            "threads_at_start": count,
            "threads_at_end": count,
            "peak_cuda_reserved_bytes": 100,
        },
    )
    write_json(
        tmp_path / "method_runs/pocs_grid/S1/a/manifest.json",
        {"method_id": "pocs_grid", "nice": 10},
    )
    if leaked:
        with pytest.raises(ValueError, match="clean up in-process resources"):
            checks._check_specific_faults([tmp_path], [])
    else:
        assert checks._check_specific_faults([tmp_path], [])
