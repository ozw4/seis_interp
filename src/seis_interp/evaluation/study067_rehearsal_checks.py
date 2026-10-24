"""Structural rehearsal gates. Never print held-out metric or prediction values."""

import hashlib
import json
import math
import re
import subprocess
import sys
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.processing.experiment_commonality import load_experiment_grid
from seis_interp.training.rehearsal import Rehearsal, load_rehearsal_plan, rehearsal_candidates


def _json(path):
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "expected a JSON object", path)
    return value


def _jsonl(path):
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    _require(all(isinstance(value, dict) for value in values), "expected JSON object samples", path)
    return values


def _require(condition, message, *paths):
    if not condition:
        raise ValueError(f"{message}; evidence=" + ", ".join(str(path) for path in paths))


def _time(value):
    _require(isinstance(value, str), "missing timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _start(record):
    return _time(record.get("started_at", record.get("start_time")))


def _end(record):
    return _time(record.get("ended_at", record.get("end_time")))


def _records(experiment):
    return [
        (path.parent, _json(path))
        for path in sorted((experiment / "method_runs").glob("*/*/*/manifest.json"))
    ]


def _all_attempts(experiment):
    return [
        (path.parent, _json(path))
        for path in sorted((experiment / "method_runs").rglob("manifest.json"))
    ]


def _admission_records(experiment):
    for path, row in _all_attempts(experiment):
        # Post-barrier scoring does not reacquire a slot. Keep the earlier seed
        # child's admission/interval evidence through that CPU-only phase.
        prediction = row.get("prediction_execution", {})
        if row.get("cuda_admission"):
            yield path, row
        elif prediction.get("cuda_admission"):
            yield path, row | prediction


def _overlap(first, second):
    return max(_start(first), _start(second)) < min(_end(first), _end(second))


def _seed_wait_reserved_capacity(record, details, samples):
    """Match raw-free outliers to sampled, already-promised regrowth capacity."""
    execution = record | record.get("prediction_execution", {})
    admission = execution["cuda_admission"]
    if "wait_seconds" not in admission or "pid" not in admission:
        return False
    end = _start(execution)
    start = end - admission["wait_seconds"]
    matching = [
        sample
        for sample in samples
        if sample.get("at")
        and start <= _time(sample["at"]) <= end
        and all(
            any(
                gpu["device"] == device and abs(gpu["free_bytes"] - free) < 2**20
                for gpu in sample["gpus"]
            )
            for device, free in details["free_memory_bytes"].items()
        )
    ]
    # nvidia-smi reports whole MiB; NVML's byte counts differ by <1 MiB.
    # Require matching evidence, and prove insufficient unpromised capacity on
    # both bracketing slot reads. Never excuse an outlier just because it is brief.
    return bool(matching) and all(
        free
        - sum(
            max(0, slot["required_bytes"] - slot["current_reserved_bytes"])
            for slot in sample[key]
            if slot["device"] == device and slot["pid"] != admission["pid"]
        )
        < details["required_bytes"]
        for sample in matching
        for key in ("slots_before", "slots_after")
        for device, free in details["free_memory_bytes"].items()
    )


def _check_seed_admission(forge, samples):
    contract = yaml.safe_load((forge / "study/config.yaml").read_text())
    expected = set(contract["trace_split"]["evaluation_seeds"])
    seeds = {
        record["trace_seed"]: record
        for _, record in _records(forge)
        if record.get("method_id") == "ccnet5d_grid" and record.get("stage") == "evaluation"
    }
    _require(
        set(seeds) == expected,
        "missing FORGE a05 seed child",
        forge / "method_runs/ccnet5d_grid/evaluation",
    )
    for record in seeds.values():
        admission = record.get("cuda_admission") or record.get("prediction_execution", {}).get(
            "cuda_admission"
        )
        _require(
            record["id"].startswith("a05_") and admission,
            "FORGE a05 seed child was never admitted",
            forge / "method_runs/ccnet5d_grid/evaluation",
        )
    path = forge / "workers/ccnet5d_grid.log"
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"ccnet5d_grid seed_(\d+): waiting for GPU: (.+)", line)
        if match is None:
            continue
        try:
            details = literal_eval(match[2])
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"invalid seed admission wait; evidence={path}") from error
        _require(int(match[1]) in expected, "unexpected seed admission wait", path)
        if details.get("reason") != "insufficient_free_memory":
            continue
        eligible = set(contract["cuda_execution"]["devices"]) - set(details["excluded_devices"])
        free = details["free_memory_bytes"]
        required = details["required_bytes"]
        _require(
            set(free) == eligible
            and isinstance(required, (int, float))
            and math.isfinite(required)
            and required > 0
            and all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in free.values()
            )
            and (
                all(value < required for value in free.values())
                or _seed_wait_reserved_capacity(seeds[int(match[1])], details, samples)
            ),
            "FORGE a05 seed waited with eligible memory available or invalid wait evidence",
            path,
        )


def _check_freeze(experiment, samples, launch_commit):
    seal = _json(experiment / "experiment.json")
    _require(
        seal.get("rehearsal") is True and seal.get("dirty_worktree") is False,
        "rehearsal must be sealed from a clean worktree",
        experiment / "experiment.json",
    )
    _require(
        seal.get("git_commit") == launch_commit,
        "seal differs from launch commit",
        experiment / "experiment.json",
    )
    Rehearsal(experiment)
    for name, digest in seal["artifacts"].items():
        if name.startswith("source/"):
            _require(
                sha256_file(experiment / name) == digest,
                "frozen source hash changed",
                experiment / name,
            )
    expected = str(experiment / "source/src")
    events_path = experiment / "workers/rehearsal_events.jsonl"
    events = _jsonl(events_path)
    workers = [event for event in events if "method_id" in event]
    _require(workers, "missing worker source evidence", events_path)
    for event in workers:
        _require(event.get("source_root") == expected, "worker source is not frozen", events_path)
        for child in event.get("child_processes", []):
            _require(
                child.get("source_root") == expected, "seed child source is not frozen", events_path
            )
    for path, record in _all_attempts(experiment):
        if "prediction_execution" in record:
            _require(
                record["prediction_execution"].get("source_root") == expected,
                "seed prediction source is not frozen",
                path / "manifest.json",
            )
        _require(
            record.get("source_root") == expected,
            "run/seed child used live source",
            path / "manifest.json",
        )
    observations = [sample["experiments"].get(str(experiment), {}) for sample in samples]
    supervisor_seen = False
    for value in observations:
        state = value.get("states", {}).get("supervisor", {})
        for process in value.get("processes", []):
            if process["pid"] == state.get("pid"):
                supervisor_seen = True
                _require(
                    process.get("pythonpath") == expected
                    and str(experiment / "source/scripts") in process["cmdline"],
                    "supervisor environment/script is not frozen",
                    experiment / "supervisor.out",
                )
    _require(supervisor_seen, "no live /proc supervisor sample", experiment / "supervisor.out")
    dataset = "forge_grid" if seal.get("dataset_id") == "forge_2017" else "c3_region"
    script = experiment / "source/scripts" / f"run_{dataset}_experiment.py"
    result = subprocess.run(
        [sys.executable, str(script), "status", "--experiment", str(experiment)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    _require(result.returncode == 0 and result.stdout.strip(), "frozen status CLI failed", script)
    return [experiment / "experiment.json", events_path, experiment / "supervisor.out"]


def _check_admission(experiments, samples):
    admitted = [
        (path, record)
        for experiment in experiments
        for path, record in _admission_records(experiment)
    ]
    _require(admitted, "no GPU admissions", *experiments)
    devices = {record["cuda_admission"]["device"] for _, record in admitted}
    _require({"cuda:0", "cuda:1"} <= devices, "both devices must be used", *experiments)
    _require(
        any(
            a["cuda_admission"]["device"] != b["cuda_admission"]["device"] and _overlap(a, b)
            for _, a in admitted
            for _, b in admitted
            if a.get("ended_at", a.get("end_time")) and b.get("ended_at", b.get("end_time"))
        ),
        "no overlapping runs across devices",
        *experiments,
    )
    forge = experiments[1]
    forge_s1 = [
        (path, record)
        for path, record in _admission_records(forge)
        if record.get("stage") == "S1" and record.get("method_id") == "ccnet5d_grid"
    ]
    _require(
        any(
            a["id"] != b["id"]
            and a.get("cuda_admission", {}).get("device")
            != b.get("cuda_admission", {}).get("device")
            and _overlap(a, b)
            for _, a in forge_s1
            for _, b in forge_s1
            if a.get("cuda_admission")
            and b.get("cuda_admission")
            and a.get("ended_at", a.get("end_time"))
            and b.get("ended_at", b.get("end_time"))
        ),
        "FORGE a05 shards did not overlap on different GPUs",
        forge / "method_runs/ccnet5d_grid/S1",
    )
    _check_seed_admission(forge, samples)
    for experiment in experiments:
        events = _jsonl(experiment / "workers/rehearsal_events.jsonl")
        for event in events:
            _require(
                len(event.get("child_processes", [])) <= 2,
                "more than two concurrent children per method",
                experiment / "workers/rehearsal_events.jsonl",
            )
        for path, record in _admission_records(experiment):
            _require(
                record.get("external_gpu_utilization_at_admission") is not None,
                "missing external utilization observation",
                path / "manifest.json",
            )
        for path, record in _all_attempts(experiment):
            if record.get("cuda_admission"):
                _require(
                    record.get("external_gpu_utilization_at_admission") is not None,
                    "missing external utilization observation",
                    path / "manifest.json",
                )
            if record.get("imported_from"):
                _require(
                    not record.get("cuda_admission"),
                    "imported cell acquired a slot",
                    path / "manifest.json",
                )
            reason = record.get("failure_reason", "").lower()
            if "out of memory" in reason:
                _require(
                    "rehearsal injected out of memory" in reason,
                    "unexpected OOM",
                    path / "manifest.json",
                )
    _require(any(sample["gpus"] for sample in samples), "missing GPU samples", *experiments)
    previous_violations = set()
    for sample in samples:
        # Admission and /proc identities share our namespace. nvidia-smi process
        # IDs may belong to the host namespace and cannot identify owned memory.
        pids = {record["pid"] for _, record in admitted if "pid" in record}
        for experiment in experiments:
            states = sample.get("experiments", {}).get(str(experiment), {}).get("states", {})
            pids.update(state.get("pid") for state in states.values())
            pids.update(
                child["pid"]
                for state in states.values()
                for child in state.get("child_processes", [])
            )
        violations = set()
        for gpu in sample["gpus"]:
            slots = _capacity_slots(sample, pids, gpu["device"])
            # Allocator observations are per process, even if it holds multiple
            # slots. Allow 1 GiB per context for memory outside PyTorch's allocator.
            reserved_by_pid = {}
            for slot in slots:
                reserved = slot["current_reserved_now_bytes"]
                reserved_by_pid[slot["pid"]] = max(reserved_by_pid.get(slot["pid"], 0), reserved)
            owned = sum(reserved + 2**30 for reserved in reserved_by_pid.values())
            external = max(0, gpu["used_bytes"] - owned)
            required = sum(slot["required_bytes"] for slot in slots)
            # Idle devices with external users alone do not constitute an admission.
            if slots and required + external > gpu["total_bytes"] - 2 * 2**30:
                violations.add(gpu["device"])
        _require(
            not (violations & previous_violations),
            "capacity exceeds total minus context allowance in consecutive samples",
            *experiments,
        )
        previous_violations = violations
    # Cached/finalized S1 manifests are unchanged when workers revisit them on relaunch.
    events = [
        event
        for experiment in experiments
        for event in _jsonl(experiment / "workers/rehearsal_events.jsonl")
    ]
    _require(
        any(event.get("last_admission") for event in events),
        "missing admission state",
        *experiments,
    )
    return [experiment / "method_runs" for experiment in experiments]


def _capacity_slots(sample, pids, device):
    """Count each slot once, with its largest bracketing allocator observation."""
    slots = {}
    for slot in sample["slots_before"] + sample["slots_after"]:
        if slot["device"] != device or slot["pid"] not in pids:
            continue
        reserved = slot["current_reserved_now_bytes"]
        _require(
            isinstance(reserved, (int, float)) and math.isfinite(reserved) and reserved >= 0,
            "invalid instantaneous reservation",
        )
        name = slot["slot_file"]
        if name not in slots:
            slots[name] = slot.copy()
        else:
            slots[name]["current_reserved_now_bytes"] = max(
                slots[name]["current_reserved_now_bytes"], reserved
            )
            slots[name]["required_bytes"] = max(
                slots[name]["required_bytes"], slot["required_bytes"]
            )
    return list(slots.values())


def _check_warmup(experiments):
    count = 0
    for experiment in experiments:
        for path, record in _admission_records(experiment):
            if (
                "/attempts/" in str(path)
                or not record.get("cuda_admission")
                or record["status"] != "completed"
            ):
                continue
            diagnostics = record["diagnostics"]
            after = diagnostics["cuda_reserved_after_prediction_bytes"]
            before = max(
                diagnostics["cuda_reserved_after_warmup_bytes"],
                diagnostics["cuda_reserved_before_prediction_bytes"],
            )
            _require(
                after <= before * 1.02,
                "prediction grew reserved memory by more than 2%",
                path / "manifest.json",
            )
            peak = max(
                diagnostics[f"cuda_max_reserved_{phase}_bytes"]
                for phase in ("after_warmup", "before_prediction", "after_prediction")
            )
            _require(
                peak <= record["cuda_admission"]["required_bytes"],
                "peak exceeds admitted budget",
                path / "manifest.json",
            )
            count += 1
    _require(count > 0, "missing warmup evidence", *experiments)
    return [experiment / "method_runs" for experiment in experiments]


def _resumed(log, step, phase):
    matches = re.findall(
        r"resumed step=(\d+) phase=(\w+) state_sha256_saved=(\w+) "
        r"state_sha256_loaded=(\w+) adamw_step_device=(\w+)",
        log,
    )
    return any(
        int(saved_step) == step and saved_phase == phase and saved == loaded and device == "cpu"
        for saved_step, saved_phase, saved, loaded, device in matches
    )


def _loss_line(log, step):
    lines = [line for line in log.splitlines() if re.search(rf"\bstep {step}/", line)]
    _require(lines, f"missing re-logged training step {step}")
    match = re.search(r"(?:loss=|minibatch_loss )([-+\deE.]+)", lines[-1])
    _require(match is not None, "missing training loss in report")
    return match[1]


def _ccnet_benchmark(output):
    config = yaml.safe_load((output / "config.resolved.yaml").read_text())
    return config.get("execution", {}).get("cudnn_benchmark", False)


def _check_ccnet_twin(left, right):
    benchmark = _ccnet_benchmark(left)
    _require(
        benchmark == _ccnet_benchmark(right), "CCNet twin benchmark settings differ", left, right
    )
    first, second = [pd.read_csv(path / "history.csv") for path in (left, right)]
    _require(
        first.step.tolist() == second.step.tolist() == list(range(1, 301))
        and np.isfinite(first.loss).all()
        and np.isfinite(second.loss).all(),
        "CCNet twin training history has invalid steps or losses",
        left / "history.csv",
        right / "history.csv",
    )
    predictions = [
        np.load(path / "prediction.npy", allow_pickle=False, mmap_mode="r")
        for path in (left, right)
    ]
    _require(
        predictions[0].shape == predictions[1].shape
        and all(np.isfinite(prediction).all() for prediction in predictions),
        "CCNet twin prediction has invalid shape or values",
        left / "prediction.npy",
        right / "prediction.npy",
    )
    if not benchmark:
        for name in ("history.csv", "prediction.npy"):
            _require(
                sha256_file(left / name) == sha256_file(right / name),
                "CCNet twin differs in bytes",
                left / name,
                right / name,
            )
        return
    _check_ccnet_summaries(left, right, first.loss, second.loss)


def _check_ccnet_summaries(left, right, first_loss, second_loss):
    log = left / "execution.log"
    resumes = re.findall(r"resumed step=(\d+) phase=training\b", log.read_text())
    _require(len(resumes) == 1, "expected one CCNet training restart", log)
    resume_step = int(resumes[0])
    _require(
        100 <= resume_step < len(first_loss), "insufficient CCNet restart comparison updates", log
    )
    delta = np.abs(np.asarray(first_loss) - np.asarray(second_loss))
    # Include every sliding 100-update window wholly before the re-executed updates.
    pre_max = float(np.convolve(delta[:resume_step], np.ones(100) / 100, mode="valid").max())
    post_mean = float(delta[resume_step:].mean())
    _require(
        post_mean <= 3 * pre_max,
        "CCNet twin post-restart loss divergence exceeds 3 times pre-restart maximum",
        left / "history.csv",
        right / "history.csv",
    )
    summaries = []
    for path in (left, right):
        record = _json(path / "manifest.json")
        _require(record.get("status") == "completed", "CCNet twin did not complete", path)
        validation = record["metrics"]["validation"]
        summaries.append(validation.get("clean_target", validation))
    for metric, relative, tolerance in (
        ("masked_trace_relative_mse", True, 0.01),
        ("mean_trace_snr_db", False, 0.05),
    ):
        a, b = [summary[metric] for summary in summaries]
        _require(
            all(isinstance(value, (int, float)) and math.isfinite(value) for value in (a, b))
            and abs(a - b) <= tolerance * (abs(a) if relative else 1.0),
            f"CCNet twin validation {metric} exceeds tolerance",
            left / "manifest.json",
            right / "manifest.json",
        )


def _check_determinism_twins(experiments, twins, determinism_twins):
    evidence = []
    _require(
        len(determinism_twins) == len(set(determinism_twins)) == 4
        and not set(determinism_twins).intersection(experiments + twins),
        "supply four separate benchmark-off experiments using --determinism-twin",
        *determinism_twins,
    )
    for primary in experiments:
        forge = _json(primary / "experiment.json").get("dataset_id") == "forge_2017"
        pair = [
            twin
            for twin in determinism_twins
            if (_json(twin / "experiment.json").get("dataset_id") == "forge_2017") == forge
        ]
        _require(
            len(pair) == 2, "determinism twins must include two per dataset", *determinism_twins
        )
        candidate = "a05_b05" if forge else "a05_b02"
        outputs = []
        for twin in pair:
            plan = Rehearsal(twin).plan
            _require(
                plan is not None and not plan.get("faults"), "determinism twin has faults", twin
            )
            output = twin / "method_runs/ccnet5d_grid/S1" / candidate
            _require(
                not _ccnet_benchmark(output), "determinism twin must disable benchmark", output
            )
            outputs.append(output)
            evidence.extend(output / name for name in ("history.csv", "prediction.npy"))
        _check_ccnet_twin(*outputs)
    return evidence


def _fault_output(experiment, fault):
    stage = "evaluation" if fault["run"].startswith("seed_") else "S1"
    return experiment / "method_runs" / fault["method"] / stage / fault["run"]


def _check_restart_evidence(experiment, plan):
    path = experiment / "workers/rehearsal_restarts.jsonl"
    restarts = _jsonl(path) if path.exists() else []
    expected = [
        fault
        for fault in plan["faults"]
        if fault["action"] in ("exit", "sticky_cuda")
        or fault["action"] == "sleep"
        and fault["method"] == "regsi_real"
    ]
    for fault in expected:
        rows = [
            row
            for row in restarts
            if row["worker"].split("_shard_", 1)[0] == fault["method"]
            and row["run"][1] == fault["run"]
        ]
        _require(
            len(rows) == 1 and rows[0]["retries"] == 1,
            "faulted worker must restart exactly once",
            path,
        )
    for row in restarts:
        _require(
            any(
                row["worker"].split("_shard_", 1)[0] == fault["method"]
                and row["run"][1] == fault["run"]
                for fault in expected
            ),
            "a worker without a process fault restarted",
            path,
        )
    events = _jsonl(experiment / "workers/rehearsal_events.jsonl")
    for event in events:
        for child in event.get("child_processes", []):
            _require(
                child.get("restarts", 0) == 0,
                "a seed child restarted unexpectedly",
                experiment / "workers/rehearsal_events.jsonl",
            )
    for worker in {event["worker"] for event in events if "method_id" in event}:
        rows = [event for event in events if event["worker"] == worker]
        for first, second in zip(rows, rows[1:], strict=False):
            if first.get("pid") != second.get("pid"):
                _require(
                    any(
                        row["worker"] == worker and row.get("previous_pid") == first.get("pid")
                        for row in restarts
                    ),
                    "worker PID changed without a recorded restart",
                    path,
                )
    return path


def _check_resume(experiments, twins, determinism_twins=()):
    evidence = []
    for experiment in experiments:
        plan = Rehearsal(experiment).plan
        evidence.append(_check_restart_evidence(experiment, plan))
        for fault in plan["faults"]:
            if fault["point"] not in ("report", "checkpoint") or fault["action"] == "nonfinite":
                continue
            output = _fault_output(experiment, fault)
            previous = output / "attempts/1"
            final = _json(output / "manifest.json")
            _require(
                previous.is_dir() and final["attempt"] == 1,
                "fault must resume exactly once",
                output,
            )
            step = 300 if fault["point"] == "checkpoint" else 200
            log = (output / "execution.log").read_text()
            _require(
                _resumed(log, step, "before_prediction" if step == 300 else "training"),
                "missing state-exact resume evidence",
                output / "execution.log",
            )
            history = pd.read_csv(output / "history.csv")
            _require(
                history.step.tolist() == list(range(1, 301)),
                "history must contain each update exactly once",
                output / "history.csv",
            )
            if fault["point"] == "report":
                old_loss = _loss_line((previous / "execution.log").read_text(), 250)
                new_loss = _loss_line(log, 250)
                benchmark = fault["method"] == "ccnet5d_grid" and _ccnet_benchmark(output)
                same = (
                    math.isclose(
                        float(old_loss),
                        float(new_loss),
                        rel_tol=1e-3,
                    )
                    if fault["method"] == "regsi_real"
                    else old_loss == new_loss
                )
                if benchmark:
                    # Benchmark-on restart trajectories are checked against the
                    # uninterrupted twin's full curve and validation summaries.
                    same = math.isfinite(float(old_loss)) and math.isfinite(float(new_loss))
                _require(
                    same,
                    "re-logged training loss differs",
                    previous / "execution.log",
                    output / "execution.log",
                )
            if fault["action"] == "oom":
                _require(
                    _json(previous / "manifest.json")["pid"] == final["pid"],
                    "OOM changed worker PID",
                    output,
                )
            evidence.append(output)
        # Verify a restart of one worker never replaced a sibling that was live.
        events = _jsonl(experiment / "workers/rehearsal_events.jsonl")
        for method in _json(experiment / "experiment.json")["methods"]:
            rows = [row for row in events if row.get("method_id") == method and "worker" in row]
            for first, second in zip(rows, rows[1:], strict=False):
                if first.get("pid") != second.get("pid") and first["worker"] == second["worker"]:
                    _require(
                        first.get("status") != "completed",
                        "completed worker was restarted",
                        experiment / "workers/rehearsal_events.jsonl",
                    )
    _require(len(twins) == 2, "supply both CCNet twin experiments using --twin", *twins)
    _require(
        {
            "forge" if _json(twin / "experiment.json").get("dataset_id") == "forge_2017" else "c3"
            for twin in twins
        }
        == {"c3", "forge"},
        "twins must include one experiment per dataset",
        *twins,
    )
    for twin in twins:
        seal = _json(twin / "experiment.json")
        _require(
            Rehearsal(twin).plan is not None and not Rehearsal(twin).plan.get("faults"),
            "twins must be sealed rehearsals without faults",
            twin / "experiment.json",
        )
        primary = experiments[1] if seal.get("dataset_id") == "forge_2017" else experiments[0]
        candidate = "a05_b05" if primary == experiments[1] else "a05_b02"
        left = primary / "method_runs/ccnet5d_grid/S1" / candidate
        right = twin / "method_runs/ccnet5d_grid/S1" / candidate
        _require(
            _ccnet_benchmark(left) and _ccnet_benchmark(right),
            "--twin must supply benchmark-on runs",
            left,
            right,
        )
        _check_ccnet_twin(left, right)
        evidence.extend(right / name for name in ("history.csv", "prediction.npy", "manifest.json"))
    evidence.extend(_check_determinism_twins(experiments, twins, determinism_twins))
    return evidence


def _check_specific_faults(experiments, samples):
    evidence = []
    for experiment in experiments:
        plan = Rehearsal(experiment).plan
        for index, fault in enumerate(plan["faults"]):
            event = experiment / "rehearsal_faults" / f"{index:03d}.json"
            fired = _json(event)
            _require(
                fired["attempt"] == fault.get("attempt", 0), "fault fired on wrong attempt", event
            )
            output = _fault_output(experiment, fault)
            final = _json(output / "manifest.json")
            previous = output / "attempts/1"
            if fault["action"] == "resource_wait":
                _require(
                    not (output / "attempts").exists() and final["attempt"] == 0,
                    "resource wait consumed an attempt",
                    output,
                )
            if fault["action"] == "oom":
                old = _json(previous / "manifest.json")
                _require(
                    old["pid"] == final["pid"]
                    and final["threads_at_end"] <= old["threads_at_start"],
                    "OOM did not clean up in-process resources",
                    output,
                )
                _require(
                    old["peak_cuda_reserved_bytes"] > 0
                    and 0
                    < final["peak_cuda_reserved_bytes"]
                    <= 1.10 * old["peak_cuda_reserved_bytes"],
                    "retry peak is nonpositive or exceeds original by more than 10%",
                    output,
                )
            if fault["action"] == "sticky_cuda":
                old = _json(previous / "manifest.json")
                _require(
                    old["pid"] != final["pid"] and _end(old) - _time(fired["fired_at"]) <= 60,
                    "sticky CUDA did not leave its process promptly",
                    previous / "manifest.json",
                    event,
                )
                # Require a sampled disappearance within the specified bound.
                later = [
                    sample
                    for sample in samples
                    if 0 <= _time(sample["at"]) - _time(fired["fired_at"]) <= 60
                ]
                _require(
                    any(
                        all(
                            process["pid"] != old["pid"]
                            for value in sample["experiments"].values()
                            for process in value["processes"]
                        )
                        for sample in later
                    ),
                    "sticky CUDA process still present after 60s",
                    event,
                )
            if fault["action"] == "sleep" and fault["method"] == "regsi_real":
                dog = _json(previous / "watchdog.json")
                _require(
                    dog["threshold_seconds"] >= plan["watchdog"]["floor_seconds"]
                    and dog.get("exit_code") == 75,
                    "watchdog exit/threshold evidence missing",
                    previous / "watchdog.json",
                )
            if fault["method"] == "drr_grid":
                _require(
                    not list(output.rglob("watchdog.json")),
                    "DRR sleeper acquired a progress watchdog",
                    output,
                )
            if fault["point"] == "checkpoint":
                import torch

                from seis_interp.training.rehearsal import state_sha256

                log = (output / "execution.log").read_text().split("resumed step=300", 1)[-1]
                _require(
                    not re.search(r"\bstep \d+/", log),
                    "predict-only retry trained again",
                    output / "execution.log",
                )
                checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
                resumed = torch.load(previous / "resume.pt", map_location="cpu", weights_only=False)
                # CCNet final weights are the primary EMA, saved in the recovery training state.
                averages = resumed["training"]["averaged"]
                parameters = (
                    checkpoint["state_dict"]
                    if "state_dict" in checkpoint
                    else checkpoint["model_state_dict"]
                )
                model_config = yaml.safe_load((output / "config.resolved.yaml").read_text())
                from seis_interp.training.forge_mvp_methods import initialize_model

                names = [name for name, _ in initialize_model(model_config).named_parameters()]
                _require(
                    state_sha256([parameters[name] for name in names]) == state_sha256(averages),
                    "predict-only final EMA differs from saved EMA",
                    output / "final.pt",
                    previous / "resume.pt",
                )
            evidence.append(event)
        cpu = [
            (path, row)
            for path, row in _all_attempts(experiment)
            if row.get("method_id") in ("pocs_grid", "drr_grid")
        ]
        _require(
            cpu and all(row.get("nice") == 10 for _, row in cpu),
            "CPU workers must use nice 10",
            experiment / "method_runs",
        )
    return evidence


def _check_divergence(experiments):
    c3, forge = experiments
    output = c3 / "method_runs/nersi_profile/S1/a01_b01"
    _require(
        _json(output / "manifest.json")["status"] == "diverged"
        and not (output / "attempts").exists(),
        "planned C3 divergence must not retry",
        output,
    )
    selection = c3 / "workers/nersi_profile_selection.json"
    _require(
        _json(selection)["winner"]["id"] == "a02_b04",
        "C3 divergence was not ranked last",
        selection,
    )
    summary = forge / "nersi_profile_summary.csv"
    table = pd.read_csv(summary)
    _require(
        table.seed_count.eq(2).all()
        and table.diverged_seeds.astype(str).eq("990672").all()
        and table["mean"].isna().all()
        and table.sample_std.isna().all(),
        "FORGE divergence summary is incomplete",
        summary,
    )
    return [output / "manifest.json", selection, summary]


def _check_barrier(experiments, samples):
    evidence = []
    for experiment in experiments:
        marker = experiment / "workers/s1_barrier.json"
        _require(
            len(list(experiment.rglob("s1_barrier.json"))) == 1,
            "experiment must have one barrier marker",
            marker,
        )
        passed = _time(_json(marker)["passed_at"])
        s1 = [(path, row) for path, row in _records(experiment) if row["stage"] == "S1"]
        _require(
            s1
            and all(
                _end(row) <= passed
                and row["test_scoring_calls"] == 0
                and not (path / "test_per_trace.parquet").exists()
                for path, row in s1
            ),
            "S1 scoring crossed the barrier",
            marker,
        )
        seeds = [(path, row) for path, row in _records(experiment) if row["stage"] == "evaluation"]
        methods = set(_json(experiment / "experiment.json")["methods"]) - {"pocs_grid", "drr_grid"}
        for method in methods:
            _require(
                any(
                    _time(row["prediction_finalized_at"]) < passed
                    for _, row in seeds
                    if row["method_id"] == method and "prediction_finalized_at" in row
                ),
                "no pre-barrier neural seed prediction",
                marker,
                experiment / "method_runs" / method,
            )
        disclosure = {}
        for path, row in _all_attempts(experiment):
            log = path / "execution.log"
            if not log.exists():
                continue
            reads = [
                line
                for line in log.read_text().splitlines()
                if "reading test targets seed=" in line
            ]
            _require(
                all(_time(line.split()[0]) >= passed for line in reads),
                "test targets read before barrier",
                log,
            )
            if reads:
                seed = row.get("trace_seed", row.get("region_seed"))
                key = f"{row['method_id']}/seed_{seed}"
                disclosure[key] = disclosure.get(key, 0) + len(reads)
        observations = [sample["experiments"].get(str(experiment), {}) for sample in samples]
        hashes = {value["barrier_sha256"] for value in observations if value.get("barrier_sha256")}
        _require(hashes == {sha256_file(marker)}, "barrier marker changed after passing", marker)
        events = _jsonl(experiment / "workers/rehearsal_events.jsonl")
        _require(
            all(
                row.get("s1_completed") is not False
                for row in events
                if _time(row["observed_at"]) >= passed
            ),
            "worker regressed S1 completion",
            experiment / "workers/rehearsal_events.jsonl",
        )
        waiting_samples = 0
        for before, after in zip(samples, samples[1:], strict=False):
            first = before["experiments"].get(str(experiment), {})
            second = after["experiments"].get(str(experiment), {})
            elapsed = after["monotonic"] - before["monotonic"]
            for state in first.get("states", {}).values():
                if state.get("phase") != "waiting_for_all_S1":
                    continue
                pid = state.get("pid")
                if not any(
                    value.get("pid") == pid and value.get("phase") == "waiting_for_all_S1"
                    for value in second.get("states", {}).values()
                ):
                    continue
                left = next((p for p in first.get("processes", []) if p["pid"] == pid), None)
                right = next((p for p in second.get("processes", []) if p["pid"] == pid), None)
                if left and right and left["start_ticks"] == right["start_ticks"]:
                    cpu = (right["cpu_ticks"] - left["cpu_ticks"]) / before["clock_ticks"] / elapsed
                    _require(cpu < 0.05, "barrier waiter used at least 5% CPU", marker)
                    waiting_samples += 1
        _require(waiting_samples > 0, "missing barrier-wait CPU samples", marker)
        for index, fault in enumerate(Rehearsal(experiment).plan["faults"]):
            if fault["action"] == "barrier_hold":
                hold = experiment / "rehearsal_faults" / f"{index:03d}.json"
                released = _json(hold)
                _require(
                    set(released["seed_predictions"]) == methods
                    and all(
                        _time(at) <= _time(released["released_at"]) < passed
                        for at in released["seed_predictions"].values()
                    ),
                    "barrier hold released without pre-barrier neural predictions",
                    hold,
                    marker,
                )
            if fault["point"] == "after_test_scored":
                output = _fault_output(experiment, fault)
                _require(
                    sha256_file(output / "test_per_trace.parquet")
                    == sha256_file(output / "attempts/1/test_per_trace.parquet"),
                    "post-crash scoring bytes changed",
                    output,
                )
                _require(
                    disclosure.get(f"{fault['method']}/{fault['run']}") == 2,
                    "post-scoring crash must disclose two target reads",
                    output,
                )
        evidence.append({"path": str(marker), "test_reads_per_seed": disclosure})
    return evidence


def _check_summaries(experiments, samples):
    evidence = []
    for experiment in experiments:
        for name in ("summary.csv", "seed_metrics.csv"):
            path = experiment / name
            table = pd.read_csv(path)
            _require(not table.empty, "missing aggregate rows", path)
            evidence.append(path)
        summary = pd.read_csv(experiment / "summary.csv")
        methods = set(_json(experiment / "experiment.json")["methods"])
        _require(
            set(summary.method_id) == methods,
            "summary is missing an active method",
            experiment / "summary.csv",
        )
        for method, group in summary.groupby("method_id"):
            expected = 2 if experiment == experiments[1] and method == "nersi_profile" else 3
            _require(
                group.seed_count.eq(expected).all(),
                "wrong summary seed count",
                experiment / "summary.csv",
            )
            if experiment == experiments[1]:
                _require(
                    (experiment / f"{method}_summary.csv").exists(),
                    "missing FORGE method summary",
                    experiment,
                )
            selection = experiment / "workers" / f"{method}_selection.json"
            _require(
                _json(selection).get("test_used_for_selection") is False,
                "test used for selection",
                selection,
            )
        _require(
            _json(experiment / "workers/supervisor.json")["status"] == "completed",
            "supervisor did not complete",
            experiment / "workers/supervisor.json",
        )
        completed = [sample["experiments"].get(str(experiment), {}) for sample in samples]
        completed = [
            value
            for value in completed
            if value.get("states", {}).get("supervisor", {}).get("status") == "completed"
        ]
        _require(completed, "no completed supervisor sample", experiment)
        for value in completed:
            for name, digest in value["summary_sha256"].items():
                _require(
                    sha256_file(experiment / name) == digest,
                    "summary bytes changed after completion/relaunch",
                    experiment / name,
                )
    # The operator captures this around the supervisor-loss drill.
    c3 = experiments[0]
    before = _json(c3 / "rehearsal_evidence/before_refusal.json")
    refused = _json(c3 / "rehearsal_evidence/after_refusal.json")
    refusal_log = c3 / "refused_supervisor.out"
    _require(
        "refusing duplicate live worker" in refusal_log.read_text(),
        "missing ordinary-run refusal evidence",
        refusal_log,
    )
    _require(
        before["worker_sha256"] == refused["worker_sha256"],
        "refused duplicate launch changed worker states",
        c3 / "rehearsal_evidence",
    )
    before_relaunch = _json(c3 / "rehearsal_evidence/before_relaunch.json")
    after_relaunch = _json(c3 / "rehearsal_evidence/after_relaunch.json")
    _require(
        before_relaunch["summary_sha256"]
        and before_relaunch["summary_sha256"] == after_relaunch["summary_sha256"],
        "summary bytes changed after drill relaunch",
        c3 / "rehearsal_evidence",
    )
    return evidence


def _check_imports(imports):
    _require(imports, "R0 replay evidence was not supplied with --imports")
    evidence = []
    for experiment in imports:
        coverage = _json(experiment / "import_coverage.json")
        for method, info in coverage["methods"].items():
            for candidate in info["s1_sources"]:
                output = experiment / "method_runs" / method / "S1" / candidate
                row = _json(output / "manifest.json")
                source = row.get("imported_from")
                _require(
                    row["status"] == "completed" and source and not row.get("cuda_admission"),
                    "predicted import was retrained or failed",
                    output,
                )
                original = Path(source["source_path"])
                if method not in ("pocs_grid", "drr_grid") or (original / "final.pt").exists():
                    _require(
                        sha256_file(original / "final.pt") == sha256_file(output / "final.pt"),
                        "import checkpoint differs",
                        output / "final.pt",
                    )
                # Compare primary validation metric internally; never display values.
                previous = _json(original / "manifest.json")

                def primary(record):
                    values = record["metrics"]["validation"]
                    return values.get("clean_target", values)["masked_trace_relative_mse"]

                _require(
                    np.isclose(primary(row), primary(previous), rtol=1e-10, atol=1e-12),
                    "import rescoring differs from source",
                    output / "manifest.json",
                )
                evidence.append(output / "manifest.json")
    _require(evidence, "R0 did not replay any importable cells", *imports)
    return evidence


def capture_drill_snapshot(experiment, label):
    directory = experiment / "rehearsal_evidence"
    directory.mkdir(exist_ok=True)
    workers = {}
    for path in (experiment / "workers").glob("*.json"):
        state = _json(path)
        if path.name != "supervisor.json" and "status" in state:
            # Workers continue normally while a duplicate launch refuses. Phase,
            # counters, status and seed-child lists can change without replacement.
            identity = {
                key: state.get(key)
                for key in (
                    "pid",
                    "process_start_time",
                    "method_id",
                    "source_root",
                    "shard",
                    "restartable",
                )
            }
            workers[path.name] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
        else:
            workers[path.name] = sha256_file(path)
    data = {
        "worker_sha256": workers,
        "summary_sha256": {path.name: sha256_file(path) for path in experiment.glob("*.csv")},
    }
    path = directory / f"{label}.json"
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
    return path


def check_rehearsal_plan(study):
    audit = _json(study / "rehearsal_generation.json")
    _require(
        audit["burned_evaluation_seeds"] == [990671, 990672, 990673], "wrong burned seeds", study
    )
    evidence = []
    for dataset in ("c3", "forge"):
        folder = study / dataset
        contract = yaml.safe_load((folder / "config.yaml").read_text())
        plan = load_rehearsal_plan(folder, contract)
        grid = load_experiment_grid(folder, contract)
        for method in contract.get("methods", grid["methods"]):
            base = yaml.safe_load((folder / f"{method}.yaml").read_text())
            options = rehearsal_candidates(
                c3_search_candidates(base, grid["methods"][method]), contract, method, plan
            )
            _require(
                len(options) == (25 if audit["variant"] == "imports" else 2),
                "unexpected rehearsal subset size",
                folder / "rehearsal.yaml",
            )
        for name in ("grid.yaml", "pocs_grid.yaml"):
            path = folder / name
            if path.exists():
                _require(
                    sha256_file(path) == audit["source_yaml_sha256"][f"{dataset}/{name}"],
                    "immutable YAML changed",
                    path,
                )
        evidence.append(folder / "rehearsal.yaml")
    return evidence


def check_study067_rehearsal(
    c3, forge, gpu_log, *, twins=(), determinism_twins=(), imports=(), launch_commit=None
):
    experiments = [Path(c3).resolve(), Path(forge).resolve()]
    samples_error = None
    try:
        samples = _jsonl(gpu_log)
    except (OSError, ValueError) as error:
        samples, samples_error = [], str(error)
    if launch_commit is None:
        launch_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    checks = [
        (
            "Freeze and launch",
            lambda: [
                item
                for experiment in experiments
                for item in _check_freeze(experiment, samples, launch_commit)
            ],
        ),
        ("Admission and sharding", lambda: _check_admission(experiments, samples)),
        ("Warm-up", lambda: _check_warmup(experiments)),
        (
            "Resume",
            lambda: _check_resume(
                experiments,
                [Path(path).resolve() for path in twins],
                [Path(path).resolve() for path in determinism_twins],
            ),
        ),
        ("Specific faults", lambda: _check_specific_faults(experiments, samples)),
        ("Divergence", lambda: _check_divergence(experiments)),
        ("Barrier", lambda: _check_barrier(experiments, samples)),
        ("Summaries", lambda: _check_summaries(experiments, samples)),
        ("R0 (imports)", lambda: _check_imports([Path(path).resolve() for path in imports])),
    ]
    results = []
    for name, check in checks:
        try:
            paths = check()
            results.append(
                {
                    "area": name,
                    "passed": True,
                    "evidence": [str(path) if isinstance(path, Path) else path for path in paths],
                }
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            subprocess.SubprocessError,
        ) as error:
            results.append(
                {
                    "area": name,
                    "passed": False,
                    "reason": str(error),
                    "evidence": [str(path) for path in experiments]
                    + ([str(gpu_log), samples_error] if samples_error else []),
                }
            )
    return results
