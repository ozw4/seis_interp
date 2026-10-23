"""Sealed rehearsal guards, candidate subsets, faults and barrier holds."""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.training.candidate_schedule import ordered_candidates

POINTS = {
    "report",
    "checkpoint",
    "pre_admission",
    "pre_method",
    "after_prediction",
    "after_test_scored",
}
ACTIONS = {
    "exit",
    "oom",
    "sticky_cuda",
    "infra",
    "resource_wait",
    "nonfinite",
    "nan_prediction",
    "sleep",
    "barrier_hold",
}


def _contract_seeds(value):
    if not isinstance(value, dict):
        return []
    seeds = []
    for key, child in value.items():
        if key.endswith("seed") and type(child) is int:
            seeds.append(child)
        elif key.endswith("seeds") and isinstance(child, list):
            seeds.extend(seed for seed in child if type(seed) is int)
        elif isinstance(child, dict):
            seeds.extend(_contract_seeds(child))
    return seeds


def load_rehearsal_plan(study, contract):
    """Validate isolation before preparing inputs or creating an experiment."""
    path = study / "rehearsal.yaml"
    enabled = path.exists()
    split = contract.get("trace_split", contract.get("region_split", {}))
    seeds = split.get("evaluation_seeds", [])
    if enabled:
        if not contract["study_id"].endswith("_rehearsal"):
            raise ValueError("rehearsal.yaml requires a study_id ending in _rehearsal")
        if (
            len(seeds) != 3
            or len(set(seeds)) != 3
            or any(type(seed) is not int or not 990000 <= seed <= 990999 for seed in seeds)
        ):
            raise ValueError(
                "rehearsal evaluation seeds must be three distinct seeds in [990000, 990999]"
            )
        plan = yaml.safe_load(path.read_text())
        validate_rehearsal_plan(plan)
        return plan
    if contract["study_id"].endswith("_rehearsal"):
        raise ValueError("a rehearsal study requires rehearsal.yaml")
    if any(990000 <= seed <= 990999 for seed in _contract_seeds(contract)):
        raise ValueError("non-rehearsal studies cannot use reserved rehearsal seeds")
    if any(
        any(part.endswith("_rehearsal") for part in Path(entry).parts)
        for entry in contract.get("import_from", [])
    ):
        raise ValueError("non-rehearsal studies cannot import from *_rehearsal")
    return None


def validate_rehearsal_plan(plan):
    if not isinstance(plan, dict) or set(plan) - {
        "candidates",
        "faults",
        "resume_interval_updates",
        "watchdog",
    }:
        raise ValueError("invalid rehearsal plan keys")
    interval = plan.get("resume_interval_updates", 2500)
    if type(interval) is not int or interval <= 0:
        raise ValueError("resume_interval_updates must be a positive integer")
    watchdog = plan.get("watchdog", {})
    if not isinstance(watchdog, dict) or set(watchdog) - {
        "floor_seconds",
        "expected_report_seconds",
    }:
        raise ValueError("invalid rehearsal watchdog keys")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in watchdog.values()
    ):
        raise ValueError("rehearsal watchdog intervals must be positive and finite")
    subsets = plan.get("candidates", {})
    if not isinstance(subsets, dict):
        raise ValueError("rehearsal candidates must map methods to IDs")
    for ids in subsets.values():
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(name, str) for name in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("rehearsal candidate IDs must be nonempty and distinct")
    faults = plan.get("faults", [])
    if not isinstance(faults, list):
        raise ValueError("rehearsal faults must be a list")
    for fault in faults:
        if not isinstance(fault, dict) or set(fault) - {
            "point",
            "method",
            "run",
            "attempt",
            "step",
            "phase",
            "action",
            "seconds",
        }:
            raise ValueError("invalid rehearsal fault keys")
        if fault.get("point") not in POINTS or fault.get("action") not in ACTIONS:
            raise ValueError("unknown rehearsal fault point or action")
        if not all(isinstance(fault.get(key), str) and fault[key] for key in ("method", "run")):
            raise ValueError("rehearsal fault requires method and run")
        if type(fault.get("attempt", 0)) is not int or fault.get("attempt", 0) < 0:
            raise ValueError("rehearsal fault attempt must be nonnegative")
        if "step" in fault and (type(fault["step"]) is not int or fault["step"] < 1):
            raise ValueError("rehearsal fault step must be positive")
        if fault["action"] in ("sleep", "barrier_hold") and (
            not isinstance(fault.get("seconds"), (int, float))
            or not math.isfinite(fault["seconds"])
            or fault["seconds"] <= 0
        ):
            raise ValueError(f"rehearsal {fault['action']} seconds must be positive and finite")
        if fault["action"] == "barrier_hold" and (
            fault["point"] != "pre_method" or fault["method"] != "drr_grid"
        ):
            raise ValueError("rehearsal barrier_hold requires DRR pre_method")


def rehearsal_candidates(candidates, contract, method, plan):
    """Order the full grid first, then filter without requiring priority IDs in a subset."""
    settings = contract.get("method_execution", {}).get(method, {})
    ordered = ordered_candidates(candidates, settings.get("candidate_priority", []))
    if plan is None:
        return ordered
    ids = plan.get("candidates", {}).get(method)
    if ids is None:
        return ordered
    if set(ids) - {item["id"] for item in ordered}:
        raise ValueError(f"{method}: unknown rehearsal candidate ID")
    return [item for item in ordered if item["id"] in ids]


def rehearsal_seal(study, plan, counts):
    if plan is None:
        return {"rehearsal": False}
    if set(plan.get("candidates", {})) - set(counts):
        raise ValueError("rehearsal subset names an inactive method")
    for fault in plan.get("faults", []):
        if fault["method"] not in counts:
            raise ValueError("rehearsal fault names an inactive method")
        subset = plan.get("candidates", {}).get(fault["method"])
        if (
            subset is not None
            and not fault["run"].startswith("seed_")
            and fault["run"] not in subset
        ):
            raise ValueError("rehearsal fault run is outside the candidate subset")
    count = next(iter(counts.values())) if len(set(counts.values())) == 1 else counts
    return {
        "rehearsal": True,
        "rehearsal_plan_sha256": sha256_file(study / "rehearsal.yaml"),
        "s1_runs_per_method": count,
        "total_runs": sum(counts.values()) + 3 * len(counts),
    }


class Rehearsal:
    """A missing/false sealed flag cannot enable hooks or runtime overrides."""

    def __init__(self, experiment):
        self.experiment = Path(experiment)
        self.plan = None
        seal_path = self.experiment / "experiment.json"
        seal = json.loads(seal_path.read_text()) if seal_path.exists() else {}
        if seal.get("rehearsal") is True:
            path = self.experiment / "study/rehearsal.yaml"
            if sha256_file(path) != seal["rehearsal_plan_sha256"]:
                raise ValueError("sealed rehearsal plan hash differs")
            contract = yaml.safe_load((self.experiment / "study/config.yaml").read_text())
            self.plan = load_rehearsal_plan(path.parent, contract)

    def candidates(self, candidates, contract, method):
        return rehearsal_candidates(candidates, contract, method, self.plan)

    def recovery(self, output, method, run, attempt, report):
        from seis_interp.training.exact_resume import ExactResume

        options = (
            {} if self.plan is None else {"updates": self.plan.get("resume_interval_updates", 2500)}
        )
        return ExactResume(
            output / "resume.pt",
            **options,
            reporter=report,
            checkpoint=lambda step, phase: self.inject(
                "checkpoint", method, run, attempt, step=step, phase=phase
            ),
            record_all_steps=self.plan is not None,
        )

    def watchdog(self, output, config, *, resumable):
        from copy import deepcopy

        from seis_interp.training.run_recovery import run_watchdog

        if self.plan is not None:
            config = deepcopy(config)
            values = self.plan.get("watchdog", {})
            config.setdefault("execution", {}).update(
                {
                    key: values[name]
                    for key, name in (
                        ("watchdog_floor_seconds", "floor_seconds"),
                        ("expected_report_seconds", "expected_report_seconds"),
                    )
                    if name in values
                }
            )
        return run_watchdog(output, config, resumable=resumable)

    def report(self, message, method, run, attempt):
        if self.plan is None:
            return
        match = re.search(r"\bstep (\d+)/(\d+)", message)
        if match:
            self.inject("report", method, run, attempt, step=int(match[1]))

    def inject(self, point, method, run, attempt, step=None, phase=None, result=None):
        if self.plan is None:
            return
        for index, fault in enumerate(self.plan.get("faults", [])):
            if (fault["point"], fault["method"], fault["run"]) != (
                point,
                method,
                run,
            ):
                continue
            holding = fault["action"] == "barrier_hold"
            if not holding and fault.get("attempt", 0) != attempt:
                continue
            if any(
                key in fault and fault[key] != value
                for key, value in (("step", step), ("phase", phase))
            ):
                continue
            directory = self.experiment / "rehearsal_faults"
            directory.mkdir(exist_ok=True)
            path = directory / f"{index:03d}.json"
            from seis_interp.data.forge_mvp_run_records import utc_now
            from seis_interp.training.worker_processes import source_root

            try:
                with path.open("x") as stream:
                    json.dump(
                        fault
                        | {
                            "fired_at": utc_now(),
                            "pid": os.getpid(),
                            "attempt": attempt,
                            "source_root": source_root(),
                            "step": step,
                            "phase": phase,
                        },
                        stream,
                        sort_keys=True,
                    )
            except FileExistsError:
                if not holding:
                    continue
            print(
                f"rehearsal fault={index} action={fault['action']} "
                f"point={point} run={run} attempt={attempt}",
                flush=True,
            )
            if holding:
                # Re-enter on retries: an interrupted or timed-out hold must
                # never turn its existing fault marker into permission to pass.
                predictions = _wait_for_seed_predictions(self.experiment, fault["seconds"])
                from seis_interp.training.run_recovery import write_run_manifest

                write_run_manifest(
                    path,
                    json.loads(path.read_text())
                    | {"released_at": utc_now(), "seed_predictions": predictions},
                )
            else:
                _fault_action(fault, result)


def _wait_for_seed_predictions(experiment, timeout):
    """Hold DRR S1 until every neural method has a finalized seed prediction."""
    seal = json.loads((experiment / "experiment.json").read_text())
    methods = set(seal["methods"]) - {"pocs_grid", "drr_grid"}
    if not methods:
        raise ValueError("barrier_hold requires neural methods")
    deadline = time.monotonic() + timeout
    while True:
        predictions = {}
        for method in sorted(methods):
            for path in sorted(
                (experiment / "method_runs" / method / "evaluation").glob("*/manifest.json")
            ):
                try:
                    row = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if (
                    row.get("status") in ("awaiting_barrier", "completed")
                    and row.get("prediction_finalized_at")
                    and (path.parent / "prediction.npy").is_file()
                ):
                    predictions[method] = row["prediction_finalized_at"]
                    break
        if predictions.keys() == methods:
            return predictions
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "rehearsal barrier_hold timed out awaiting seed predictions: "
                + ", ".join(sorted(methods - predictions.keys()))
            )
        time.sleep(min(10, remaining))


def _fault_action(fault, result):
    import torch

    from seis_interp.training.run_recovery import ResourceWait

    action = fault["action"]
    if action == "exit":
        os._exit(137)
    if action == "oom":
        raise torch.cuda.OutOfMemoryError("rehearsal injected out of memory")
    if action == "infra":
        raise OSError("rehearsal injected infrastructure error")
    if action == "resource_wait":
        raise ResourceWait("rehearsal injected resource wait")
    if action == "nonfinite":
        raise FloatingPointError("non-finite loss")
    if action == "nan_prediction":
        result.prediction.flat[0] = float("nan")
    if action == "sleep":
        time.sleep(fault["seconds"])
    if action == "sticky_cuda":
        device = torch.device("cuda", torch.cuda.current_device())
        values = torch.zeros(1, device=device)
        values[torch.tensor([2], device=device)] = 1
        torch.cuda.synchronize(device)


def state_sha256(value):
    """Canonical tensor/value digest independent of pickle storage identities."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, np.random.Generator):
            visit(item.bit_generator.state)
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest()
