"""Copy the frozen common-grid study with an explicit, audited rehearsal diff."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.experiment_commonality import flatten_settings
from seis_interp.training.rehearsal import load_rehearsal_plan

SUBSETS = {
    "regsi_real": ["a01_b05", "a05_b01"],
    "ccnet5d_grid": ["a05_b02", "a01_b01"],
    "nersi_profile": ["a02_b04", "a01_b01"],
    "pocs_grid": ["a05_b05", "a04_b05"],
    "drr_grid": ["a01_b01", "a02_b01"],
}


def rehearsal_faults(dataset):
    def fault(method, run, point, action, **conditions):
        return {
            "method": method,
            "run": run,
            "point": point,
            "action": action,
            "attempt": 0,
            **conditions,
        }

    if dataset == "c3":
        return [
            fault("regsi_real", "a01_b05", "report", "exit", step=250),
            fault("regsi_real", "a05_b01", "pre_admission", "resource_wait"),
            fault("ccnet5d_grid", "a05_b02", "report", "oom", step=250),
            fault("nersi_profile", "a02_b04", "report", "exit", step=250),
            fault("nersi_profile", "a01_b01", "report", "nonfinite", step=150),
            fault("nersi_profile", "seed_990672", "after_test_scored", "exit"),
            fault("drr_grid", "a01_b01", "pre_method", "barrier_hold", seconds=21600),
        ]
    return [
        fault("regsi_real", "a05_b01", "report", "sleep", step=250, seconds=1800),
        fault("regsi_real", "seed_990672", "after_test_scored", "exit"),
        fault("ccnet5d_grid", "a05_b05", "report", "sticky_cuda", step=250),
        fault("ccnet5d_grid", "a05_b02", "checkpoint", "infra", phase="before_prediction"),
        fault("nersi_profile", "seed_990672", "after_prediction", "nan_prediction"),
        fault("drr_grid", "a01_b01", "pre_method", "barrier_hold", seconds=21600),
    ]


def _write_allowed_edits(path, edits, changes, relative):
    original = yaml.safe_load(path.read_text())
    updated = deepcopy(original)
    for key, value in edits.items():
        target = updated
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    # Compare leaf values for exact allowlist enforcement, including list values.
    for key in edits:
        first, second = original, updated
        for part in key.split("."):
            first = first.get(part) if isinstance(first, dict) else None
            second = second[part]
        if first != second:
            changes.append({"file": relative, "key": key, "before": first, "after": second})
    path.write_text(yaml.safe_dump(updated, sort_keys=False))


def make_study067_rehearsal(source, out, *, no_faults=False, imports_only=False, determinism=False):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source or source in out.parents:
        raise ValueError("rehearsal output must be outside the frozen study")
    if out.exists():
        raise FileExistsError(f"refusing to replace rehearsal output: {out}")
    if sum((no_faults, imports_only, determinism)) > 1:
        raise ValueError("choose --no-faults, --imports-only or --determinism")
    changes = []
    hashes = {str(path.relative_to(source)): sha256_file(path) for path in source.rglob("*.yaml")}
    shutil.copytree(source, out)
    for dataset in ("c3", "forge"):
        folder = out / dataset
        edits = {
            "study_id": "study_067_forge_c3_common_grid_rehearsal",
            "trace_split.evaluation_seeds": [990671, 990672, 990673],
        }
        if not imports_only:
            edits["matched_training.max_steps"] = 300
            if dataset == "forge":
                edits["method_execution.ccnet5d_grid.worker_shards"] = 2
        _write_allowed_edits(folder / "config.yaml", edits, changes, f"{dataset}/config.yaml")
        if not imports_only:
            for method in ("regsi_real", "ccnet5d_grid", "nersi_profile"):
                method_edits = {"training.max_steps": 300, "training.report_interval": 50}
                if determinism and method == "ccnet5d_grid":
                    method_edits["execution.cudnn_benchmark"] = False
                _write_allowed_edits(
                    folder / f"{method}.yaml",
                    method_edits,
                    changes,
                    f"{dataset}/{method}.yaml",
                )
            _write_allowed_edits(
                folder / "drr_grid.yaml",
                {"algorithm.n_iterations": 2},
                changes,
                f"{dataset}/drr_grid.yaml",
            )
        subsets = deepcopy(SUBSETS)
        if dataset == "forge":
            subsets["ccnet5d_grid"] = ["a05_b02", "a05_b05"]
        plan = (
            {"faults": []}
            if imports_only
            else {
                "resume_interval_updates": 100,
                "watchdog": {"floor_seconds": 900, "expected_report_seconds": 60},
                "candidates": subsets,
                "faults": [] if no_faults or determinism else rehearsal_faults(dataset),
            }
        )
        (folder / "rehearsal.yaml").write_text(yaml.safe_dump(plan, sort_keys=False))
        load_rehearsal_plan(folder, yaml.safe_load((folder / "config.yaml").read_text()))
        changes.append({"file": f"{dataset}/rehearsal.yaml", "key": "<new file>", "after": plan})
    # Immutable grid, budgets, imports and all other YAML leaves must be identical.
    allowed = {}
    for change in changes:
        allowed.setdefault(change["file"], set()).add(change["key"])
    for name, digest in hashes.items():
        copied = out / name
        if name not in allowed:
            if sha256_file(copied) != digest:
                raise ValueError(f"unexpected rehearsal change: {name}")
            continue
        before = flatten_settings(yaml.safe_load((source / name).read_text()))
        after = flatten_settings(yaml.safe_load(copied.read_text()))
        for key in before.keys() | after.keys():
            if before.get(key) != after.get(key) and not any(
                key == edit or key.startswith(edit + ".") for edit in allowed[name]
            ):
                raise ValueError(f"unexpected rehearsal change: {name}:{key}")
    audit = {
        "source": str(source),
        "source_yaml_sha256": hashes,
        "changes": changes,
        "burned_evaluation_seeds": [990671, 990672, 990673],
        "variant": "imports"
        if imports_only
        else "determinism"
        if determinism
        else "twin"
        if no_faults
        else "faults",
    }
    (out / "rehearsal_generation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
