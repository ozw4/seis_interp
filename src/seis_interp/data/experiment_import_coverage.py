"""Read-only reuse audit for a prepared grid experiment, including possible seed winners."""

import json

import yaml

from seis_interp.data.c3_region_inputs import load_c3_region_inputs
from seis_interp.data.experiment_imports import find_prediction_import, saved_input_fingerprint
from seis_interp.data.forge_grid_inputs import load_forge_grid_inputs
from seis_interp.processing.c3_region_search import (
    c3_search_candidates,
    c3_split_settings,
    resolve_c3_candidate,
)
from seis_interp.training.rehearsal import Rehearsal


def experiment_import_coverage(repo, experiment, dataset):
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    grid = yaml.safe_load((experiment / "study/grid.yaml").read_text())
    _, split = c3_split_settings(contract)
    rehearsal = Rehearsal(experiment)
    result = {}
    for method in grid["methods"]:
        base = yaml.safe_load((experiment / "study" / f"{method}.yaml").read_text())
        candidates = rehearsal.candidates(
            c3_search_candidates(base, grid["methods"][method]), contract, method
        )
        matches = {}
        for seed in [split["search_seed"], *split["evaluation_seeds"]]:
            stage = "S1" if seed == split["search_seed"] else "evaluation"
            domains = ("validation",) if stage == "S1" else ("validation", "test")
            if dataset == "forge":
                inputs, loaded, _ = load_forge_grid_inputs(experiment, method, seed, domains)
                alignment = {"alignment": loaded["alignment"]}
            else:
                inputs, _ = load_c3_region_inputs(experiment, seed, domains)
                alignment = {}
            fingerprint = saved_input_fingerprint(experiment, seed, method, domains)
            matches[seed] = {}
            for candidate in candidates:
                config = resolve_c3_candidate(
                    candidate["config"] | alignment,
                    len(inputs.observed_cells),
                    contract["matched_training"]["loss_traces_per_update"],
                )
                match = find_prediction_import(
                    repo, contract.get("import_from", []), config, fingerprint, seed, stage
                )
                if match is not None:
                    matches[seed][candidate["id"]] = match["source_path"]
        seeds_by_candidate = {
            candidate["id"]: [
                seed for seed in split["evaluation_seeds"] if candidate["id"] in matches[seed]
            ]
            for candidate in candidates
        }
        result[method] = {
            "s1_importable": len(matches[split["search_seed"]]),
            "s1_total": len(candidates),
            "evaluation_importable_max": max(map(len, seeds_by_candidate.values())),
            "evaluation_total": 3,
            "evaluation_condition": "depends on the S1 winner; see evaluation_seeds_by_candidate",
            "evaluation_seeds_by_candidate": {
                key: value for key, value in seeds_by_candidate.items() if value
            },
            "s1_sources": matches[split["search_seed"]],
        }
    return {
        "dataset": dataset,
        "rehearsal": rehearsal.plan is not None,
        "methods": result,
        "experiment": str(experiment),
        "targets_read_for_scoring": 0,
    }


def check_grid_prepare(repo, study, dataset):
    """Use temporary CPU preparation outside runs/, then report import matches."""
    import tempfile
    from pathlib import Path

    from seis_interp.data.c3_region_inputs import prepare_c3_region_experiment
    from seis_interp.data.forge_grid_inputs import prepare_forge_grid_experiment

    prepare = prepare_c3_region_experiment if dataset == "c3" else prepare_forge_grid_experiment
    with tempfile.TemporaryDirectory(prefix=f"seis_{dataset}_prepare_check_") as directory:
        experiment = prepare(repo, study, destination=Path(directory) / "experiment")
        coverage = experiment_import_coverage(repo, experiment, dataset)
        coverage["experiment"] = "temporary check-only preparation (removed)"
    print(json.dumps(coverage, indent=2, ensure_ascii=False))


def s1_candidates_without_admission(repo, experiment, candidates, dataset):
    """Identify cached and importable S1 cells before consulting any GPU state."""
    ready = set()
    pending = []
    for candidate in candidates:
        method = candidate["config"]["method_id"]
        path = experiment / "method_runs" / method / "S1" / candidate["id"] / "manifest.json"
        if path.exists() and json.loads(path.read_text())["status"] in ("completed", "diverged"):
            ready.add(candidate["id"])
        else:
            pending.append(candidate)
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    if not pending or not contract.get("import_from"):
        return ready
    _, split = c3_split_settings(contract)
    seed = split["search_seed"]
    method = pending[0]["config"]["method_id"]
    domains = ("validation",)
    if dataset == "forge":
        inputs, base, _ = load_forge_grid_inputs(experiment, method, seed, domains)
        alignment = {"alignment": base["alignment"]}
    else:
        inputs, _ = load_c3_region_inputs(experiment, seed, domains)
        alignment = {}
    fingerprint = saved_input_fingerprint(experiment, seed, method, domains)
    for candidate in pending:
        config = resolve_c3_candidate(
            candidate["config"] | alignment,
            len(inputs.observed_cells),
            contract["matched_training"]["loss_traces_per_update"],
        )
        if (
            find_prediction_import(repo, contract["import_from"], config, fingerprint, seed, "S1")
            is not None
        ):
            ready.add(candidate["id"])
    return ready
