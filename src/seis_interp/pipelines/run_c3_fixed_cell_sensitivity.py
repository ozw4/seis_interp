"""Evaluate the pre-registered external CCNet winner with production C3 scoring."""

import fcntl
import json
import os
from copy import deepcopy

import yaml

from seis_interp.data.experiment_imports import method_config_hash
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes
from seis_interp.evaluation.c3_region_results import summarize_c3_region_seeds
from seis_interp.pipelines.run_c3_region_experiment import execute_c3_region_run
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.training.worker_processes import process_start_time, write_worker_state

METHOD = "ccnet5d_grid"
CANDIDATE = "a05_b02"


def validate_fixed_cell_sensitivity(repo, experiment, production):
    """Check the completed production barrier and identical inputs, read-only."""
    if experiment.resolve() == production.resolve():
        raise ValueError("sensitivity must use a separate prepared experiment")
    seals = [
        json.loads((root / "experiment.json").read_text()) for root in (experiment, production)
    ]
    if any(seal.get("rehearsal") or seal.get("continuing_experiment") for seal in seals):
        raise ValueError("fixed-cell sensitivity requires native non-rehearsal experiments")
    if seals[0].get("methods") != [METHOD]:
        raise ValueError("sensitivity requires only CCNet")
    barrier = json.loads((production / "workers/s1_barrier.json").read_text())
    if barrier.get("passed") is not True or set(barrier.get("methods", [])) != set(
        seals[1]["methods"]
    ):
        raise ValueError("production S1 barrier has not passed")
    verify_hashes(experiment, seals[0]["artifacts"])
    input_hashes = {
        name: digest for name, digest in seals[1]["artifacts"].items() if name.startswith("input/")
    }
    if (
        input_hashes
        != {
            name: digest
            for name, digest in seals[0]["artifacts"].items()
            if name.startswith("input/")
        }
        or seals[0]["input_hashes"] != seals[1]["input_hashes"]
    ):
        raise ValueError("sensitivity inputs differ from production")
    verify_hashes(production, input_hashes)
    contracts = [
        yaml.safe_load((root / "study/config.yaml").read_text())
        for root in (experiment, production)
    ]
    if contracts[0]["trace_split"] != contracts[1]["trace_split"] or contracts[0]["trace_split"][
        "evaluation_seeds"
    ] != [7321, 7322, 7323]:
        raise ValueError("sensitivity requires the production evaluation seeds 7321–7323")
    candidates = []
    for root in (experiment, production):
        base = yaml.safe_load((root / "study" / f"{METHOD}.yaml").read_text())
        grid = yaml.safe_load((root / "study/grid.yaml").read_text())
        candidates.append(
            next(
                c
                for c in c3_search_candidates(base, grid["methods"][METHOD])
                if c["id"] == CANDIDATE
            )
        )
    restored = deepcopy(candidates[0]["config"])
    if (
        restored["patches"]["inner_mask_fraction"] != 0.1
        or candidates[1]["config"]["patches"]["inner_mask_fraction"] != 0.25
    ):
        raise ValueError("sensitivity requires the mask 0.1 versus 0.25 contrast")
    restored["patches"]["inner_mask_fraction"] = 0.25
    if method_config_hash(restored) != method_config_hash(candidates[1]["config"]):
        raise ValueError("sensitivity changes more than the inner mask fraction")
    selection = contracts[0]["fixed_cell_selection"]
    verify_hashes(repo, {selection["path"]: selection["sha256"]})
    historical = json.loads((repo / selection["path"]).read_text())
    if historical["winner"]["id"] != CANDIDATE:
        raise ValueError("the historical selection does not name the registered fixed cell")
    return candidates[0], contracts[0]["trace_split"]["evaluation_seeds"], selection


def run_c3_fixed_cell_sensitivity(repo, experiment, production):
    """Run three fixed seeds after production's barrier, without a new S1 search."""
    candidate, seeds, provenance = validate_fixed_cell_sensitivity(repo, experiment, production)
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    with (workers / "fixed_cell.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = workers / f"{METHOD}.json"
        state = {
            "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
            "status": "running",
            "rule": "pre_registered_external_winner_no_selection",
            "fixed_configuration": CANDIDATE,
            "production_experiment": str(production),
            "production_seal_sha256": sha256_file(production / "experiment.json"),
            "historical_selection": provenance,
            "completed_evaluation": 0,
        }
        write_worker_state(state_path, state)
        try:
            # This separate sensitivity has an externally fixed cell. Grant test
            # access only after verifying the real production barrier above;
            # do not fabricate 25 S1 outcomes or a sensitivity selection file.
            write_worker_state(
                workers / "s1_barrier.json",
                {
                    "passed": True,
                    "methods": [METHOD],
                    "rule": state["rule"],
                    "production_experiment": str(production),
                    "production_barrier_sha256": sha256_file(
                        production / "workers/s1_barrier.json"
                    ),
                    "historical_selection": provenance,
                },
            )
            state.update(phase="evaluation")
            write_worker_state(state_path, state)
            evaluations = []
            for seed in seeds:
                state.update(active_run=f"seed_{seed}")
                write_worker_state(state_path, state)
                evaluations.append(
                    execute_c3_region_run(repo, experiment, candidate, seed, "evaluation")
                )
                state["completed_evaluation"] = sum(
                    row["status"] == "completed" for row in evaluations
                )
                write_worker_state(state_path, state)
            summarize_c3_region_seeds(evaluations, expected_seeds=seeds).to_csv(
                workers / f"{METHOD}_summary.csv", index=False
            )
            state.update(status="completed", phase="completed", active_run=None)
            write_worker_state(state_path, state)
        except BaseException as error:
            state.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
            write_worker_state(state_path, state)
            raise
    return evaluations
