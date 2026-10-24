"""Read-only orchestration of the registered study_067 analysis."""

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from seis_interp.data.study067_analysis_inputs import (
    analysis_hashes,
    partition,
    read_json,
    read_yaml,
    require_output_directory,
    saved_traces,
    verify_run_rows,
)
from seis_interp.evaluation.forge_fair_grid_statistics import cluster_bootstrap_weights
from seis_interp.evaluation.study067_report import analyze_dataset, evaluation_runs, selected_runs
from seis_interp.evaluation.study067_statistics import adjust_families, best_claims, contrast_table


def write_manifest(repo, evaluation, experiments, output, details):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tags = subprocess.check_output(
        ["git", "tag", "--points-at", "HEAD"], cwd=repo, text=True
    ).splitlines()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True
    ).splitlines()
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_tags": tags,
        "worktree_changes": dirty,
        "sha256": analysis_hashes(repo, evaluation, experiments),
        "experiments": [
            str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p) for p in experiments
        ],
        "analysis_configuration": read_yaml(evaluation),
        **details,
    }
    (output / "manifest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (output / "evaluation.yaml").write_bytes(evaluation.read_bytes())
    return record


def evaluate_study067(repo, evaluation, c3, forge, output):
    config = read_yaml(evaluation)
    output = require_output_directory(repo, output)
    write_manifest(
        repo,
        evaluation,
        [c3, forge],
        output,
        {
            "status": "started",
            "bootstrap_p_value": "(1 + count(abs(delta_star - delta) >= abs(delta))) / (B + 1)",
            "weights": (
                "one multinomial matrix per dataset source universe, shared across seeds, "
                "methods, contrasts, metrics and validation-free sensitivity"
            ),
        },
    )
    primary, sensitivity = [], []
    for dataset, experiment in (("c3", c3), ("forge", forge)):
        first, second = analyze_dataset(repo, experiment, dataset, config, output / dataset)
        primary.append(first)
        sensitivity.append(second)
    table = adjust_families(pd.concat(primary, ignore_index=True), config["families"])
    table.to_csv(output / "contrasts.csv", index=False)
    # Sensitivity is descriptive: no new multiplicity family is invented.
    subset = pd.concat(sensitivity, ignore_index=True)
    subset.to_csv(output / "contrasts_validation_free.csv", index=False)
    best_claims(table).to_csv(output / "best_claims.csv", index=False)
    best_claims(subset).to_csv(output / "best_claims_validation_free.csv", index=False)
    for name in (
        "selection",
        "per_seed_metrics",
        "seed_summary",
        "ema_shadow",
        "cost",
        "search_scores",
        "expected_best_of_k",
        "winners_curse",
    ):
        pd.concat(
            [pd.read_csv(output / dataset / f"{name}.csv") for dataset in ("c3", "forge")],
            ignore_index=True,
        ).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_type": "fresh neural EMA shadow",
                "missing_array": "shadow predictions/weights are not saved",
                "available": "validation/test scalar metrics in ema_shadow_metrics",
                "minimal_fix": (
                    "a future run could save shadow predictions and cell_ids; "
                    "no production code changed"
                ),
            },
            {
                "run_type": "imported S1 without shadow metrics",
                "missing_array": "original run did not save EMA shadow predictions or metrics",
                "available": "primary saved prediction and validation metrics",
                "minimal_fix": (
                    "report missing; recovering historical shadow predictions requires "
                    "retained weights or a future rerun"
                ),
            },
            {
                "run_type": "classical",
                "missing_array": "EMA is not applicable",
                "available": "primary saved prediction",
                "minimal_fix": "none",
            },
        ]
    ).to_csv(output / "array_availability.csv", index=False)
    record = read_json(output / "manifest.json")
    record["status"] = "complete"
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return output


def reproduce_c3(repo, evaluation, experiment, reference, output):
    """Acceptance: same per-seed pairing/bootstrap code, without full report costs."""
    config = read_yaml(evaluation)
    contract = read_yaml(experiment / "study/config.yaml")
    seeds = contract["trace_split"]["evaluation_seeds"]
    if seeds != [7311, 7312, 7313]:
        raise ValueError("acceptance reproduction requires final seeds 7311–7313")
    output = require_output_directory(repo, output)
    tests, selections, partitions = {}, [], []
    for seed in [contract["trace_split"]["search_seed"], *seeds]:
        current, original = partition(experiment, seed), partition(reference, seed)
        if not current.equals(original):
            raise ValueError("study_063 and study_064 partitions differ")
        partitions.append({"seed": seed, "study_063_equals_study_064": True})
    for method in ("regsi_real", "ccnet5d_grid"):
        _, winner = selected_runs(experiment, method, 25)
        selections.append(
            {
                "method": method,
                **{k: winner[k] for k in ("id", "axis_a", "axis_b", "validation_score")},
            }
        )
        for directory, record in evaluation_runs(experiment, method, seeds, winner["id"]):
            table = saved_traces(directory, "test")
            verify_run_rows(table, partition(experiment, record["trace_seed"]), "test")
            tests[method, record["trace_seed"]] = table
    clusters = np.sort(partition(experiment, seeds[0])[config["cluster_column"]].unique())
    weights = cluster_bootstrap_weights(
        len(clusters), config["bootstrap"]["resamples"], config["bootstrap"]["seed"]
    )
    table = contrast_table(
        tests, ["ccnet5d_grid"], seeds, clusters, weights, cluster_column=config["cluster_column"]
    )
    table.to_csv(output / "contrasts.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "selection.csv", index=False)
    pd.DataFrame(partitions).to_csv(output / "partition_equality.csv", index=False)
    np.savez_compressed(output / "bootstrap_weights.npz", clusters=clusters, weights=weights)
    write_manifest(
        repo,
        evaluation,
        [experiment, reference],
        output,
        {
            "status": "complete",
            "scope": (
                "study_063/064 C3 acceptance; no Holm family because only one of "
                "the eight contrasts is evaluated"
            ),
        },
    )
    return table
