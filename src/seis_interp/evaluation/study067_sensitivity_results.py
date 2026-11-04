"""Read-only S-a/S-b results using the frozen study_067 loaders and statistics."""

import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.study067_analysis_inputs import (
    partition,
    read_json,
    read_yaml,
    require_output_directory,
    saved_traces,
    scoring_rows,
    verify_run_rows,
)
from seis_interp.evaluation.forge_fair_grid_statistics import cluster_bootstrap_weights
from seis_interp.evaluation.study067_statistics import (
    adjust_families,
    bootstrap_seed_means,
    contrast_table,
)


def paired_rows(left, right):
    """Align exact scoring identities; never intersect differing populations."""
    left, right = (
        table.set_index("cell_id", verify_integrity=True).sort_index() for table in (left, right)
    )
    if not left.index.equals(right.index):
        raise ValueError("unpaired cell_id")
    for column in ("trace_id", "source_cell", "clean_target", "target_energy"):
        if not left[column].equals(right[column]):
            raise ValueError(f"paired {column} differs")
    return left, right


def reused_weights(path, clusters, *, resamples=10000, seed=670703):
    """Load primary multiplicities and prove exact equality to the registered RNG."""
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["clusters"], clusters):
            raise ValueError("primary bootstrap source-cell universe differs")
        weights = saved["weights"]
    regenerated = cluster_bootstrap_weights(len(clusters), resamples, seed)
    if not np.array_equal(weights, regenerated):
        raise ValueError("primary bootstrap weights differ from registered RNG")
    return weights


def c3_contrast(left, right, seeds, clusters, weights, validation_sets=()):
    """Delegate all seed balancing, inference and identity checks to frozen code."""
    tests = {}
    for seed in seeds:
        paired_rows(left[seed], right[seed])
        tests["regsi_real", seed] = left[seed]
        tests["ccnet5d_grid", seed] = right[seed]
    return contrast_table(
        tests,
        ["ccnet5d_grid"],
        seeds,
        clusters,
        weights,
        cluster_column="source_cell",
        validation_sets=validation_sets,
    ).assign(dataset="c3")


def primary_self_check(computed, registered, families):
    """Compare exact full rows, recalculating primary Holm with other saved rows.

    Only the C3 CCNet contrast is recomputed from traces. The other seven
    primary contrasts supply their registered p values for this self-check.
    No Holm adjustment is applied to any sensitivity result.
    """
    keep = registered.dataset.eq("c3") & registered.baseline.eq("ccnet5d_grid")
    expected = registered.loc[keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        computed.reset_index(drop=True), expected[computed.columns], check_exact=True
    )
    combined = registered.copy()
    combined.loc[keep, computed.columns] = computed.to_numpy()
    adjusted = adjust_families(combined, families).loc[keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(adjusted, expected, check_exact=True)
    return adjusted


def load_run(experiment, method, stage, candidate, seed, paths):
    directory = (
        experiment
        / "method_runs"
        / method
        / stage
        / (candidate if stage == "S1" else f"seed_{seed}")
    )
    domain = "validation" if stage == "S1" else "test"
    paths.update(
        [
            directory / "manifest.json",
            directory / "config.resolved.yaml",
            directory / f"{domain}_per_trace.parquet",
            experiment / "input" / f"seed_{seed}" / "partition.parquet",
        ]
    )
    record = read_json(directory / "manifest.json")
    if (record["id"], record["stage"], record["trace_seed"], record["status"]) != (
        candidate,
        stage,
        seed,
        "completed",
    ):
        raise ValueError(f"unexpected run identity/status: {directory}")
    table = saved_traces(directory, domain)
    assignment = partition(experiment, seed)
    verify_run_rows(table, assignment, domain)
    expected = scoring_rows(assignment, domain)
    for column in ("source_cell", "clean_target"):
        if (
            not table.set_index("cell_id")[column]
            .sort_index()
            .equals(expected.set_index("cell_id")[column].sort_index())
        ):
            raise ValueError(f"partition {column} differs: {directory}")
    return table, record


def single_cell(left, right, left_record, right_record, dataset, name, clusters, weights):
    left, right = paired_rows(left, right)
    if dataset == "forge":
        left, right = left.loc[left.clean_target], right.loc[left.clean_target]
    metrics = []
    for table, record in ((left, left_record), (right, right_record)):
        metric = record["metrics"]["validation"]
        if dataset == "forge":
            metric = metric["clean_target"]
        if len(table) != metric["trace_count"] or not np.allclose(
            [table.snr_db.mean(), table.relative_mse.mean()],
            [metric["mean_trace_snr_db"], metric["masked_trace_relative_mse"]],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("saved production validation metrics mismatch")
        metrics.append(metric)
    result = {
        "sensitivity": name,
        "dataset": dataset,
        "candidate_id": left_record["id"],
        "seed": left_record["trace_seed"],
        "domain": "S1 validation",
        "population": "clean_target" if dataset == "forge" else "all_eligible",
        "trace_count": len(left),
        "left_mean_trace_snr_db": metrics[0]["mean_trace_snr_db"],
        "right_mean_trace_snr_db": metrics[1]["mean_trace_snr_db"],
        "difference_db": metrics[0]["mean_trace_snr_db"] - metrics[1]["mean_trace_snr_db"],
        "left_relative_mse": metrics[0]["masked_trace_relative_mse"],
        "right_relative_mse": metrics[1]["masked_trace_relative_mse"],
    }
    result["registered_ge_0_2_db"] = result["difference_db"] >= 0.2
    if dataset == "forge":
        delta = (left.snr_db - right.snr_db).to_numpy()
        draws = bootstrap_seed_means([delta], [left.source_cell.to_numpy()], clusters, weights)
        low, high = np.percentile(draws[:, 0], [2.5, 97.5])
        result.update(
            descriptive_paired_trace_mean_db=float(delta.mean()),
            descriptive_ci95_low=float(low),
            descriptive_ci95_high=float(high),
            bootstrap_resamples=len(weights),
            cluster_count=len(clusters),
        )
    return result


def summary_text(contrasts, cells, output):
    lines = [
        "# study_067 registered S-a / S-b sensitivity",
        "",
        f"Output: `{output}`. Human-readable numbers are rounded to four decimals; "
        "CSVs retain precision.",
        "",
        "C3 uses each seed's own test set (7321, 7322, 7323), then equal seed weights. "
        "Source-cell percentile bootstrap: RNG 670703, B=10,000; weights are loaded from "
        "the primary analysis and exactly equal regenerated weights. p is null-centred, "
        "two-sided, with the plus-one convention. No Holm correction: these sensitivities "
        "do not form a registered multiplicity family. Validation-free excludes the primary "
        "union of all S1 validation identities, including pilots; sensitivity S1 partitions "
        "are identical to the corresponding production partition.",
        "",
        "| C3 contrast | Population | Metric | Delta | Percentile 95% CI | p | "
        "Delta 7321 / 7322 / 7323 | Paired t 95% CI (df=2) | Trace counts |",
        "|---|---|---|---:|---|---:|---|---|---|",
    ]
    for row in contrasts.to_dict("records"):
        t_high = f"{row['t_ci95_high']:.4f}"
        if float(t_high) == 0:
            t_high = "approximately 0"
        lines.append(
            f"| {row['contrast']} | {row['population']} | {row['metric']} | "
            f"{row['estimate']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['bootstrap_p_value']:.10g} | "
            + " / ".join(f"{row[f'delta_seed_{i}']:.4f}" for i in (1, 2, 3))
            + f" | [{row['t_ci95_low']:.4f}, {t_high}] | "
            + " / ".join(str(row[f"trace_count_seed_{i}"]) for i in (1, 2, 3))
            + " |"
        )
    lines += [
        "",
        "CCNet@0.1 minus CCNet@0.25 test contrasts are descriptive. "
        "Positive SNR and negative relative MSE favour the left method.",
        "",
        "| Registered single-cell pair | Dataset | Left SNR | Right SNR | Left minus right (dB) | "
        ">=0.2 dB | Descriptive paired source-cell 95% CI |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in cells.to_dict("records"):
        interval = (
            f"[{row['descriptive_ci95_low']:.4f}, {row['descriptive_ci95_high']:.4f}]"
            if row["dataset"] == "forge"
            else "not requested"
        )
        lines.append(
            f"| {row['sensitivity']} | {row['dataset']} | {row['left_mean_trace_snr_db']:.4f} | "
            f"{row['right_mean_trace_snr_db']:.4f} | {row['difference_db']:.4f} | "
            f"{row['registered_ge_0_2_db']} | {interval} |"
        )
    sa = cells.loc[cells.sensitivity.eq("S-a mask 0.1 minus 0.25")]
    sb = cells.loc[cells.sensitivity.eq("S-b FP32 minus bf16")].iloc[0]
    lines += [
        "",
        f"S-a: >=0.2 dB on either dataset: {bool(sa.registered_ge_0_2_db.any())}. "
        "The historical C3 comparison uses study_064 a05_b02 (19.073273231598527 dB); "
        "it is the registered historical pair, not an identical modern RNG implementation. "
        "The new C3 seeds change only the mask under the current production implementation.",
        f"S-b: FP32 minus bf16 >=0.2 dB: {bool(sb.registered_ge_0_2_db)}. "
        + (
            "The registered FP32 three-seed remedy is triggered."
            if sb.registered_ge_0_2_db
            else "The registered FP32 three-seed remedy is not triggered."
        ),
        "S-b timing deviation (already recorded in decisions.md): FORGE CCNet seeds had "
        "already been trained and predicted at preparation on 2026-11-01, while test scoring "
        "was waiting for the all-method S1 barrier. The requirement to report before starting "
        "those seeds was not met.",
        "",
        "Self-check: both recomputed primary C3 ReGSI minus CCNet@0.25 rows exactly equal "
        "contrasts.csv, including all raw statistics and the original primary Holm fields "
        "recalculated using the seven other registered contrasts. See primary_self_check.csv. "
        "All original analysis code SHA256 values match the primary manifest.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_sensitivity(repo, experiments, primary, output, provenance_paths=()):
    """Compute only saved-trace statistics; create a fresh directory outside runs/."""
    if output.resolve().is_relative_to(primary.resolve()):
        raise ValueError("sensitivity output must be outside the primary analysis")
    config_path = repo / "studies/study_067_forge_c3_common_grid/evaluation.yaml"
    config = read_yaml(config_path)
    primary_record = read_json(primary / "manifest.json")
    paths = {config_path, primary / "manifest.json", primary / "contrasts.csv", *provenance_paths}
    code_paths = {
        repo / name for name in primary_record["sha256"] if name.startswith(("src/", "scripts/"))
    }
    for path in code_paths | {config_path}:
        if sha256_file(path) != primary_record["sha256"][str(path.relative_to(repo))]:
            raise ValueError(f"frozen analysis SHA256 differs: {path}")
    seeds = [7321, 7322, 7323]
    clusters, weights, search_seeds = {}, {}, {}
    for dataset in ("c3", "forge"):
        experiment = experiments[dataset]
        paths.add(experiment / "study/config.yaml")
        contract = read_yaml(experiment / "study/config.yaml")
        if contract["trace_split"]["evaluation_seeds"] != seeds:
            raise ValueError("unexpected evaluation seeds")
        search_seeds[dataset] = seed = contract["trace_split"]["search_seed"]
        paths.add(experiment / "input" / f"seed_{seed}" / "partition.parquet")
        clusters[dataset] = np.sort(partition(experiment, seed).source_cell.unique())
        if len(clusters[dataset]) != config["datasets"][dataset]["source_cell_count"]:
            raise ValueError("unexpected source-cell count")
        weight_path = primary / dataset / "bootstrap_weights.npz"
        paths.add(weight_path)
        weights[dataset] = reused_weights(
            weight_path,
            clusters[dataset],
            resamples=config["bootstrap"]["resamples"],
            seed=config["bootstrap"]["seed"],
        )
    tests = {}
    for label, method, candidate in (
        ("regsi", "regsi_real", "a02_b05"),
        ("ccnet025", "ccnet5d_grid", "a05_b02"),
        ("ccnet010", "ccnet5d_grid", "a05_b02"),
    ):
        experiment = experiments["sa_c3" if label == "ccnet010" else "c3"]
        tests[label] = {
            seed: load_run(experiment, method, "evaluation", candidate, seed, paths)[0]
            for seed in seeds
        }
    arguments = seeds, clusters["c3"], weights["c3"]
    computed = c3_contrast(tests["regsi"], tests["ccnet025"], *arguments)
    registered = pd.read_csv(primary / "contrasts.csv", float_precision="round_trip")
    self_check = primary_self_check(computed, registered, config["families"])
    union_path = primary / "c3/s1_validation_union.parquet"
    paths.add(union_path)
    union = pd.read_parquet(union_path).trace_id.to_numpy()
    c3_tables = []
    for left, right, name in (
        ("regsi", "ccnet010", "ReGSI minus CCNet@0.1"),
        ("ccnet010", "ccnet025", "CCNet@0.1 minus CCNet@0.25 (descriptive)"),
    ):
        for population, validation in (("all_eligible", ()), ("validation_free", (union,))):
            table = c3_contrast(tests[left], tests[right], *arguments, validation)
            table = table.drop(columns=["direction", "passes_best_condition", "baseline"])
            c3_tables.append(table.assign(contrast=name, population=population))
    cells = []
    for dataset, arm, candidate, name in (
        ("c3", "study064", "a05_b02", "S-a mask 0.1 minus 0.25"),
        ("forge", "sa_forge", "a03_b02", "S-a mask 0.1 minus 0.25"),
        ("forge", "sb_forge", "a03_b02", "S-b FP32 minus bf16"),
    ):
        seed = search_seeds[dataset]
        left, left_record = load_run(experiments[arm], "ccnet5d_grid", "S1", candidate, seed, paths)
        right, right_record = load_run(
            experiments[dataset], "ccnet5d_grid", "S1", candidate, seed, paths
        )
        if not partition(experiments[arm], seed).equals(partition(experiments[dataset], seed)):
            raise ValueError("sensitivity S1 partition differs from production")
        cells.append(
            single_cell(
                left,
                right,
                left_record,
                right_record,
                dataset,
                name,
                clusters[dataset],
                weights[dataset],
            )
        )
    contrasts, cells = pd.concat(c3_tables, ignore_index=True), pd.DataFrame(cells)
    output = require_output_directory(repo, output)
    contrasts.to_csv(output / "c3_contrasts.csv", index=False)
    cells.to_csv(output / "single_cell.csv", index=False)
    self_check.to_csv(output / "primary_self_check.csv", index=False)
    (output / "summary.md").write_text(summary_text(contrasts, cells, output.relative_to(repo)))
    code_paths.update(
        [
            repo / "scripts/evaluate_study067_sensitivity.py",
            repo / "src/seis_interp/evaluation/study067_sensitivity_results.py",
            repo / "tests/unit/test_study067_sensitivity_results.py",
        ]
    )
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "worktree_changes": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True
        ).splitlines(),
        "experiments": {key: str(path.relative_to(repo)) for key, path in experiments.items()},
        "input_sha256": {str(p.relative_to(repo)): sha256_file(p) for p in sorted(paths)},
        "code_sha256": {str(p.relative_to(repo)): sha256_file(p) for p in sorted(code_paths)},
        "output_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir())},
        "bootstrap": config["bootstrap"],
        "weights_exactly_equal_regenerated": {"c3": True, "forge": True},
        "primary_self_check_exact": True,
        "pairing_checks": "cell_id, trace_id, source_cell, clean_target, target_energy; all passed",
        "c3_paired_test_counts": {str(seed): len(tests["ccnet010"][seed]) for seed in seeds},
        "multiplicity": "no Holm correction for sensitivities",
        "forge_bootstrap": (
            "descriptive paired source-cell percentile CI; production universe and weights"
        ),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return output
