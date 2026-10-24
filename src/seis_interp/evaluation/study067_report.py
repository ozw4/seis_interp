"""Assemble the common-grid report from completed, immutable saved runs."""

from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.study067_analysis_inputs import (
    forge_idw_scales,
    manifest_records,
    partition,
    read_json,
    read_yaml,
    saved_traces,
    scoring_rows,
    target_source,
    verify_run_rows,
)
from seis_interp.evaluation.forge_fair_grid_statistics import (
    cluster_bootstrap_weights,
    seed_mean_sd,
)
from seis_interp.evaluation.study067_search_diagnostics import (
    ranking_summary,
    search_curve,
    winners_curse,
)
from seis_interp.evaluation.study067_statistics import contrast_table, select_candidate
from seis_interp.evaluation.study067_trace_metrics import derived_traces, population_summaries
from seis_interp.visualization.study067_figures import (
    grid_edge,
    search_figures,
    validation_heatmaps,
)


def selected_runs(experiment, method, expected_count):
    records = manifest_records(experiment, method, "S1")
    if len(records) != expected_count:
        raise ValueError(f"{method}: expected {expected_count} S1 candidates, found {len(records)}")
    winner = select_candidate([record for _, record in records])
    if winner["status"] != "completed":
        raise ValueError(f"{method}: all S1 candidates diverged; no finite selected evaluation")
    return records, winner


def evaluation_runs(experiment, method, seeds, selected_id):
    records = manifest_records(experiment, method, "evaluation")
    if sorted(r["trace_seed"] for _, r in records) != sorted(seeds):
        raise ValueError(f"{method}: need exactly the three registered evaluation seeds")
    for _, record in records:
        if record["status"] not in ("completed", "diverged") or record["id"] != selected_id:
            raise ValueError(f"{method}: evaluation must be finished and use the S1 winner")
    return records


def shadow_rows(directory, record):
    config = read_yaml(directory / "config.resolved.yaml")
    decays = config.get("training", {}).get("ema_eval_decays", [])
    domains = ("validation",) if record["stage"] == "S1" else ("validation", "test")
    rows = []
    for domain in domains:
        available = record.get("ema_shadow_metrics", {}).get(domain, {})
        for decay in sorted(set(map(str, decays)) | set(available)) or [None]:
            metrics = available.get(decay)
            if metrics is None:
                rows.append(
                    {
                        "domain": domain,
                        "decay": decay,
                        "status": "not_configured" if decay is None else "missing_saved_shadow",
                        "population": None,
                    }
                )
                continue
            populations = metrics if "clean_target" in metrics else {"all_eligible": metrics}
            for population, values in populations.items():
                flat = {key: value for key, value in values.items() if not isinstance(value, dict)}
                flat["median_correlation"] = values.get("correlation", {}).get("median")
                rows.append(
                    {
                        "domain": domain,
                        "decay": decay,
                        "status": "available",
                        "population": population,
                        **flat,
                    }
                )
    return rows


def cost_rows(directory, record):
    rows = []
    paths = [(directory / "manifest.json", record, "current")]
    paths += [
        (p, read_json(p), "prior_attempt")
        for p in sorted((directory / "attempts").glob("*/manifest.json"))
    ]
    for path, current, kind in paths:
        imported = current.get("imported_from", {})
        rows.append(
            {
                "manifest": str(path),
                "kind": kind,
                "status": current["status"],
                "runtime_seconds": current.get("runtime_seconds"),
                "prediction_runtime_seconds": current.get("prediction_execution", {}).get(
                    "runtime_seconds"
                ),
                "imported": bool(imported),
                "import_source": imported.get("source_path"),
                "peak_cpu_rss_bytes": current.get("peak_cpu_rss_bytes"),
                "peak_cuda_reserved_bytes": current.get("peak_cuda_reserved_bytes"),
            }
        )
        if imported:
            source_path = Path(imported["source_path"]) / "manifest.json"
            source = read_json(source_path)
            rows.append(
                {
                    "manifest": str(source_path),
                    "kind": "import_source_do_not_sum_duplicates",
                    "status": source["status"],
                    "runtime_seconds": source.get("runtime_seconds"),
                    "prediction_runtime_seconds": source.get("prediction_execution", {}).get(
                        "runtime_seconds"
                    ),
                    "imported": True,
                    "import_source": str(source_path.parent),
                }
            )
    return rows


def seed_summary(metrics, seeds):
    selected = metrics.loc[metrics.stage.eq("evaluation")]
    rows = []
    columns = [
        "relative_mse",
        "global_nmse",
        "global_snr_db",
        "median_correlation",
        "mean_trace_snr_db",
        "trace_snr_p10",
        "trace_snr_p50",
        "trace_snr_p90",
        "fraction_below_0_db",
    ]
    for keys, group in selected.groupby(
        ["dataset", "method", "domain", "population", "restoration"], sort=False
    ):
        group = group.set_index("seed").loc[seeds]
        for metric in columns:
            values = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(values).all()
            mean, sd = seed_mean_sd(values) if valid else (None, None)
            rows.append(
                {
                    **dict(
                        zip(
                            ("dataset", "method", "domain", "population", "restoration"),
                            keys,
                            strict=True,
                        )
                    ),
                    "metric": metric,
                    "mean": mean,
                    "sample_sd": sd,
                    "seed_count": 3,
                    "status": "available"
                    if valid
                    else "unavailable: a selected evaluation seed diverged"
                    if "status" in group and group.status.eq("diverged").any()
                    else "undefined_or_not_applicable",
                    **{f"seed_{i + 1}": seed for i, seed in enumerate(seeds)},
                    **{
                        f"value_seed_{i + 1}": float(v) if np.isfinite(v) else None
                        for i, v in enumerate(values)
                    },
                }
            )
    return pd.DataFrame(rows)


def analyze_method_runs(dataset, method, s1, evaluations, destination, assignments, source, scales):
    metrics, costs, shadows, records, validation, tests = [], [], [], [], {}, {}
    for directory, record in [*s1, *evaluations]:
        identity = {
            "dataset": dataset,
            "method": method,
            "stage": record["stage"],
            "candidate_id": record["id"],
            "seed": record["trace_seed"],
            "imported": bool(record.get("imported_from")),
        }
        costs.extend({**identity, **row} for row in cost_rows(directory, record))
        shadows.extend({**identity, **row} for row in shadow_rows(directory, record))
        if record["stage"] == "S1":
            search_record = {
                "id": record["id"],
                "status": record["status"],
                "validation_score": record.get("validation_score"),
                "validation_snr_db": np.nan,
                "source": str(directory),
                "imported": identity["imported"],
            }
            records.append(search_record)
        if record["status"] == "diverged":
            if record["stage"] == "evaluation":
                tests[method, record["trace_seed"]] = None
                metrics.extend(
                    {**identity, **row} for row in unavailable_seed_metrics(dataset, "diverged")
                )
            continue
        run_output = destination / record["stage"] / directory.name
        run_output.mkdir(parents=True)
        for domain in ("validation",) if record["stage"] == "S1" else ("validation", "test"):
            saved = saved_traces(directory, domain)
            verify_run_rows(saved, assignments[record["trace_seed"]], domain)
            table = derived_traces(directory, saved, source, scales.get(record["trace_seed"]))
            table.to_parquet(run_output / f"{domain}_per_trace.parquet", index=False)
            metrics.extend(
                {**identity, "domain": domain, **summary}
                for summary in population_summaries(table, dataset)
            )
            primary = table if dataset == "c3" else table.loc[table.clean_target]
            if domain == "validation" and not np.isclose(
                primary.relative_mse.mean(), record["validation_score"], rtol=1e-9, atol=1e-12
            ):
                raise ValueError("saved S1/seed validation score mismatch")
            if record["stage"] == "S1":
                search_record["validation_snr_db"] = float(primary.snr_db.mean())
                validation[record["id"]] = primary
            elif domain == "test":
                tests[method, record["trace_seed"]] = primary
    return metrics, costs, shadows, records, validation, tests


def unavailable_seed_metrics(dataset, status):
    """Keep failed seeds visible; never average just the remaining seeds."""
    populations = ("all_eligible",) if dataset == "c3" else ("clean_target", "all_eligible")
    restorations = (
        ("physical",)
        if dataset == "c3"
        else ("trace_oracle", "non_oracle_log_rms_idw", "scale_invariant")
    )
    return [
        {"domain": domain, "population": population, "restoration": restoration, "status": status}
        for domain in ("validation", "test")
        for population in populations
        for restoration in restorations
    ]


def add_pilot_scores(repo, dataset, method, pilots, records, validation, validation_sets):
    for pilot in pilots:
        if pilot["dataset"] != dataset or pilot["method"] != method:
            continue
        directory = repo / pilot["directory"]
        record = read_json(directory / "manifest.json")
        if record["stage"] != "S1" or record["status"] not in ("completed", "diverged"):
            raise ValueError("pilot score source must be a completed/diverged S1 run")
        pilot_id = f"pilot:{pilot['directory']}"
        pilot_assignment = partition(directory.parents[3], record["trace_seed"])
        validation_sets.append(scoring_rows(pilot_assignment, "validation").trace_id.to_numpy())
        table = saved_traces(directory, "validation") if record["status"] == "completed" else None
        if table is not None:
            verify_run_rows(table, pilot_assignment, "validation")
            table = table if dataset == "c3" else table.loc[table.clean_target]
            validation[pilot_id] = table
        records.append(
            {
                "id": pilot_id,
                "status": record["status"],
                "validation_score": record.get("validation_score"),
                "validation_snr_db": float(table.snr_db.mean()) if table is not None else np.nan,
                "source": str(directory),
                "imported": False,
            }
        )


def analyze_dataset(repo, experiment, dataset, config, output):
    """Produce per-dataset tables/figures and return unadjusted paired contrasts."""
    contract = read_yaml(experiment / "study/config.yaml")
    grid = read_yaml(experiment / "study/grid.yaml")["methods"]
    seal = read_json(experiment / "experiment.json")
    seeds = contract["trace_split"]["evaluation_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("evaluation requires exactly three distinct seeds")
    search_seed = contract["trace_split"]["search_seed"]
    methods = ["regsi_real", *config["contrasts"]]
    plans = {}
    for method in methods:
        s1, winner = selected_runs(experiment, method, seal["s1_runs_per_method"])
        plans[method] = (s1, winner, evaluation_runs(experiment, method, seeds, winner["id"]))
    # The source-cell universe is fixed by the physical dataset, not a test intersection.
    assignment = partition(experiment, search_seed)
    clusters = np.sort(assignment[config["cluster_column"]].unique())
    expected = config["datasets"][dataset]["source_cell_count"]
    if len(clusters) != expected:
        raise ValueError(f"{dataset}: expected {expected} source cells, found {len(clusters)}")
    weights = cluster_bootstrap_weights(
        len(clusters), config["bootstrap"]["resamples"], config["bootstrap"]["seed"]
    )
    output.mkdir()
    np.savez_compressed(output / "bootstrap_weights.npz", clusters=clusters, weights=weights)
    source = target_source(repo, experiment, dataset, contract)
    scales = {}
    if dataset == "forge":
        for seed in [search_seed, *seeds]:
            table, _ = forge_idw_scales(experiment, seed, contract)
            scales[seed] = table
            table.to_parquet(output / f"idw_scales_seed_{seed}.parquet", index=False)
    assignments = {seed: partition(experiment, seed) for seed in [search_seed, *seeds]}
    validation_sets = [scoring_rows(assignment, "validation").trace_id.to_numpy()]
    metrics, shadows, costs, selections, search_rows, curves, curses = [], [], [], [], [], [], []
    tests = {}
    for method, (s1, winner, evaluations) in plans.items():
        print(f"{dataset}/{method}: saved-array analysis", flush=True)
        destination = output / method
        destination.mkdir()
        method_metrics, method_costs, method_shadows, records, validation, method_tests = (
            analyze_method_runs(
                dataset, method, s1, evaluations, destination, assignments, source, scales
            )
        )
        metrics.extend(method_metrics)
        costs.extend(method_costs)
        shadows.extend(method_shadows)
        tests.update(method_tests)
        ranking = ranking_summary(records)
        selections.append(
            {
                "dataset": dataset,
                "method": method,
                "candidate_id": winner["id"],
                "axis_a": winner["axis_a"],
                "axis_b": winner["axis_b"],
                "validation_relmse": winner["validation_score"],
                "grid_edge": grid_edge(winner["id"], grid[method]),
                "s1_candidates": len(s1),
                "s1_diverged": sum(r["status"] == "diverged" for _, r in s1),
                "s1_imported": sum(bool(r.get("imported_from")) for _, r in s1),
                **ranking,
            }
        )
        validation_heatmaps(
            records, grid[method], winner["id"], ranking, destination, f"{dataset}/{method}"
        )
        add_pilot_scores(
            repo,
            dataset,
            method,
            config["extra_pilot_score_sources"],
            records,
            validation,
            validation_sets,
        )
        method_curves, method_curses = [], []
        for metric in ("relative_mse", "snr_db"):
            method_curves.append(
                search_curve(records, metric).assign(dataset=dataset, method=method)
            )
            method_curses.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **winners_curse(
                        validation, records, metric, clusters, weights, config["cluster_column"]
                    ),
                }
            )
        method_curves, method_curses = (
            pd.concat(method_curves, ignore_index=True),
            pd.DataFrame(method_curses),
        )
        search_figures(method_curves, method_curses, destination, f"{dataset}/{method}")
        curves.append(method_curves)
        curses.extend(method_curses.to_dict("records"))
        search_rows.extend({"dataset": dataset, "method": method, **r} for r in records)
    seed_metrics = pd.DataFrame(metrics)
    tables = {
        "selection": pd.DataFrame(selections),
        "per_seed_metrics": seed_metrics,
        "seed_summary": seed_summary(seed_metrics, seeds),
        "ema_shadow": pd.DataFrame(shadows),
        "cost": pd.DataFrame(costs),
        "search_scores": pd.DataFrame(search_rows),
        "expected_best_of_k": pd.concat(curves, ignore_index=True),
        "winners_curse": pd.DataFrame(curses),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    primary = contrast_table(
        tests,
        config["contrasts"],
        seeds,
        clusters,
        weights,
        cluster_column=config["cluster_column"],
    ).assign(dataset=dataset)
    sensitivity = contrast_table(
        tests,
        config["contrasts"],
        seeds,
        clusters,
        weights,
        cluster_column=config["cluster_column"],
        validation_sets=validation_sets,
    ).assign(dataset=dataset)
    pd.DataFrame(
        {"trace_id": sorted(set().union(*(set(ids) for ids in validation_sets)))}
    ).to_parquet(output / "s1_validation_union.parquet", index=False)
    return primary, sensitivity
