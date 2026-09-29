"""Score every fair-grid final prediction on the mask-B test set in one pass.

The final runs were trained and predicted with ``evaluate: false``: nothing has
read their targets yet. Before any target is opened this pipeline binds every
listed prediction to the pre-registration: ``studies/<study>/grid.yaml`` fixes
the bootstrap, and each method's selection record
``runs/<study>/fair_grid/<grid method>/final/stage.json`` (same grid.yaml sha256
and final preparation) lists the final runs. Every such run must be listed once,
be an unevaluated ``purpose: final_evaluation`` tuning run of the recorded yaml,
and predict the selected EMA as its primary checkpoint. It then reads the test
targets once and writes per-seed metrics, seed means and sds, paired contrasts
with source-cell cluster bootstrap CIs and Holm adjustment, a non-oracle
(observed log-RMS IDW) restoration, a scale-invariant metric and the
expected-best curves of the recorded S1-E validation search (diverged runs at
the grid's reported score).

Evaluation config (paths relative to the repository)::

    preparation: runs/<study>/<final preparation run>
    bootstrap: {resamples: 10000, seed: <int>}   # must equal grid.yaml evaluation.bootstrap
    final_runs:
      <method>:          # all six methods
        - {seed: <int or null>, tuning_run: <dir>, candidate: <name>}

A prediction is ``<tuning_run>/<candidate>/prediction.npy``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import read_selected_traces, verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.evaluation.forge_fair_grid_statistics import (
    FAIR_GRID_CONTRASTS,
    cluster_bootstrap_weights,
    error_summary,
    expected_best_curve,
    paired_contrasts,
    scale_invariant_relative_mse,
    seed_average_errors,
    seed_mean_sd,
)
from seis_interp.evaluation.forge_mvp_metrics import (
    evaluate_traces,
    validate_predictions,
    verify_metrics_from_waveforms,
    verify_saved_metrics,
)
from seis_interp.forge_mvp_config import NEURAL_METHODS
from seis_interp.processing.fair_grid_rules import decay_key, is_divergence
from seis_interp.processing.forge_log_rms_idw import LOG_RMS_IDW_SETTINGS, log_rms_idw_scales
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces

MASK_PATH = "masks/m1_random80_mvp_v1.parquet"
RESTORATIONS = ("trace_oracle", "non_oracle_log_rms_idw", "scale_invariant")
# (family, restoration, domain); Holm adjustment runs within each family.
CONTRAST_FAMILIES = (
    ("primary", "trace_oracle", "clean_target"),
    ("secondary_all_eligible", "trace_oracle", "all_eligible"),
    ("secondary_non_oracle", "non_oracle_log_rms_idw", "clean_target"),
    ("secondary_scale_invariant", "scale_invariant", "clean_target"),
)
FINAL_RUN_KEYS = {"seed", "tuning_run", "candidate"}
SEARCH_STAGES = ("S1", "S2", "S3", "E")


def evaluate_forge_fair_grid(repo: Path, evaluation_config: Path) -> Path:
    config = yaml.safe_load(evaluation_config.read_text())
    _validate_evaluation_config(config)
    preparation = repo / config["preparation"]
    seal, contract = _final_preparation(preparation)
    preparation_hash = sha256_file(preparation / "preparation_manifest.json")
    grid_path = repo / "studies" / contract["study_id"] / "grid.yaml"
    grid, grid_hash = yaml.safe_load(grid_path.read_text()), sha256_file(grid_path)
    if grid["study_id"] != contract["study_id"]:
        raise ValueError("grid.yaml belongs to another study")
    if config["bootstrap"] != grid["evaluation"]["bootstrap"]:
        raise ValueError("bootstrap differs from the pre-registered grid.yaml")
    final_preparation = {"path": config["preparation"], "manifest_sha256": preparation_hash}
    selections = {
        method: _selection(repo, grid, grid_hash, method, final_preparation) for method in METHODS
    }
    mask = pd.read_parquet(preparation / MASK_PATH)
    test = mask.split.eq("test").to_numpy()
    index = mask.loc[test].reset_index(drop=True)
    # Verify every prediction and search record before any target is opened.
    runs = [
        _final_run(repo, method, entry, preparation_hash, index, mask, contract, selections[method])
        for method in METHODS
        for entry in config["final_runs"][method]
    ]
    for method in METHODS:
        planned = sorted(run["name"] for run in _planned_runs(selections[method], method))
        if sorted(run["run"] for run in runs if run["method_id"] == method) != planned:
            raise ValueError(f"{method}: final runs differ from the recorded selection")
    search = _search_scores(repo, grid, selections, seal["mask_hash"])
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo
        / "runs"
        / contract["study_id"]
        / f"{stamp}_{identity['git_commit'][:12]}_final_evaluation"
    )
    output.mkdir(parents=True, exist_ok=False)
    print(output, flush=True)
    (output / "evaluation.yaml").write_bytes(evaluation_config.read_bytes())
    record = dict(
        **identity,
        status="running",
        start_time=utc_now(),
        preparation=config["preparation"],
        preparation_hash=preparation_hash,
        grid_sha256=grid_hash,
        selection_records={
            method: sha256_file(_selection_path(repo, grid, method)) for method in METHODS
        },
        mask_hash=seal["mask_hash"],
        selection_domain="final_test",
        independent_test_estimate=True,
        bootstrap=config["bootstrap"],
        cluster="source_cell",
        contrasts=[list(contrast) for contrast in FAIR_GRID_CONTRASTS],
        non_oracle_restoration={**LOG_RMS_IDW_SETTINGS, "interpolated_quantity": "log_rms"},
        final_runs=[{k: v for k, v in run.items() if k != "prediction"} for run in runs],
        test_waveform_reads=0,
    )
    write_json(output / "manifest.json", record)
    try:
        target = _read_test_targets(repo, preparation, contract, mask)
        record.update(test_waveform_reads=1, test_waveforms_read=len(target))
        scales, spacing = _restoration_scales(preparation, contract, mask, index)
        record["non_oracle_restoration"]["axis_scales_m"] = list(spacing)
        aggregate = output / "aggregate"
        aggregate.mkdir()
        scales.to_parquet(aggregate / "non_oracle_trace_scales.parquet", index=False)
        errors, seed_rows = {}, []
        for run in runs:
            run_errors, rows = _score_run(run, target, index, scales, output)
            errors.setdefault(run["method_id"], []).append(run_errors)
            seed_rows.extend(rows)
        seed_metrics = pd.DataFrame(seed_rows)
        seed_metrics.to_csv(aggregate / "seed_metrics.csv", index=False)
        _method_summary(seed_metrics).to_csv(aggregate / "method_summary.csv", index=False)
        averaged = _seed_averaged_errors(errors, index)
        averaged.to_parquet(aggregate / "seed_averaged_errors.parquet", index=False)
        _contrast_table(averaged, config["bootstrap"]).to_csv(
            aggregate / "contrasts.csv", index=False
        )
        if search is not None:
            search.to_csv(aggregate / "search_scores.csv", index=False)
            curves = [
                expected_best_curve(group.primary_relative_mse).assign(method_id=method)
                for method, group in search.groupby("method_id", sort=False)
            ]
            pd.concat(curves, ignore_index=True).to_csv(
                aggregate / "expected_best_curves.csv", index=False
            )
        record["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failure_reason=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        record["end_time"] = utc_now()
        record["artifacts"] = {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "manifest.json"
        }
        write_json(output / "manifest.json", record)
    return output


def _validate_evaluation_config(config):
    if set(config) - {"preparation", "bootstrap", "final_runs"}:
        raise ValueError("unsupported final-evaluation keys")
    bootstrap = config["bootstrap"]
    if set(bootstrap) != {"resamples", "seed"} or any(
        isinstance(v, bool) or not isinstance(v, int) for v in bootstrap.values()
    ):
        raise ValueError("bootstrap needs integer resamples and seed")
    if set(config["final_runs"]) != set(METHODS):
        raise ValueError("final_runs must list every method")
    for method, entries in config["final_runs"].items():
        if not entries:
            raise ValueError(f"{method}: no final runs")
        for entry in entries:
            if not {"seed", "tuning_run", "candidate"} <= set(entry) <= FINAL_RUN_KEYS:
                raise ValueError(f"{method}: final runs need seed, tuning_run and candidate")
        seeds = [entry["seed"] for entry in entries]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{method}: duplicate final seeds")


def _grid_method(grid, method):
    keys = [
        key
        for key, spec in grid["methods"].items()
        if method in (spec["method_id"], spec.get("pair_method_id"))
    ]
    if len(keys) != 1:
        raise ValueError(f"{method}: not a method of the pre-registered grid")
    return keys[0]


def _selection_path(repo, grid, method):
    return (
        repo / "runs" / grid["study_id"] / "fair_grid" / _grid_method(grid, method) / "final"
    ) / "stage.json"


def _selection(repo, grid, grid_hash, method, final_preparation):
    """The method's selection record, checked against the grid and the final preparation."""
    path = _selection_path(repo, grid, method)
    if not path.is_file():
        raise ValueError(f"{method}: no selection record {path.relative_to(repo)}")
    document = json.loads(path.read_text())
    if (
        document["grid_sha256"] != grid_hash
        or document["preparations"]["final"] != final_preparation
    ):
        raise ValueError(f"{method}: selection record is from another grid or preparation")
    return document


def _planned_runs(selection, method):
    return [run for run in selection["runs"] if run["method_id"] == method]


def _final_preparation(preparation):
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    if seal["status"] != "prepared":
        raise ValueError("preparation is not complete")
    if seal.get("selection_domain", "final_test") != "final_test":
        raise ValueError("final evaluation requires the final (mask-B test) preparation")
    if seal.get("amplitude_scaling") != "trace_oracle":
        raise ValueError("final evaluation requires the trace-oracle preparation")
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    return seal, contract


def _final_run(repo, method, entry, preparation_hash, index, mask, contract, selection):
    tuning = repo / entry["tuning_run"]
    manifest = json.loads((tuning / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["method_id"] != method:
        raise ValueError(f"{method}: tuning run is not a completed run of this method")
    if manifest["preparation_hash"] != preparation_hash:
        raise ValueError(f"{method}: tuning run used another preparation")
    if (
        manifest.get("purpose") != "final_evaluation"
        or manifest.get("evaluate") is not False
        or any("metrics" in Path(name).name for name in manifest["artifacts"])
    ):
        raise ValueError(f"{method}: final runs must be unevaluated final_evaluation runs")
    candidate = entry["candidate"]
    relative = f"{candidate}/prediction.npy"
    bound = (
        relative,
        "search.yaml",
        "prediction_index.parquet",
        f"{candidate}/resolved_config.yaml",
    )
    if any(name not in manifest["artifacts"] for name in bound):
        raise ValueError(f"{method}: tuning manifest does not bind {relative}")
    verify_hashes(tuning, {name: manifest["artifacts"][name] for name in bound})
    planned = [
        run
        for run in _planned_runs(selection, method)
        if run["sha256"] == manifest["artifacts"]["search.yaml"]
    ]
    if len(planned) != 1 or planned[0]["name"] != candidate:
        raise ValueError(f"{method}: tuning run is not a final run of the recorded selection")
    resolved = yaml.safe_load((tuning / candidate / "resolved_config.yaml").read_text())
    if resolved["method_id"] != method or resolved.get("model_seed") != entry["seed"]:
        raise ValueError(f"{method}: resolved config method or seed mismatch")
    selected = selection["decision"].get("ema_decay")
    primary = resolved.get("training", {}).get("ema_decay")
    if (selected, primary) != (None, None) and (primary is None or decay_key(primary) != selected):
        raise ValueError(f"{method}: final run does not predict the selected EMA {selected}")
    stored = pd.read_parquet(tuning / "prediction_index.parquet")
    if not stored.drop(columns="prediction_row").equals(index) or not np.array_equal(
        stored.prediction_row, np.arange(len(index))
    ):
        raise ValueError(f"{method}: prediction rows do not follow the test mask order")
    prediction = np.load(tuning / relative, mmap_mode="r")
    order = validate_predictions(
        prediction, index.cell_id.to_numpy(), mask, contract["time_sample_count"]
    )
    if not np.array_equal(order, np.arange(len(index))):
        raise ValueError(f"{method}: prediction rows do not follow the test mask order")
    return {
        "method_id": method,
        "seed": entry["seed"],
        "ema_decay": selected,
        "run": planned[0]["name"],
        "tuning_run": entry["tuning_run"],
        "candidate": candidate,
        "prediction_hash": manifest["artifacts"][relative],
        "prediction": tuning / relative,
    }


def _search_scores(repo, grid, selections, final_mask_hash):
    """V scores of every S1-E configuration in the selection records' rule inputs.

    Each input is bound by its tuning-manifest sha256; a diverged run enters at
    the grid's reported score for every EMA shadow.
    """
    diverged = grid["selection"]["diverged_runs"]
    keys = [decay_key(decay) for decay in grid["ema_eval_decays"]]
    tables = []
    for method, selection in selections.items():
        for entry in selection["rule_inputs"]:
            if entry["method_id"] != method or entry["stage"] not in SEARCH_STAGES:
                continue
            tuning = repo / entry["tuning_run"]
            if sha256_file(tuning / "manifest.json") != entry["tuning_manifest_sha256"]:
                raise ValueError(f"{method}: search run manifest differs from its record")
            manifest = json.loads((tuning / "manifest.json").read_text())
            if manifest["method_id"] != method:
                raise ValueError(f"{method}: search run is not a run of this method")
            if manifest.get("selection_domain") != "validation_holdout":
                raise ValueError(f"{method}: search runs must score the validation holdout")
            seal_path = repo / manifest["preparation"] / "preparation_manifest.json"
            seal = json.loads(seal_path.read_text())
            if (
                sha256_file(seal_path) != manifest["preparation_hash"]
                or seal.get("selection_domain") != "validation_holdout"
                or seal.get("parent_mask_hash") != final_mask_hash
            ):
                raise ValueError(f"{method}: search preparation is not the mask-B holdout")
            if entry["status"] == "diverged":
                if manifest["status"] != "failed" or not is_divergence(
                    manifest.get("failure_reason"), diverged["failure_reason_pattern"]
                ):
                    raise ValueError(f"{method}: {entry['run']} is recorded as diverged")
                shadows = keys if method in NEURAL_METHODS else [None]
                rows = pd.DataFrame(
                    {
                        "name": entry["run"],
                        "ema_decay": shadows,
                        "primary_relative_mse": float(diverged["reported_score"]),
                        "diverged": True,
                    }
                )
            else:
                if manifest["status"] != "completed":
                    raise ValueError(f"{method}: search run is not a completed run")
                verify_hashes(
                    tuning, {"search_results.csv": manifest["artifacts"]["search_results.csv"]}
                )
                rows = pd.read_csv(tuning / "search_results.csv", dtype={"ema_decay": str})
                rows = rows.loc[rows.name.eq(entry["run"])].assign(diverged=False)
            tables.append(
                rows.assign(method_id=method, stage=entry["stage"], tuning_run=entry["tuning_run"])
            )
    if not tables:
        return None
    scores = pd.concat(tables, ignore_index=True)
    if not np.isfinite(scores.primary_relative_mse.to_numpy(dtype=np.float64)).all():
        raise ValueError("search scores must be finite")
    return scores


def _read_test_targets(repo, preparation, contract, mask):
    raw = json.loads((preparation / "input/raw_input_hashes.json").read_text())
    verify_hashes(repo / raw["dataset_root"], raw["hashes"])
    rows = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    native = native_time_contract(contract)
    target = read_selected_traces(
        repo / raw["dataset_root"],
        rows.loc[mask.split.eq("test")],
        native["time_sample_count"],
        native["sample_interval_us"],
    )
    return resample_forge_traces(target, contract)


def _restoration_scales(preparation, contract, mask, index):
    """Oracle test RMS from the sealed scales; non-oracle RMS from observed traces only."""
    oracle = pd.read_parquet(preparation / "input/trace_scales.parquet").set_index("cell_id")
    observed = pd.read_parquet(preparation / "input/observed_index.parquet").cell_id.to_numpy()
    if not np.array_equal(observed, mask.loc[mask.split.eq("observed"), "cell_id"]):
        raise ValueError("observed array index does not match the mask")
    grids = json.loads((preparation / "input/candidate_grids.json").read_text())
    idw, spacing = log_rms_idw_scales(
        np.load(preparation / "input/observed_waveforms.npy", mmap_mode="r"),
        observed,
        index.cell_id.to_numpy(),
        tuple(contract["spatial_shape"]),
        grids,
        LOG_RMS_IDW_SETTINGS,
    )
    scales = pd.DataFrame(
        {
            "cell_id": index.cell_id.to_numpy(),
            "oracle_rms": oracle.loc[index.cell_id, "rms"].to_numpy(),
            "idw_rms": idw,
        }
    )
    return scales, spacing


def _score_run(run, target, index, scales, output):
    """Score one prediction: oracle and non-oracle restoration, and the scale-invariant error."""
    prediction = np.load(run["prediction"])
    ratio = scales.idw_rms.to_numpy() / scales.oracle_rms.to_numpy()
    non_oracle = (prediction.astype(np.float64) * ratio[:, None]).astype(np.float32)
    label = "deterministic" if run["seed"] is None else f"seed_{run['seed']}"
    path = output / "runs" / run["method_id"] / label
    frames = {}
    for restoration, values in (
        ("trace_oracle", prediction),
        ("non_oracle_log_rms_idw", non_oracle),
    ):
        metrics, traces = evaluate_traces(values, target, index)
        verify_saved_metrics(metrics, traces)
        verify_metrics_from_waveforms(metrics, values, target, index.clean_target.to_numpy())
        write_json(path / f"{restoration}_metrics.json", metrics)
        traces.to_parquet(path / f"{restoration}_metrics_by_trace.parquet", index=False)
        frames[restoration] = (metrics, traces)
    invariant = scale_invariant_relative_mse(prediction, target)
    pd.DataFrame({"cell_id": index.cell_id, "relative_mse": invariant}).to_parquet(
        path / "scale_invariant_by_trace.parquet", index=False
    )
    rows = []
    errors = {"scale_invariant": invariant}
    clean = index.clean_target.to_numpy()
    for restoration in RESTORATIONS:
        if restoration == "scale_invariant":
            per_trace, metrics = invariant, None
        else:
            metrics, traces = frames[restoration]
            per_trace = traces.relative_mse.to_numpy()
            errors[restoration] = per_trace
        for domain, selected in (("clean_target", clean), ("all_eligible", np.ones_like(clean))):
            summary = error_summary(per_trace[selected])
            rows.append(
                {
                    "method_id": run["method_id"],
                    "seed": run["seed"],
                    "ema_decay": run["ema_decay"],
                    "restoration": restoration,
                    "domain": domain,
                    "trace_count": summary["trace_count"],
                    "mean_relative_mse": summary["mean"]
                    if metrics is None
                    else metrics[domain]["masked_trace_relative_mse"],
                    "median_relative_mse": summary["median"],
                    "global_nmse": None if metrics is None else metrics[domain]["global_nmse"],
                    "median_correlation": None
                    if metrics is None
                    else metrics[domain]["correlation"]["median"],
                }
            )
    return errors, rows


def _method_summary(seed_metrics):
    rows = []
    keys = ["method_id", "restoration", "domain"]
    for key, group in seed_metrics.groupby(keys, sort=False):
        row = dict(zip(keys, key, strict=True), seed_count=len(group))
        for metric in (
            "mean_relative_mse",
            "median_relative_mse",
            "global_nmse",
            "median_correlation",
        ):
            values = group[metric].dropna()
            mean, sd = seed_mean_sd(values) if len(values) else (None, None)
            row.update({f"{metric}_seed_mean": mean, f"{metric}_seed_sd": sd})
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_averaged_errors(errors, index):
    """Long table: one row per (restoration, test trace) with one column per method."""
    tables = []
    for restoration in RESTORATIONS:
        table = index[["trace_id", "cell_id", "source_cell", "clean_target"]].copy()
        table.insert(0, "restoration", restoration)
        for method in METHODS:
            table[method] = seed_average_errors([run[restoration] for run in errors[method]])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _contrast_table(averaged, bootstrap):
    tables = []
    for family, restoration, domain in CONTRAST_FAMILIES:
        rows = averaged.loc[averaged.restoration.eq(restoration)]
        if domain == "clean_target":
            rows = rows.loc[rows.clean_target]
        clusters = rows.source_cell.to_numpy()
        weights = cluster_bootstrap_weights(
            len(np.unique(clusters)), bootstrap["resamples"], bootstrap["seed"]
        )
        table = paired_contrasts(
            {method: rows[method].to_numpy() for method in METHODS}, clusters, weights
        )
        table.insert(0, "domain", domain)
        table.insert(0, "restoration", restoration)
        table.insert(0, "family", family)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
