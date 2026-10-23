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
      <method>:          # every method of the grid
        - {seed: <int or null>, tuning_run: <dir>, candidate: <name>}

A prediction is ``<tuning_run>/<candidate>/prediction.npy``.

A grid that adds methods to a completed study (a preparation derived by
``derive_forge_preparation``) declares ``evaluation.reference_evaluation``
(``{path, manifest_sha256}`` of the parent study's completed final evaluation on
the parent preparation) and ``evaluation.contrasts``. The reference methods'
seed-averaged errors enter the contrasts unchanged after their rows, test traces
and non-oracle restoration scales are checked to equal this evaluation's.

A C3 preparation (``dataset_id: seg_c3_na``) uses the observed global RMS: its
predictions are already physical, so the restorations are ``observed_global_rms``
and ``scale_invariant`` (no oracle or IDW scales). Its test targets come from
the interim C3 trace array (``read_c3_scored_targets``), its grid must declare
``evaluation.contrasts`` and no reference evaluation, and a fourth Holm family
compares the seed-averaged per-trace SNR dB (``left - right``; positive favours
the left method).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.c3_fair_grid_truth import read_c3_scored_targets
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import read_selected_traces, verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import (
    execution_identity,
    read_search_results,
    utc_now,
)
from seis_interp.evaluation.fair_grid_trace_snr import (
    paired_trace_snr_contrasts,
    seed_averaged_trace_snr,
    trace_snr_by_trace,
)
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
from seis_interp.evaluation.trace_snr import summarize_trace_snr
from seis_interp.forge_mvp_config import ADDED_METHODS, TUNABLE_NEURAL_METHODS
from seis_interp.processing.c3_fair_grid_contract import C3_DATASET_ID
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
# Observed-global-RMS (C3) preparations: physical predictions, no trace scales.
GLOBAL_RMS_RESTORATIONS = ("observed_global_rms", "scale_invariant")
GLOBAL_RMS_CONTRAST_FAMILIES = (
    ("primary", "observed_global_rms", "clean_target"),
    ("secondary_all_eligible", "observed_global_rms", "all_eligible"),
    ("secondary_scale_invariant", "scale_invariant", "clean_target"),
)
TRACE_SNR_CONTRAST_FAMILY = ("secondary_trace_snr_db", "observed_global_rms", "clean_target")
FINAL_RUN_KEYS = {"seed", "tuning_run", "candidate"}
SEARCH_STAGES = ("S1", "S2", "S3", "E")
ERROR_KEYS = ["restoration", "trace_id", "cell_id", "source_cell", "clean_target"]


def evaluate_forge_fair_grid(repo: Path, evaluation_config: Path) -> Path:
    config = yaml.safe_load(evaluation_config.read_text())
    _validate_evaluation_config(config)
    preparation = repo / config["preparation"]
    seal, contract = _final_preparation(preparation)
    oracle = seal["amplitude_scaling"] == "trace_oracle"
    preparation_hash = sha256_file(preparation / "preparation_manifest.json")
    grid_path = repo / "studies" / contract["study_id"] / "grid.yaml"
    grid, grid_hash = yaml.safe_load(grid_path.read_text()), sha256_file(grid_path)
    if grid["study_id"] != contract["study_id"]:
        raise ValueError("grid.yaml belongs to another study")
    methods = _grid_methods(grid)
    if set(config["final_runs"]) != set(methods):
        raise ValueError("final_runs must list every method")
    if config["bootstrap"] != grid["evaluation"]["bootstrap"]:
        raise ValueError("bootstrap differs from the pre-registered grid.yaml")
    if not oracle and (
        "contrasts" not in grid["evaluation"] or "reference_evaluation" in grid["evaluation"]
    ):
        raise ValueError("a global-RMS evaluation declares contrasts and no reference evaluation")
    contrasts = tuple(
        tuple(contrast) for contrast in grid["evaluation"].get("contrasts", FAIR_GRID_CONTRASTS)
    )
    reference = _reference_evaluation(repo, grid, seal)
    final_preparation = {"path": config["preparation"], "manifest_sha256": preparation_hash}
    selections = {
        method: _selection(repo, grid, grid_hash, method, final_preparation) for method in methods
    }
    mask = pd.read_parquet(preparation / MASK_PATH)
    test = mask.split.eq("test").to_numpy()
    index = mask.loc[test].reset_index(drop=True)
    # Verify every prediction and search record before any target is opened.
    runs = [
        _final_run(repo, method, entry, preparation_hash, index, mask, contract, selections[method])
        for method in methods
        for entry in config["final_runs"][method]
    ]
    reference_errors = None if reference is None else _reference_errors(reference, index)
    compared = [*methods, *([] if reference_errors is None else reference["methods"])]
    if {m for contrast in contrasts for m in contrast[1:3]} - set(compared):
        raise ValueError("a contrast names a method that is not evaluated or referenced")
    for method in methods:
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
            method: sha256_file(_selection_path(repo, grid, method)) for method in methods
        },
        mask_hash=seal["mask_hash"],
        selection_domain="final_test",
        independent_test_estimate=True,
        bootstrap=config["bootstrap"],
        cluster="source_cell",
        contrasts=[list(contrast) for contrast in contrasts],
        **(
            {}
            if reference is None
            else {
                "reference_evaluation": {
                    key: reference[key] for key in ("path", "manifest_sha256", "methods")
                }
            }
        ),
        **(
            {"non_oracle_restoration": {**LOG_RMS_IDW_SETTINGS, "interpolated_quantity": "log_rms"}}
            if oracle
            else {
                "amplitude_scaling": "observed_global_rms",
                "restorations": list(GLOBAL_RMS_RESTORATIONS),
                "trace_snr_db": "10 log10(target energy / error energy) per test trace",
            }
        ),
        final_runs=[{k: v for k, v in run.items() if k != "prediction"} for run in runs],
        test_waveform_reads=0,
    )
    write_json(output / "manifest.json", record)
    try:
        scales = None
        if oracle:
            # Restoration scales read observed traces only; they are checked before any target.
            scales, spacing = _restoration_scales(preparation, contract, mask, index)
            record["non_oracle_restoration"]["axis_scales_m"] = list(spacing)
            if reference is not None and not scales.equals(reference["scales"]):
                raise ValueError(
                    "non-oracle restoration scales differ from the reference evaluation"
                )
        target = _read_test_targets(repo, preparation, contract, mask)
        record.update(test_waveform_reads=1, test_waveforms_read=len(target))
        aggregate = output / "aggregate"
        aggregate.mkdir()
        if oracle:
            scales.to_parquet(aggregate / "non_oracle_trace_scales.parquet", index=False)
        errors, seed_rows = {}, []
        for run in runs:
            run_errors, rows = (
                _score_run(run, target, index, scales, output)
                if oracle
                else _score_global_rms_run(run, target, index, output)
            )
            errors.setdefault(run["method_id"], []).append(run_errors)
            seed_rows.extend(rows)
        seed_metrics = pd.DataFrame(seed_rows)
        seed_metrics.to_csv(aggregate / "seed_metrics.csv", index=False)
        summary = _method_summary(seed_metrics) if oracle else _trace_snr_summary(seed_metrics)
        summary.to_csv(aggregate / "method_summary.csv", index=False)
        restorations = RESTORATIONS if oracle else GLOBAL_RMS_RESTORATIONS
        averaged = _seed_averaged_errors(errors, index, methods, restorations)
        averaged.to_parquet(aggregate / "seed_averaged_errors.parquet", index=False)
        if reference_errors is not None:
            averaged = pd.concat([averaged, reference_errors], axis=1)
        if oracle:
            table = _contrast_table(averaged, config["bootstrap"], compared, contrasts)
        else:
            # A non-finite per-trace SNR is counted in seed_metrics; it skips only the SNR
            # family so that the one test read still yields the relative-MSE contrasts.
            nonfinite = sum(
                int(np.count_nonzero(~np.isfinite(run["trace_snr_db"])))
                for method in methods
                for run in errors[method]
            )
            record["trace_snr_contrasts"] = (
                f"skipped: {nonfinite} non-finite per-trace SNR values" if nonfinite else "computed"
            )
            snr = None if nonfinite else _seed_averaged_trace_snr(errors, index, methods)
            if snr is not None:
                snr.to_parquet(aggregate / "seed_averaged_trace_snr_db.parquet", index=False)
            table = _global_rms_contrast_table(
                averaged, snr, config["bootstrap"], methods, contrasts
            )
        table.to_csv(aggregate / "contrasts.csv", index=False)
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


def _grid_methods(grid):
    """Method IDs of the grid in the fixed method order (the six MVP methods first)."""
    ids = {m for spec in grid["methods"].values() for m in _spec_method_ids(spec)}
    if ids - {*METHODS, *ADDED_METHODS}:
        raise ValueError("grid.yaml lists an unknown method")
    return [method for method in (*METHODS, *ADDED_METHODS) if method in ids]


def _spec_method_ids(spec):
    return [spec["method_id"], *([spec["pair_method_id"]] if "pair_method_id" in spec else [])]


def _reference_evaluation(repo, grid, seal):
    """The parent study's completed final evaluation on this preparation's parent, or None."""
    declared = grid["evaluation"].get("reference_evaluation")
    if declared is None:
        return None
    if set(declared) != {"path", "manifest_sha256"}:
        raise ValueError("reference_evaluation needs path and manifest_sha256")
    path = repo / declared["path"]
    if sha256_file(path / "manifest.json") != declared["manifest_sha256"]:
        raise ValueError("reference evaluation manifest differs from grid.yaml")
    manifest = json.loads((path / "manifest.json").read_text())
    names = ("seed_averaged_errors.parquet", "non_oracle_trace_scales.parquet")
    verify_hashes(
        path, {f"aggregate/{name}": manifest["artifacts"][f"aggregate/{name}"] for name in names}
    )
    parent = seal.get("derived_from", {})
    if (
        manifest["status"] != "completed"
        or manifest["preparation_hash"] != parent.get("manifest_sha256")
        or manifest["mask_hash"] != seal["mask_hash"]
        or manifest["bootstrap"] != grid["evaluation"]["bootstrap"]
    ):
        raise ValueError("reference evaluation is not the completed evaluation of the parent")
    order = (*METHODS, *ADDED_METHODS)
    methods = sorted({run["method_id"] for run in manifest["final_runs"]}, key=order.index)
    if set(methods) & set(_grid_methods(grid)):
        raise ValueError("the grid re-evaluates a method of its reference evaluation")
    return {
        **declared,
        "methods": methods,
        "errors": pd.read_parquet(path / "aggregate/seed_averaged_errors.parquet"),
        "scales": pd.read_parquet(path / "aggregate/non_oracle_trace_scales.parquet"),
    }


def _reference_errors(reference, index):
    """Reference method columns, on exactly this evaluation's error-table rows."""
    table = reference["errors"]
    expected = pd.concat(
        [
            index[ERROR_KEYS[1:]].assign(restoration=restoration)[ERROR_KEYS]
            for restoration in RESTORATIONS
        ],
        ignore_index=True,
    )
    if list(table.columns) != [*ERROR_KEYS, *reference["methods"]] or not table[ERROR_KEYS].equals(
        expected
    ):
        raise ValueError("reference errors cover other rows or methods")
    return table[reference["methods"]]


def _validate_evaluation_config(config):
    if set(config) - {"preparation", "bootstrap", "final_runs"}:
        raise ValueError("unsupported final-evaluation keys")
    bootstrap = config["bootstrap"]
    if set(bootstrap) != {"resamples", "seed"} or any(
        isinstance(v, bool) or not isinstance(v, int) for v in bootstrap.values()
    ):
        raise ValueError("bootstrap needs integer resamples and seed")
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
    if seal.get("amplitude_scaling") != "trace_oracle" and not (
        seal.get("amplitude_scaling") == "observed_global_rms"
        and seal.get("dataset_id") == C3_DATASET_ID
    ):
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
        or any(
            "metrics" in Path(name).name and Path(name).parts[0] != "source"
            for name in manifest["artifacts"]
        )
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
                shadows = keys if method in TUNABLE_NEURAL_METHODS else [None]
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
                rows = read_search_results(tuning / "search_results.csv")
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
    if contract.get("dataset_id") == C3_DATASET_ID:
        return read_c3_scored_targets(repo, preparation, mask)
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


def _score_global_rms_run(run, target, index, output):
    """Score one physical prediction: relative MSE, scale-invariant error and trace SNR dB."""
    prediction = np.load(run["prediction"])
    label = "deterministic" if run["seed"] is None else f"seed_{run['seed']}"
    path = output / "runs" / run["method_id"] / label
    metrics, traces = evaluate_traces(prediction, target, index)
    verify_saved_metrics(metrics, traces)
    verify_metrics_from_waveforms(metrics, prediction, target, index.clean_target.to_numpy())
    write_json(path / "observed_global_rms_metrics.json", metrics)
    traces.to_parquet(path / "observed_global_rms_metrics_by_trace.parquet", index=False)
    invariant = scale_invariant_relative_mse(prediction, target)
    pd.DataFrame({"cell_id": index.cell_id, "relative_mse": invariant}).to_parquet(
        path / "scale_invariant_by_trace.parquet", index=False
    )
    snr = trace_snr_by_trace(target, prediction)
    pd.DataFrame({"cell_id": index.cell_id, "trace_snr_db": snr}).to_parquet(
        path / "trace_snr_db_by_trace.parquet", index=False
    )
    errors = {
        "observed_global_rms": traces.relative_mse.to_numpy(),
        "scale_invariant": invariant,
        "trace_snr_db": snr,
    }
    rows = []
    clean = index.clean_target.to_numpy()
    for restoration in GLOBAL_RMS_RESTORATIONS:
        physical = restoration == "observed_global_rms"
        for domain, selected in (("clean_target", clean), ("all_eligible", np.ones_like(clean))):
            summary = error_summary(errors[restoration][selected])
            snr_summary = summarize_trace_snr(snr[selected])
            nonfinite = sum(
                snr_summary[f"trace_snr_{kind}_count"]
                for kind in ("positive_infinity", "negative_infinity", "undefined")
            )
            rows.append(
                {
                    "method_id": run["method_id"],
                    "seed": run["seed"],
                    "ema_decay": run["ema_decay"],
                    "restoration": restoration,
                    "domain": domain,
                    "trace_count": summary["trace_count"],
                    "mean_relative_mse": metrics[domain]["masked_trace_relative_mse"]
                    if physical
                    else summary["mean"],
                    "median_relative_mse": summary["median"],
                    "global_nmse": metrics[domain]["global_nmse"] if physical else None,
                    "median_correlation": metrics[domain]["correlation"]["median"]
                    if physical
                    else None,
                    "mean_trace_snr_db": snr_summary["mean_trace_snr_db"] if physical else None,
                    "trace_snr_nonfinite_count": nonfinite if physical else None,
                    "global_snr_db": _global_snr_db(metrics[domain]["global_nmse"])
                    if physical
                    else None,
                }
            )
    return errors, rows


def _global_snr_db(global_nmse):
    """``10 log10(target energy / error energy)`` summed over the domain; None without error."""
    return None if global_nmse is None or global_nmse <= 0 else float(-10 * np.log10(global_nmse))


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


def _trace_snr_summary(seed_metrics):
    """``_method_summary`` plus the seed mean/sd of the mean trace and global SNR dB.

    A seed whose SNR has a non-finite trace has no mean SNR, so neither has its
    method row; the summed non-finite count shows why.
    """
    rows = []
    for _, group in seed_metrics.groupby(["method_id", "restoration", "domain"], sort=False):
        row = {}
        for metric in ("mean_trace_snr_db", "global_snr_db"):
            values = group[metric]
            mean, sd = (None, None) if values.isna().any() else seed_mean_sd(values)
            row.update({f"{metric}_seed_mean": mean, f"{metric}_seed_sd": sd})
        counts = group.trace_snr_nonfinite_count
        row["trace_snr_nonfinite_count"] = None if counts.isna().all() else int(counts.sum())
        rows.append(row)
    return pd.concat([_method_summary(seed_metrics), pd.DataFrame(rows)], axis=1)


def _seed_averaged_errors(errors, index, methods, restorations=RESTORATIONS):
    """Long table: one row per (restoration, test trace) with one column per method."""
    tables = []
    for restoration in restorations:
        table = index[["trace_id", "cell_id", "source_cell", "clean_target"]].copy()
        table.insert(0, "restoration", restoration)
        for method in methods:
            table[method] = seed_average_errors([run[restoration] for run in errors[method]])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _contrast_table(averaged, bootstrap, methods, contrasts, families=CONTRAST_FAMILIES):
    tables = []
    for family, restoration, domain in families:
        rows = averaged.loc[averaged.restoration.eq(restoration)]
        if domain == "clean_target":
            rows = rows.loc[rows.clean_target]
        clusters = rows.source_cell.to_numpy()
        weights = cluster_bootstrap_weights(
            len(np.unique(clusters)), bootstrap["resamples"], bootstrap["seed"]
        )
        table = paired_contrasts(
            {method: rows[method].to_numpy() for method in methods}, clusters, weights, contrasts
        )
        table.insert(0, "domain", domain)
        table.insert(0, "restoration", restoration)
        table.insert(0, "family", family)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _seed_averaged_trace_snr(errors, index, methods):
    """One row per test trace with each method's seed-mean per-trace SNR dB."""
    table = index[["trace_id", "cell_id", "source_cell", "clean_target"]].copy()
    for method in methods:
        table[method] = seed_averaged_trace_snr([run["trace_snr_db"] for run in errors[method]])
    return table


def _global_rms_contrast_table(averaged, snr, bootstrap, methods, contrasts):
    """Relative-MSE families (a negative difference favours the left method), then trace SNR.

    The SNR family's difference is ``left - right`` dB, so a positive one favours left;
    without finite per-trace SNR (``snr`` None) only the relative-MSE families are returned.
    """
    errors = _contrast_table(averaged, bootstrap, methods, contrasts, GLOBAL_RMS_CONTRAST_FAMILIES)
    errors.insert(3, "left_better_when", "difference_negative")
    if snr is None:
        return errors
    family, restoration, domain = TRACE_SNR_CONTRAST_FAMILY
    rows = snr.loc[snr.clean_target] if domain == "clean_target" else snr
    clusters = rows.source_cell.to_numpy()
    weights = cluster_bootstrap_weights(
        len(np.unique(clusters)), bootstrap["resamples"], bootstrap["seed"]
    )
    table = paired_trace_snr_contrasts(
        {method: rows[method].to_numpy() for method in methods}, clusters, weights, contrasts
    )
    for position, (name, value) in enumerate(
        (
            ("family", family),
            ("restoration", restoration),
            ("domain", domain),
            ("left_better_when", "difference_positive"),
        )
    ):
        table.insert(position, name, value)
    return pd.concat([errors, table], ignore_index=True)
