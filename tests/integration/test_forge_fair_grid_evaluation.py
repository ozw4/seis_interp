import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import read_selected_traces, verify_hashes, write_json
from seis_interp.evaluation.forge_fair_grid_statistics import FAIR_GRID_CONTRASTS
from seis_interp.pipelines import evaluate_forge_fair_grid as pipeline
from seis_interp.pipelines.evaluate_forge_fair_grid import evaluate_forge_fair_grid
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.tune_forge_matched import tune_forge_matched
from seis_interp.processing.forge_mvp_contract import METHODS
from tests.fixtures.forge_mvp import tiny_forge_study

NEURAL = ("ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid")
FINAL_SEEDS = (7311, 7312, 7313)
BOOTSTRAP = {"resamples": 400, "seed": 600703}
# Grid method -> its method ids, as in study_060 grid.yaml.
GRID_METHODS = {
    "pocs_grid": ("pocs_grid",),
    "drr_grid": ("drr_grid",),
    "ccnet5d_grid": ("ccnet5d_grid",),
    "nersi_real": ("nersi_real",),
    "regsi": ("regsi_real", "regsi_grid"),
}
KEYS = ("0.999", "0.9995", "0.9998")
SELECTED_EMA = "0.999"
DIVERGED = "RuntimeError: non-finite CCNet-5D training loss at step 17"
# method -> (stage, status) of each recorded search run; R runs never enter the curves.
SEARCH = {
    "pocs_grid": [("S1", "completed")] * 3,
    "drr_grid": [("S1", "completed")] * 4,
    "ccnet5d_grid": [("S1", "completed"), ("S1", "diverged"), ("S2", "completed")],
    "nersi_real": [("S1", "completed"), ("E", "completed"), ("R", "completed")],
    "regsi_real": [("S1", "completed"), ("S3", "completed"), ("E", "completed")],
    "regsi_grid": [("R", "completed")],
}
CURVE_SIZES = {
    "pocs_grid": 3,
    "drr_grid": 4,
    "ccnet5d_grid": 9,
    "nersi_real": 6,
    "regsi_real": 9,
}


@pytest.fixture
def final_preparation(tmp_path, monkeypatch):
    study, contract, *_ = tiny_forge_study(tmp_path)
    contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    identity = {"git_commit": "testcommit", "dirty_worktree": True}
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.tune_forge_matched.execution_identity", lambda _: identity
    )
    monkeypatch.setattr(pipeline, "execution_identity", lambda _: identity)
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    preparation = prepare_forge_mvp(tmp_path, study)
    yield tmp_path, preparation
    torch.set_num_threads(previous)


def _test_frame(preparation):
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    return mask.loc[mask.split.eq("test")].reset_index(drop=True)


def _targets(root, preparation):
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    rows = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    return read_selected_traces(root / "raw", rows.loc[mask.split.eq("test")], 9, 1000)


def _write_tuning_run(root, name, method, preparation, index, files, **manifest_keys):
    """A tuning-run directory with the tuner's manifest, index and per-candidate files."""
    run = root / "runs/tuning" / name
    index = index.copy()
    index["prediction_row"] = np.arange(len(index))
    run.mkdir(parents=True)
    index.to_parquet(run / "prediction_index.parquet", index=False)
    for relative, value in files.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, np.ndarray):
            np.save(path, value, allow_pickle=False)
        elif isinstance(value, pd.DataFrame):
            value.to_csv(path, index=False)
        else:
            path.write_text(yaml.safe_dump(value, sort_keys=False))
    manifest = {
        "status": "completed",
        "method_id": method,
        "preparation": str(preparation.relative_to(root)),
        "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
        "artifacts": {
            str(p.relative_to(run)): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file()
        },
        **manifest_keys,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return str(run.relative_to(root))


def _final_runs(root, preparation, pocs_run, pocs_yaml):
    """Final runs of every method and the stage.json run records they must match."""
    target = _targets(root, preparation)
    rng = np.random.default_rng(9)
    runs = {"pocs_grid": [{"seed": None, "tuning_run": pocs_run, "candidate": "final_pocs"}]}
    planned = {
        "pocs_grid": [
            {"name": "final_pocs", "method_id": "pocs_grid", "sha256": sha256_file(pocs_yaml)}
        ]
    }
    for level, method in enumerate(("drr_grid", *NEURAL)):
        seeds = (None,) if method == "drr_grid" else FINAL_SEEDS
        runs[method], planned[method] = [], []
        config = yaml.safe_load((preparation / "configs" / f"{method}.yaml").read_text())
        for seed in seeds:
            name = f"final_{method}_{seed}"
            resolved = {**config, "model_seed": seed}
            if method in NEURAL:
                resolved["training"] = {**config["training"], "ema_decay": float(SELECTED_EMA)}
            noise = rng.normal(0, 0.05 * (level + 1), target.shape) * target.std()
            search = {
                "method_id": method,
                "purpose": "final_evaluation",
                "evaluate": False,
                "device": "cpu",
                "candidates": [{"name": name}],
            }
            files = {
                "search.yaml": search,
                f"{name}/prediction.npy": (target * (1 + 0.1 * level) + noise).astype(np.float32),
                f"{name}/resolved_config.yaml": resolved,
            }
            tuning = _write_tuning_run(
                root,
                name,
                method,
                preparation,
                _test_frame(preparation),
                files,
                purpose="final_evaluation",
                evaluate=False,
            )
            runs[method].append({"seed": seed, "tuning_run": tuning, "candidate": name})
            planned[method].append(
                {
                    "name": name,
                    "method_id": method,
                    "sha256": sha256_file(root / tuning / "search.yaml"),
                }
            )
    return runs, planned


def _search_inputs(root, preparation):
    """Recorded validation-holdout search runs (``rule_inputs``) of every method."""
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    validation = root / "runs/validation_prep"
    validation.mkdir(parents=True)
    (validation / "preparation_manifest.json").write_text(
        json.dumps(
            {"selection_domain": "validation_holdout", "parent_mask_hash": seal["mask_hash"]}
        )
    )
    inputs = {}
    for method, entries in SEARCH.items():
        inputs[method] = []
        for number, (stage, status) in enumerate(entries):
            name = f"{stage.lower()}_{method}_{number}"
            keys = [None] if method in ("pocs_grid", "drr_grid") else list(KEYS)
            files, manifest = {}, {"selection_domain": "validation_holdout"}
            if status == "completed":
                scores = 0.9 - 0.05 * number - 0.01 * np.arange(len(keys))
                files["search_results.csv"] = pd.DataFrame(
                    {"name": name, "ema_decay": keys, "primary_relative_mse": scores}
                )
            else:
                manifest.update(status="failed", failure_reason=DIVERGED)
            tuning = _write_tuning_run(
                root, name, method, validation, _test_frame(preparation), files, **manifest
            )
            inputs[method].append(
                {
                    "run": name,
                    "stage": stage,
                    "method_id": method,
                    "tuning_run": tuning,
                    "tuning_manifest_sha256": sha256_file(root / tuning / "manifest.json"),
                    "status": status,
                }
            )
    return inputs


def _pocs_tuning_run(root, preparation):
    """Use the real tuner once so the evaluator reads its actual output layout."""
    search = root / "pocs_final.yaml"
    search.write_text(
        yaml.safe_dump(
            {
                "method_id": "pocs_grid",
                "purpose": "final_evaluation",
                "evaluate": False,
                "device": "cpu",
                "candidates": [{"name": "final_pocs"}],
            }
        )
    )
    return str(tune_forge_matched(root, preparation, search).relative_to(root)), search


def _selection_path(root, grid_method):
    return root / "runs/tiny_forge_mvp/fair_grid" / grid_method / "final/stage.json"


def _write_registration(root, preparation, planned, inputs):
    """grid.yaml of the tiny study and one selection record per grid method."""
    grid = {
        "study_id": "tiny_forge_mvp",
        "selection": {
            "diverged_runs": {"failure_reason_pattern": "non-?finite", "reported_score": 1.0}
        },
        "ema_eval_decays": [float(key) for key in KEYS],
        "evaluation": {"bootstrap": BOOTSTRAP},
        "methods": {
            key: {"method_id": ids[0], **({"pair_method_id": ids[1]} if ids[1:] else {})}
            for key, ids in GRID_METHODS.items()
        },
    }
    grid_path = root / "studies/tiny_forge_mvp/grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid, sort_keys=False))
    final = {
        "path": str(preparation.relative_to(root)),
        "manifest_sha256": sha256_file(preparation / "preparation_manifest.json"),
    }
    for key, ids in GRID_METHODS.items():
        classical = key in ("pocs_grid", "drr_grid")
        write_json(
            _selection_path(root, key),
            {
                "study_id": "tiny_forge_mvp",
                "method": key,
                "stage": "final",
                "grid_sha256": sha256_file(grid_path),
                "preparations": {"validation": {"path": "runs/validation_prep"}, "final": final},
                "rule_inputs": [entry for m in ids for entry in inputs[m]],
                "decision": {} if classical else {"ema_decay": SELECTED_EMA},
                "runs": [run for m in ids for run in planned[m]],
            },
        )


@pytest.fixture
def evaluation(final_preparation):
    root, preparation = final_preparation
    runs, planned = _final_runs(root, preparation, *_pocs_tuning_run(root, preparation))
    _write_registration(root, preparation, planned, _search_inputs(root, preparation))
    config = {
        "preparation": str(preparation.relative_to(root)),
        "bootstrap": BOOTSTRAP,
        "final_runs": runs,
    }
    path = root / "final_evaluation.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return root, preparation, config, path


def _count_target_reads(monkeypatch):
    calls = []

    def counted(*args, **kwargs):
        values = read_selected_traces(*args, **kwargs)
        calls.append(len(values))
        return values

    monkeypatch.setattr(pipeline, "read_selected_traces", counted)
    return calls


def test_final_evaluation_scores_every_run_once_and_records_statistics(evaluation, monkeypatch):
    root, preparation, config, path = evaluation
    reads = _count_target_reads(monkeypatch)
    output = evaluate_forge_fair_grid(root, path)
    index = _test_frame(preparation)
    assert reads == [len(index)]
    assert output.parent == root / "runs/tiny_forge_mvp"
    assert output.name.endswith("_testcommit_final_evaluation")
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["test_waveform_reads"] == 1
    assert record["test_waveforms_read"] == len(index)
    assert record["non_oracle_restoration"]["axis_scales_m"] == [200.0, 50.0, 200.0, 50.0]
    assert record["grid_sha256"] == sha256_file(root / "studies/tiny_forge_mvp/grid.yaml")
    assert record["bootstrap"] == BOOTSTRAP
    assert set(record["selection_records"]) == set(METHODS)
    assert {run["ema_decay"] for run in record["final_runs"] if run["method_id"] in NEURAL} == {
        SELECTED_EMA
    }
    aggregate = output / "aggregate"
    seed_metrics = pd.read_csv(aggregate / "seed_metrics.csv")
    assert len(seed_metrics) == (2 + 3 * len(NEURAL)) * 3 * 2
    summary = pd.read_csv(aggregate / "method_summary.csv")
    primary = summary.loc[
        summary.restoration.eq("trace_oracle") & summary.domain.eq("clean_target")
    ]
    assert primary.set_index("method_id").seed_count.to_dict() == {
        "pocs_grid": 1,
        "drr_grid": 1,
        **dict.fromkeys(NEURAL, 3),
    }
    averaged = pd.read_parquet(aggregate / "seed_averaged_errors.parquet")
    oracle = averaged.loc[averaged.restoration.eq("trace_oracle")].reset_index(drop=True)
    assert oracle.cell_id.tolist() == index.cell_id.tolist()
    for method in NEURAL:
        per_seed = [
            pd.read_parquet(
                output / f"runs/{method}/seed_{seed}/trace_oracle_metrics_by_trace.parquet"
            ).relative_mse
            for seed in FINAL_SEEDS
        ]
        np.testing.assert_allclose(oracle[method], np.mean(per_seed, axis=0))
        row = primary.set_index("method_id").loc[method]
        clean = [series[index.clean_target].mean() for series in per_seed]
        assert row.mean_relative_mse_seed_mean == pytest.approx(np.mean(clean))
        assert row.mean_relative_mse_seed_sd == pytest.approx(np.std(clean, ddof=1))
    contrasts = pd.read_csv(aggregate / "contrasts.csv")
    assert len(contrasts) == 4 * len(FAIR_GRID_CONTRASTS)
    first = contrasts.loc[contrasts.family.eq("primary")].reset_index(drop=True)
    assert first.contrast.tolist() == [c[0] for c in FAIR_GRID_CONTRASTS]
    clean = oracle.loc[oracle.clean_target]
    for row, (_, left, right, _) in zip(first.itertuples(), FAIR_GRID_CONTRASTS, strict=True):
        assert row.estimate == pytest.approx((clean[left] - clean[right]).mean())
        assert row.cluster_count == clean.source_cell.nunique()
        assert row.bootstrap_resamples == BOOTSTRAP["resamples"]
    assert np.all(contrasts.holm_p_value >= contrasts.bootstrap_p_value)
    scales = pd.read_parquet(aggregate / "non_oracle_trace_scales.parquet")
    assert scales.shape == (len(index), 3)
    assert np.all(scales[["oracle_rms", "idw_rms"]].to_numpy() > 0)
    # The curves cover the recorded S1-E configurations; a diverged run enters at 1.0.
    curves = pd.read_csv(aggregate / "expected_best_curves.csv")
    assert curves.groupby("method_id").size().to_dict() == CURVE_SIZES
    search = pd.read_csv(aggregate / "search_scores.csv")
    assert set(search.stage) == {"S1", "S2", "S3", "E"}
    diverged = search.loc[search.diverged]
    assert diverged.name.unique().tolist() == ["s1_ccnet5d_grid_1"]
    assert diverged.primary_relative_mse.tolist() == [1.0, 1.0, 1.0]
    # A second run reproduces the bootstrap exactly.
    second = evaluate_forge_fair_grid(root, path)
    pd.testing.assert_frame_equal(contrasts, pd.read_csv(second / "aggregate/contrasts.csv"))


def test_non_oracle_restoration_rescales_the_unit_prediction(evaluation):
    root, preparation, config, path = evaluation
    output = evaluate_forge_fair_grid(root, path)
    scales = pd.read_parquet(output / "aggregate/non_oracle_trace_scales.parquet")
    entry = config["final_runs"]["drr_grid"][0]
    prediction = np.load(root / entry["tuning_run"] / entry["candidate"] / "prediction.npy")
    target = _targets(root, preparation)
    restored = (
        prediction.astype(np.float64) / scales.oracle_rms.to_numpy()[:, None]
    ) * scales.idw_rms.to_numpy()[:, None]
    expected = np.sum((restored - target) ** 2, 1) / np.sum(target.astype(np.float64) ** 2, 1)
    saved = pd.read_parquet(
        output / "runs/drr_grid/deterministic/non_oracle_log_rms_idw_metrics_by_trace.parquet"
    )
    np.testing.assert_allclose(saved.relative_mse, expected, rtol=1e-5)
    trace_scales = pd.read_parquet(preparation / "input/trace_scales.parquet").set_index("cell_id")
    np.testing.assert_array_equal(
        scales.oracle_rms, trace_scales.loc[scales.cell_id, "rms"].to_numpy()
    )


def _rewrite(path, value):
    path.chmod(0o644)
    path.write_text(json.dumps(value))


def _rebind(tuning, relative):
    manifest = json.loads((tuning / "manifest.json").read_text())
    manifest["artifacts"][relative] = sha256_file(tuning / relative)
    _rewrite(tuning / "manifest.json", manifest)


def _break(fault, root, preparation, config):
    """Apply one fault; return the message it must raise before any target is read."""
    entry = config["final_runs"]["nersi_real"][0]
    tuning = root / entry["tuning_run"]
    manifest = json.loads((tuning / "manifest.json").read_text())
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    selection = json.loads(_selection_path(root, "nersi_real").read_text())
    if fault == "tampered_prediction":
        prediction = tuning / entry["candidate"] / "prediction.npy"
        np.save(prediction, np.load(prediction) * 2, allow_pickle=False)
        return "input hash mismatch"
    if fault == "seed_mismatch":
        entry["seed"] = 7301
        return "seed mismatch"
    if fault == "missing_method":
        del config["final_runs"]["regsi_grid"]
        return "every method"
    if fault == "duplicate_seeds":
        config["final_runs"]["nersi_real"][1]["seed"] = entry["seed"]
        return "duplicate final seeds"
    if fault == "shadow_entry":
        entry["ema_decay"] = 0.9995
        return "need seed, tuning_run and candidate"
    if fault == "bootstrap":
        config["bootstrap"] = {**BOOTSTRAP, "seed": 1}
        return "bootstrap differs"
    if fault == "validation_seal":
        _rewrite(
            preparation / "preparation_manifest.json",
            {**seal, "selection_domain": "validation_holdout"},
        )
        return "final \\(mask-B test\\) preparation"
    if fault == "non_oracle_seal":
        _rewrite(
            preparation / "preparation_manifest.json",
            {**seal, "amplitude_scaling": "observed_trace_rms_idw"},
        )
        return "trace-oracle preparation"
    if fault == "other_preparation":
        _rewrite(tuning / "manifest.json", {**manifest, "preparation_hash": "0" * 64})
        return "another preparation"
    if fault == "evaluated_run":
        _rewrite(tuning / "manifest.json", {**manifest, "purpose": "selection", "evaluate": True})
        return "unevaluated final_evaluation"
    if fault == "scored_run":
        artifacts = {**manifest["artifacts"], f"{entry['candidate']}/metrics.json": "0" * 64}
        _rewrite(tuning / "manifest.json", {**manifest, "artifacts": artifacts})
        return "unevaluated final_evaluation"
    if fault == "unrecorded_yaml":
        selection["runs"][0]["sha256"] = "0" * 64
        _rewrite(_selection_path(root, "nersi_real"), selection)
        return "not a final run of the recorded selection"
    if fault == "other_ema":
        path = tuning / entry["candidate"] / "resolved_config.yaml"
        resolved = yaml.safe_load(path.read_text())
        resolved["training"]["ema_decay"] = 0.9995
        path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        _rebind(tuning, f"{entry['candidate']}/resolved_config.yaml")
        return "does not predict the selected EMA 0.999"
    if fault == "missing_run":
        config["final_runs"]["nersi_real"].pop()
        return "final runs differ from the recorded selection"
    if fault == "stale_selection":
        _rewrite(_selection_path(root, "nersi_real"), {**selection, "grid_sha256": "0" * 64})
        return "another grid or preparation"
    if fault == "no_selection":
        _selection_path(root, "drr_grid").unlink()
        return "no selection record"
    raise AssertionError(fault)


FAULTS = [
    "tampered_prediction",
    "seed_mismatch",
    "missing_method",
    "duplicate_seeds",
    "shadow_entry",
    "bootstrap",
    "validation_seal",
    "non_oracle_seal",
    "other_preparation",
    "evaluated_run",
    "scored_run",
    "unrecorded_yaml",
    "other_ema",
    "missing_run",
    "stale_selection",
    "no_selection",
]


@pytest.mark.parametrize("fault", FAULTS)
def test_invalid_final_runs_are_rejected_before_targets_are_read(evaluation, monkeypatch, fault):
    root, preparation, config, path = evaluation
    message = _break(fault, root, preparation, config)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    reads = _count_target_reads(monkeypatch)
    with pytest.raises(ValueError, match=message):
        evaluate_forge_fair_grid(root, path)
    assert reads == []
    assert not list((root / "runs/tiny_forge_mvp").glob("*_final_evaluation"))


@pytest.mark.parametrize(
    "fault,message",
    [
        ("other_parent_mask", "mask-B holdout"),
        ("edited_manifest", "manifest differs from its record"),
        ("not_diverged", "recorded as diverged"),
    ],
)
def test_search_records_must_be_the_bound_mask_b_validation_runs(
    evaluation, monkeypatch, fault, message
):
    root, preparation, config, path = evaluation
    selection_path = _selection_path(root, "ccnet5d_grid")
    selection = json.loads(selection_path.read_text())
    if fault == "other_parent_mask":
        validation = root / "runs/validation_prep/preparation_manifest.json"
        validation.write_text(
            json.dumps({"selection_domain": "validation_holdout", "parent_mask_hash": "other"})
        )
    elif fault == "edited_manifest":
        tuning = root / selection["rule_inputs"][0]["tuning_run"]
        manifest = json.loads((tuning / "manifest.json").read_text())
        (tuning / "manifest.json").write_text(json.dumps({**manifest, "edited": True}))
    else:
        selection["rule_inputs"][0]["status"] = "diverged"
        _rewrite(selection_path, selection)
    reads = _count_target_reads(monkeypatch)
    with pytest.raises(ValueError, match=message):
        evaluate_forge_fair_grid(root, path)
    assert reads == []
