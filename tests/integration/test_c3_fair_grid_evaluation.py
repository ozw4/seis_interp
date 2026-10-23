"""The FORGE fair-grid final evaluation on an observed-global-RMS C3 preparation.

A tiny synthetic C3 study is prepared (final and validation); POCS is retrained
once on the final preparation by the real tuner and the other four methods get
recorded final runs with synthetic physical-amplitude predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.c3_fair_grid_truth import read_c3_scored_targets
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import read_search_results
from seis_interp.evaluation.trace_snr import trace_snr_db
from seis_interp.pipelines import evaluate_forge_fair_grid as pipeline
from seis_interp.pipelines import prepare_c3_fair_grid as preparation_pipeline
from seis_interp.pipelines import tune_forge_matched as tuner
from seis_interp.pipelines.evaluate_forge_fair_grid import evaluate_forge_fair_grid
from seis_interp.pipelines.prepare_c3_fair_grid import prepare_c3_fair_grid
from seis_interp.pipelines.tune_forge_matched import tune_forge_matched
from tests.fixtures.c3_fair_grid import DECAYS, STUDY_ID, tiny_c3_study

MASK = "masks/m1_random80_mvp_v1.parquet"
NEURAL = ("ccnet5d_grid", "regsi_real", "nersi_profile")
METHODS = ("pocs_grid", "drr_grid", *NEURAL)
FINAL_SEEDS = (7311, 7312, 7313)
SELECTED_EMA = "0.8"
BOOTSTRAP = {"resamples": 200, "seed": 620703}
CONTRASTS = [
    ["regsi_real_vs_ccnet5d_grid", "regsi_real", "ccnet5d_grid", "headline"],
    ["regsi_real_vs_nersi_profile", "regsi_real", "nersi_profile", "headline"],
    ["regsi_real_vs_pocs_grid", "regsi_real", "pocs_grid", "classical_baseline"],
    ["regsi_real_vs_drr_grid", "regsi_real", "drr_grid", "classical_baseline"],
]
FAMILIES = [
    "primary",
    "secondary_all_eligible",
    "secondary_scale_invariant",
    "secondary_trace_snr_db",
]


def _grid_key(method):
    return {"regsi_real": "regsi"}.get(method, method)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preparation_pipeline.subprocess,
        "check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    identity = {"git_commit": "testcommit", "dirty_worktree": True}
    monkeypatch.setattr(tuner, "execution_identity", lambda _: identity)
    monkeypatch.setattr(pipeline, "execution_identity", lambda _: identity)
    study, _, _ = tiny_c3_study(tmp_path)
    runs = {
        "final": prepare_c3_fair_grid(tmp_path, study),
        "validation": prepare_c3_fair_grid(tmp_path, study, validation_holdout=True),
    }
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield tmp_path, runs
    torch.set_num_threads(previous)


def _test_frame(preparation):
    mask = pd.read_parquet(preparation / MASK)
    return mask.loc[mask.split.eq("test")].reset_index(drop=True)


def _targets(root, preparation):
    return read_c3_scored_targets(root, preparation, pd.read_parquet(preparation / MASK))


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


def _final_runs(root, preparation):
    """POCS by the real tuner; recorded synthetic final runs for the other methods."""
    search = root / "pocs_final.yaml"
    search.write_text(
        yaml.safe_dump(
            {
                "method_id": "pocs_grid",
                "purpose": "final_evaluation",
                "evaluate": False,
                "device": "cpu",
                "backend": "batched",
                "candidates": [{"name": "final_pocs"}],
            }
        )
    )
    pocs = str(tune_forge_matched(root, preparation, search).relative_to(root))
    runs = {"pocs_grid": [{"seed": None, "tuning_run": pocs, "candidate": "final_pocs"}]}
    planned = {
        "pocs_grid": [
            {"name": "final_pocs", "method_id": "pocs_grid", "sha256": sha256_file(search)}
        ]
    }
    target = _targets(root, preparation)
    rng = np.random.default_rng(9)
    for level, method in enumerate(("drr_grid", *NEURAL)):
        seeds = (None,) if method == "drr_grid" else FINAL_SEEDS
        runs[method], planned[method] = [], []
        config = yaml.safe_load((preparation / "configs" / f"{method}.yaml").read_text())
        for seed in seeds:
            name = f"final_{method}_{seed}"
            resolved = {**config, "model_seed": seed}
            if method in NEURAL:
                resolved["training"] = {**config["training"], "ema_decay": float(SELECTED_EMA)}
            noise = rng.normal(0, 0.1 * (level + 1), target.shape) * target.std()
            files = {
                "search.yaml": {
                    "method_id": method,
                    "purpose": "final_evaluation",
                    "evaluate": False,
                    "device": "cpu",
                    "candidates": [{"name": name}],
                },
                f"{name}/prediction.npy": (target + noise).astype(np.float32),
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


def _search_inputs(root, runs):
    """One recorded S1 validation run per method on the real validation preparation."""
    keys = [str(decay) for decay in DECAYS]
    inputs = {}
    for number, method in enumerate(METHODS):
        name = f"s1_{method}"
        shadows = [None] if method in ("pocs_grid", "drr_grid") else keys
        scores = 0.9 - 0.05 * number - 0.01 * np.arange(len(shadows))
        files = {
            "search_results.csv": pd.DataFrame(
                {
                    "name": name,
                    "ema_decay": shadows,
                    "primary_relative_mse": scores,
                    # A tuner hash that the pandas C parser would read as a huge float.
                    "initialization_hash": "8228172598e729840098733f451de5541f809dd7",
                }
            )
        }
        tuning = _write_tuning_run(
            root,
            name,
            method,
            runs["validation"],
            _test_frame(runs["validation"]),
            files,
            selection_domain="validation_holdout",
        )
        inputs[method] = [
            {
                "run": name,
                "stage": "S1",
                "method_id": method,
                "tuning_run": tuning,
                "tuning_manifest_sha256": sha256_file(root / tuning / "manifest.json"),
                "status": "completed",
            }
        ]
    return inputs


def _write_registration(root, runs, planned, inputs, **evaluation):
    grid = {
        "study_id": STUDY_ID,
        "selection": {
            "diverged_runs": {"failure_reason_pattern": "non-?finite", "reported_score": 1.0}
        },
        "ema_eval_decays": DECAYS,
        "evaluation": {"bootstrap": BOOTSTRAP, "contrasts": CONTRASTS, **evaluation},
        "methods": {_grid_key(method): {"method_id": method} for method in METHODS},
    }
    grid_path = root / "studies" / STUDY_ID / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid, sort_keys=False))
    final = {
        "path": str(runs["final"].relative_to(root)),
        "manifest_sha256": sha256_file(runs["final"] / "preparation_manifest.json"),
    }
    for method in METHODS:
        write_json(
            root / "runs" / STUDY_ID / "fair_grid" / _grid_key(method) / "final/stage.json",
            {
                "study_id": STUDY_ID,
                "method": _grid_key(method),
                "stage": "final",
                "grid_sha256": sha256_file(grid_path),
                "preparations": {"final": final},
                "rule_inputs": inputs[method],
                "decision": {}
                if method in ("pocs_grid", "drr_grid")
                else {"ema_decay": SELECTED_EMA},
                "runs": planned[method],
            },
        )


@pytest.fixture
def evaluation(prepared):
    root, runs = prepared
    final_runs, planned = _final_runs(root, runs["final"])
    _write_registration(root, runs, planned, _search_inputs(root, runs))
    path = root / "final_evaluation.yaml"
    config = {
        "preparation": str(runs["final"].relative_to(root)),
        "bootstrap": BOOTSTRAP,
        "final_runs": final_runs,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return root, runs, final_runs, path


def _count_target_reads(monkeypatch):
    calls = []

    def counted(repo, preparation, mask):
        values = read_c3_scored_targets(repo, preparation, mask)
        calls.append(len(values))
        return values

    monkeypatch.setattr(pipeline, "read_c3_scored_targets", counted)
    monkeypatch.setattr(pipeline, "read_selected_traces", lambda *a, **k: pytest.fail("SEG-Y"))
    return calls


def test_global_rms_evaluation_scores_physical_predictions_and_trace_snr(evaluation, monkeypatch):
    root, runs, final_runs, path = evaluation
    reads = _count_target_reads(monkeypatch)
    output = evaluate_forge_fair_grid(root, path)
    index = _test_frame(runs["final"])
    target = _targets(root, runs["final"])
    assert reads == [len(index)] == [205]
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["amplitude_scaling"] == "observed_global_rms"
    assert record["restorations"] == ["observed_global_rms", "scale_invariant"]
    assert "non_oracle_restoration" not in record
    assert record["test_waveform_reads"] == 1 and record["test_waveforms_read"] == 205
    assert record["contrasts"] == CONTRASTS
    assert record["trace_snr_contrasts"] == "computed"
    aggregate = output / "aggregate"
    names = {path.name for path in aggregate.iterdir()}
    assert names == {
        "seed_metrics.csv",
        "method_summary.csv",
        "seed_averaged_errors.parquet",
        "seed_averaged_trace_snr_db.parquet",
        "contrasts.csv",
        "search_scores.csv",
        "expected_best_curves.csv",
    }
    files = {p.name for p in (output / "runs/regsi_real/seed_7311").iterdir()}
    assert files == {
        "observed_global_rms_metrics.json",
        "observed_global_rms_metrics_by_trace.parquet",
        "scale_invariant_by_trace.parquet",
        "trace_snr_db_by_trace.parquet",
    }
    # Per-seed relative MSE and SNR are computed on the physical prediction as stored.
    seed_metrics = pd.read_csv(aggregate / "seed_metrics.csv")
    assert len(seed_metrics) == (2 + 3 * len(NEURAL)) * 2 * 2
    assert set(seed_metrics.restoration) == {"observed_global_rms", "scale_invariant"}
    physical = seed_metrics.loc[
        seed_metrics.restoration.eq("observed_global_rms") & seed_metrics.domain.eq("clean_target")
    ].assign(seed=lambda rows: rows.seed.fillna(-1).astype(int))
    physical = physical.set_index(["method_id", "seed"])
    snr = {}
    for method, entries in final_runs.items():
        for entry in entries:
            tuning = root / entry["tuning_run"] / entry["candidate"]
            prediction = np.load(tuning / "prediction.npy").astype(np.float64)
            relative = ((prediction - target) ** 2).sum(1) / (target.astype(np.float64) ** 2).sum(1)
            values = trace_snr_db(target, prediction)
            snr.setdefault(method, []).append(values)
            row = physical.loc[(method, -1 if entry["seed"] is None else entry["seed"])]
            assert row.mean_relative_mse == pytest.approx(relative.mean(), rel=1e-10)
            assert row.mean_trace_snr_db == pytest.approx(values.mean(), rel=1e-10)
            global_snr = 10 * np.log10(
                (target.astype(np.float64) ** 2).sum() / ((prediction - target) ** 2).sum()
            )
            assert row.global_snr_db == pytest.approx(global_snr, rel=1e-10)
            assert row.trace_snr_nonfinite_count == 0
    summary = pd.read_csv(aggregate / "method_summary.csv").set_index(
        ["method_id", "restoration", "domain"]
    )
    row = summary.loc[("regsi_real", "observed_global_rms", "clean_target")]
    means = [values.mean() for values in snr["regsi_real"]]
    assert row.mean_trace_snr_db_seed_mean == pytest.approx(np.mean(means))
    assert row.mean_trace_snr_db_seed_sd == pytest.approx(np.std(means, ddof=1))
    global_means = physical.loc["regsi_real", "global_snr_db"].to_numpy()
    assert row.global_snr_db_seed_mean == pytest.approx(global_means.mean())
    assert row.global_snr_db_seed_sd == pytest.approx(np.std(global_means, ddof=1))
    assert row.trace_snr_nonfinite_count == 0
    assert np.isnan(
        summary.loc[
            ("regsi_real", "scale_invariant", "clean_target"), "mean_trace_snr_db_seed_mean"
        ]
    )
    averaged = pd.read_parquet(aggregate / "seed_averaged_trace_snr_db.parquet")
    assert averaged.cell_id.tolist() == index.cell_id.tolist()
    for method in METHODS:
        np.testing.assert_allclose(averaged[method], np.mean(snr[method], axis=0))
    errors = pd.read_parquet(aggregate / "seed_averaged_errors.parquet")
    assert errors.restoration.unique().tolist() == ["observed_global_rms", "scale_invariant"]
    # Four Holm families; the SNR family's difference is left - right dB (positive: left).
    contrasts = pd.read_csv(aggregate / "contrasts.csv")
    assert contrasts.family.unique().tolist() == FAMILIES
    assert contrasts.groupby("family").size().to_dict() == dict.fromkeys(FAMILIES, len(CONTRASTS))
    orientation = contrasts.set_index("family").left_better_when.groupby(level=0).unique()
    assert orientation.to_dict() == {
        "primary": ["difference_negative"],
        "secondary_all_eligible": ["difference_negative"],
        "secondary_scale_invariant": ["difference_negative"],
        "secondary_trace_snr_db": ["difference_positive"],
    }
    assert set(contrasts.restoration) == {"observed_global_rms", "scale_invariant"}
    assert (contrasts.cluster_count == index.source_cell.nunique()).all()
    primary = contrasts.loc[contrasts.family.eq("primary")].reset_index(drop=True)
    snr_rows = contrasts.loc[contrasts.family.eq("secondary_trace_snr_db")].reset_index(drop=True)
    physical_errors = errors.loc[errors.restoration.eq("observed_global_rms")]
    for position, (_, left, right, _) in enumerate(CONTRASTS):
        assert primary.estimate[position] == pytest.approx(
            (physical_errors[left] - physical_errors[right]).mean()
        )
        difference = averaged[left] - averaged[right]
        assert snr_rows.estimate[position] == pytest.approx(difference.mean())
        assert snr_rows.left_mean_trace_snr_db[position] == pytest.approx(averaged[left].mean())
        assert snr_rows.fraction_left_higher_snr[position] == pytest.approx((difference > 0).mean())
    assert snr_rows.left_mean_error.isna().all() and snr_rows.fraction_left_lower_error.isna().all()
    # On these predictions a lower relative MSE always comes with a higher SNR.
    np.testing.assert_array_equal(np.sign(primary.estimate), -np.sign(snr_rows.estimate))
    assert np.all(contrasts.holm_p_value >= contrasts.bootstrap_p_value)
    search = read_search_results(aggregate / "search_scores.csv")
    assert set(search.method_id) == set(METHODS) and set(search.stage) == {"S1"}


def test_a_non_finite_trace_snr_skips_only_the_snr_family(evaluation, monkeypatch):
    # A perfect trace has an infinite SNR: it is counted, and the one test read still
    # yields every relative-MSE family.
    root, _, _, path = evaluation
    original = pipeline.trace_snr_by_trace

    def with_perfect_trace(target, prediction):
        values = original(target, prediction)
        values[0] = np.inf
        return values

    monkeypatch.setattr(pipeline, "trace_snr_by_trace", with_perfect_trace)
    reads = _count_target_reads(monkeypatch)
    output = evaluate_forge_fair_grid(root, path)
    record = json.loads((output / "manifest.json").read_text())
    assert reads == [205] and record["status"] == "completed"
    runs = 2 + 3 * len(NEURAL)
    assert record["trace_snr_contrasts"] == f"skipped: {runs} non-finite per-trace SNR values"
    aggregate = output / "aggregate"
    assert not (aggregate / "seed_averaged_trace_snr_db.parquet").exists()
    contrasts = pd.read_csv(aggregate / "contrasts.csv")
    assert contrasts.family.unique().tolist() == FAMILIES[:3]
    seed_metrics = pd.read_csv(aggregate / "seed_metrics.csv")
    counts = seed_metrics.loc[
        seed_metrics.restoration.eq("observed_global_rms"), "trace_snr_nonfinite_count"
    ]
    assert (counts == 1).all()


@pytest.mark.parametrize(
    "evaluation_keys,message",
    [
        ({"contrasts": None}, "declares contrasts"),
        (
            {"reference_evaluation": {"path": "x", "manifest_sha256": "0" * 64}},
            "declares contrasts",
        ),
    ],
)
def test_global_rms_grids_declare_contrasts_and_no_reference(
    evaluation, monkeypatch, evaluation_keys, message
):
    root, _, _, path = evaluation
    grid_path = root / "studies" / STUDY_ID / "grid.yaml"
    grid = yaml.safe_load(grid_path.read_text())
    for key, value in evaluation_keys.items():
        if value is None:
            del grid["evaluation"][key]
        else:
            grid["evaluation"][key] = value
    grid_path.write_text(yaml.safe_dump(grid, sort_keys=False))
    reads = _count_target_reads(monkeypatch)
    with pytest.raises(ValueError, match=message):
        evaluate_forge_fair_grid(root, path)
    assert reads == []
    assert not list((root / "runs" / STUDY_ID).glob("*_final_evaluation"))


def test_only_c3_preparations_evaluate_without_trace_oracle(evaluation, monkeypatch):
    root, runs, _, path = evaluation
    seal_path = runs["final"] / "preparation_manifest.json"
    seal = json.loads(seal_path.read_text())
    seal_path.chmod(0o644)
    seal_path.write_text(json.dumps({**seal, "dataset_id": "forge_2017"}))
    reads = _count_target_reads(monkeypatch)
    with pytest.raises(ValueError, match="trace-oracle preparation"):
        evaluate_forge_fair_grid(root, path)
    assert reads == []
