"""Final evaluation of an added method on a derived preparation, against a reference evaluation.

A tiny parent study is prepared for real; its completed six-method evaluation is
synthesized (seed-averaged errors in the evaluator's row layout, restoration
scales as the evaluator computes them). A second study derives the final
preparation with ``nersi_profile`` and pre-registers the reference and its
contrasts in grid.yaml; only ``nersi_profile`` final runs are scored.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import read_selected_traces, verify_hashes, write_json
from seis_interp.evaluation.forge_fair_grid_statistics import (
    cluster_bootstrap_weights,
    paired_contrasts,
)
from seis_interp.pipelines import evaluate_forge_fair_grid as pipeline
from seis_interp.pipelines.derive_forge_preparation import derive_forge_preparation
from seis_interp.pipelines.evaluate_forge_fair_grid import evaluate_forge_fair_grid
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.processing.forge_mvp_contract import METHODS
from tests.fixtures.forge_mvp import tiny_forge_study

REPO = Path(__file__).resolve().parents[2]
CANONICAL = REPO / "studies/study_061_forge_nersi_profile"
STUDY = "tiny_forge_profile"
METHOD = "nersi_profile"
MASK = "masks/m1_random80_mvp_v1.parquet"
SEAL = "preparation_manifest.json"
REFERENCE = "runs/tiny_forge_mvp/20261008T000000000000Z_testcommit_final_evaluation"
FINAL_SEEDS = (7311, 7312, 7313)
BOOTSTRAP = {"resamples": 200, "seed": 600703}
KEYS = ("0.999", "0.9995", "0.9998")
SELECTED_EMA = "0.999"
ERROR_KEYS = ["restoration", "trace_id", "cell_id", "source_cell", "clean_target"]
# The pre-registered study_061 contrasts: nersi_profile against every reference method.
CONTRASTS = yaml.safe_load((CANONICAL / "grid.yaml").read_text())["evaluation"]["contrasts"]


def _test_frame(preparation):
    mask = pd.read_parquet(preparation / MASK)
    return mask.loc[mask.split.eq("test")].reset_index(drop=True)


def _targets(root, preparation):
    mask = pd.read_parquet(preparation / MASK)
    rows = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    return read_selected_traces(root / "raw", rows.loc[mask.split.eq("test")], 9, 1000)


def _write_reference(root, parent, *, errors=None, scales=None, **manifest_keys):
    """The parent study's completed six-method final evaluation; returns its manifest sha256."""
    output = root / REFERENCE
    aggregate = output / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    index = _test_frame(parent)
    if errors is None:
        rng = np.random.default_rng(5)
        tables = []
        for restoration in pipeline.RESTORATIONS:
            table = index[ERROR_KEYS[1:]].copy()
            table.insert(0, "restoration", restoration)
            for level, method in enumerate(METHODS):
                table[method] = rng.uniform(0.2, 1.0, len(index)) * (1 + 0.1 * level)
            tables.append(table)
        errors = pd.concat(tables, ignore_index=True)
    if scales is None:
        contract = json.loads((parent / "input/input_contract.json").read_text())
        mask = pd.read_parquet(parent / MASK)
        scales, _ = pipeline._restoration_scales(parent, contract, mask, index)
    errors.to_parquet(aggregate / "seed_averaged_errors.parquet", index=False)
    scales.to_parquet(aggregate / "non_oracle_trace_scales.parquet", index=False)
    seal = json.loads((parent / SEAL).read_text())
    manifest = {
        "status": "completed",
        "preparation": str(parent.relative_to(root)),
        "preparation_hash": sha256_file(parent / SEAL),
        "mask_hash": seal["mask_hash"],
        "bootstrap": BOOTSTRAP,
        # Listed out of method order: the recorded reference methods follow METHODS.
        "final_runs": [
            {"method_id": method, "seed": None, "run": f"final_{method}"}
            for method in reversed(METHODS)
        ],
        "artifacts": {
            str(path.relative_to(output)): sha256_file(path) for path in sorted(aggregate.glob("*"))
        },
        **manifest_keys,
    }
    write_json(output / "manifest.json", manifest)
    return sha256_file(output / "manifest.json")


def _profile_config():
    """study_061's centre configuration, shrunk to the tiny survey."""
    config = yaml.safe_load((CANONICAL / f"{METHOD}.yaml").read_text())
    config["device"] = "cpu"
    # 9 time samples and 3 receiver points padded to multiples of eight.
    config["model"].update(
        fourier_components=2,
        encoder_width=4,
        decoder_channels=[2, 2, 2],
        kernel_size=1,
        profile_shape=[16, 8],
    )
    config["training"].update(max_steps=1, report_interval=1, supervised_traces_per_update=4)
    return config


def _derive(root, parent):
    study = root / "studies" / STUDY
    study.mkdir()
    contract = json.loads((parent / "input/input_contract.json").read_text())
    (study / "config.yaml").write_text(yaml.safe_dump({**contract, "study_id": STUDY}))
    (study / f"{METHOD}.yaml").write_text(yaml.safe_dump(_profile_config(), sort_keys=False))
    return derive_forge_preparation(root, parent, study, [METHOD])


def _tuning_run(root, name, preparation, index, files, **manifest_keys):
    """A tuning-run directory with the tuner's manifest, index and per-candidate files."""
    run = root / "runs/tuning" / name
    run.mkdir(parents=True)
    index.assign(prediction_row=np.arange(len(index))).to_parquet(
        run / "prediction_index.parquet", index=False
    )
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
        "method_id": METHOD,
        "preparation": str(preparation.relative_to(root)),
        "preparation_hash": sha256_file(preparation / SEAL),
        "artifacts": {
            str(p.relative_to(run)): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file()
        },
        **manifest_keys,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return str(run.relative_to(root))


def _final_runs(root, derived):
    """Three unevaluated nersi_profile final runs and the selection run records they match."""
    target = _targets(root, derived)
    index = _test_frame(derived)
    config = yaml.safe_load((derived / "configs" / f"{METHOD}.yaml").read_text())
    rng = np.random.default_rng(9)
    runs, planned = [], []
    for seed in FINAL_SEEDS:
        name = f"final_{METHOD}_{seed}"
        training = {**config["training"], "ema_decay": float(SELECTED_EMA)}
        noise = rng.normal(0, 0.1, target.shape) * target.std()
        files = {
            "search.yaml": {
                "method_id": METHOD,
                "purpose": "final_evaluation",
                "evaluate": False,
                "device": "cpu",
                "candidates": [{"name": name}],
            },
            f"{name}/prediction.npy": (target + noise).astype(np.float32),
            f"{name}/resolved_config.yaml": {**config, "model_seed": seed, "training": training},
        }
        tuning = _tuning_run(
            root, name, derived, index, files, purpose="final_evaluation", evaluate=False
        )
        runs.append({"seed": seed, "tuning_run": tuning, "candidate": name})
        search = sha256_file(root / tuning / "search.yaml")
        planned.append({"name": name, "method_id": METHOD, "sha256": search})
    return runs, planned


def _search_inputs(root, derived):
    """One completed and one diverged S1 validation run of the added method."""
    seal = json.loads((derived / SEAL).read_text())
    validation = root / "runs/validation_prep"
    validation.mkdir(parents=True)
    (validation / SEAL).write_text(
        json.dumps(
            {"selection_domain": "validation_holdout", "parent_mask_hash": seal["mask_hash"]}
        )
    )
    inputs = []
    for number, status in enumerate(("completed", "diverged")):
        name = f"s1_{METHOD}_{number}"
        files, manifest = {}, {"selection_domain": "validation_holdout"}
        if status == "completed":
            files["search_results.csv"] = pd.DataFrame(
                {"name": name, "ema_decay": KEYS, "primary_relative_mse": [0.5, 0.4, 0.45]}
            )
        else:
            manifest.update(status="failed", failure_reason="RuntimeError: nonfinite NeRSI loss")
        tuning = _tuning_run(root, name, validation, _test_frame(derived), files, **manifest)
        inputs.append(
            {
                "run": name,
                "stage": "S1",
                "method_id": METHOD,
                "tuning_run": tuning,
                "tuning_manifest_sha256": sha256_file(root / tuning / "manifest.json"),
                "status": status,
            }
        )
    return inputs


def _selection_path(root):
    return root / "runs" / STUDY / "fair_grid" / METHOD / "final/stage.json"


def _grid_path(root):
    return root / "studies" / STUDY / "grid.yaml"


def _register(root, grid, selection=None):
    """Write grid.yaml and bind the nersi_profile selection record to it."""
    _grid_path(root).write_text(yaml.safe_dump(grid, sort_keys=False))
    if selection is None:
        selection = json.loads(_selection_path(root).read_text())
    write_json(_selection_path(root), {**selection, "grid_sha256": sha256_file(_grid_path(root))})


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    root = tmp_path
    study, contract, *_ = tiny_forge_study(root)
    contract["preprocessing"].update(amplitude_scaling="trace_oracle", trace_rms=True)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    identity = {"git_commit": "testcommit", "dirty_worktree": True}
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.derive_forge_preparation.execution_identity", lambda _: identity
    )
    monkeypatch.setattr(pipeline, "execution_identity", lambda _: identity)
    parent = prepare_forge_mvp(root, study)
    reference_sha256 = _write_reference(root, parent)
    derived = _derive(root, parent)
    runs, planned = _final_runs(root, derived)
    grid = {
        "study_id": STUDY,
        "selection": {
            "diverged_runs": {"failure_reason_pattern": "non-?finite", "reported_score": 1.0}
        },
        "ema_eval_decays": [float(key) for key in KEYS],
        "evaluation": {
            "bootstrap": BOOTSTRAP,
            "reference_evaluation": {"path": REFERENCE, "manifest_sha256": reference_sha256},
            "contrasts": CONTRASTS,
        },
        "methods": {METHOD: {"method_id": METHOD}},
    }
    final = {
        "path": str(derived.relative_to(root)),
        "manifest_sha256": sha256_file(derived / SEAL),
    }
    selection = {
        "study_id": STUDY,
        "method": METHOD,
        "stage": "final",
        "preparations": {"validation": {"path": "runs/validation_prep"}, "final": final},
        "rule_inputs": _search_inputs(root, derived),
        "decision": {"ema_decay": SELECTED_EMA},
        "runs": planned,
    }
    _register(root, grid, selection)
    config = {"preparation": final["path"], "bootstrap": BOOTSTRAP, "final_runs": {METHOD: runs}}
    path = root / "final_evaluation.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return root, parent, derived, path


def _count_target_reads(monkeypatch):
    calls = []

    def counted(*args, **kwargs):
        values = read_selected_traces(*args, **kwargs)
        calls.append(len(values))
        return values

    monkeypatch.setattr(pipeline, "read_selected_traces", counted)
    return calls


def test_scenario_derives_the_final_preparation_from_the_reference_parent(scenario):
    root, parent, derived, _ = scenario
    seal = json.loads((derived / SEAL).read_text())
    assert seal["derived_from"]["manifest_sha256"] == sha256_file(parent / SEAL)
    assert seal["derived_from"]["added_methods"] == [METHOD]
    assert seal["amplitude_scaling"] == "trace_oracle"
    assert sha256_file(derived / MASK) == sha256_file(parent / MASK)
    resolved = yaml.safe_load((derived / "configs" / f"{METHOD}.yaml").read_text())
    assert resolved["model"]["profile_shape"] == [16, 8]


def test_added_method_is_contrasted_with_the_reference_methods(scenario, monkeypatch):
    root, parent, derived, path = scenario
    reads = _count_target_reads(monkeypatch)

    output = evaluate_forge_fair_grid(root, path)

    index = _test_frame(derived)
    assert reads == [len(index)]
    assert output.parent == root / "runs" / STUDY
    assert output.name.endswith("_testcommit_final_evaluation")
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["test_waveform_reads"] == 1
    assert record["test_waveforms_read"] == len(index)
    assert list(record["selection_records"]) == [METHOD]
    assert record["reference_evaluation"] == {
        "path": REFERENCE,
        "manifest_sha256": sha256_file(root / REFERENCE / "manifest.json"),
        "methods": list(METHODS),
    }
    assert record["contrasts"] == CONTRASTS
    assert [run["method_id"] for run in record["final_runs"]] == [METHOD] * 3
    assert {run["ema_decay"] for run in record["final_runs"]} == {SELECTED_EMA}
    aggregate = output / "aggregate"
    reference = root / REFERENCE / "aggregate"
    # Own table: the added method only, on the reference's exact rows.
    averaged = pd.read_parquet(aggregate / "seed_averaged_errors.parquet")
    reference_errors = pd.read_parquet(reference / "seed_averaged_errors.parquet")
    assert list(averaged.columns) == [*ERROR_KEYS, METHOD]
    pd.testing.assert_frame_equal(averaged[ERROR_KEYS], reference_errors[ERROR_KEYS])
    oracle = averaged.loc[averaged.restoration.eq("trace_oracle"), METHOD].to_numpy()
    per_seed = [
        pd.read_parquet(
            output / f"runs/{METHOD}/seed_{seed}/trace_oracle_metrics_by_trace.parquet"
        ).relative_mse
        for seed in FINAL_SEEDS
    ]
    np.testing.assert_allclose(oracle, np.mean(per_seed, axis=0))
    pd.testing.assert_frame_equal(
        pd.read_parquet(aggregate / "non_oracle_trace_scales.parquet"),
        pd.read_parquet(reference / "non_oracle_trace_scales.parquet"),
    )
    # Every family holds exactly the declared contrasts, computed from the reference
    # columns unchanged and this evaluation's nersi_profile errors.
    contrasts = pd.read_csv(aggregate / "contrasts.csv")
    assert len(contrasts) == len(pipeline.CONTRAST_FAMILIES) * len(CONTRASTS)
    combined = pd.concat([averaged, reference_errors[list(METHODS)]], axis=1)
    for family, restoration, domain in pipeline.CONTRAST_FAMILIES:
        actual = contrasts.loc[contrasts.family.eq(family)].reset_index(drop=True)
        assert actual.contrast.tolist() == [contrast[0] for contrast in CONTRASTS]
        assert set(actual.restoration) == {restoration} and set(actual.domain) == {domain}
        rows = combined.loc[combined.restoration.eq(restoration)]
        if domain == "clean_target":
            rows = rows.loc[rows.clean_target]
        clusters = rows.source_cell.to_numpy()
        weights = cluster_bootstrap_weights(
            len(np.unique(clusters)), BOOTSTRAP["resamples"], BOOTSTRAP["seed"]
        )
        expected = paired_contrasts(
            {method: rows[method].to_numpy() for method in (METHOD, *METHODS)},
            clusters,
            weights,
            [tuple(contrast) for contrast in CONTRASTS],
        )
        pd.testing.assert_frame_equal(
            actual.drop(columns=["family", "restoration", "domain"]),
            pd.read_csv(io.StringIO(expected.to_csv(index=False))),
        )
    # The added method's diverged search run enters at the reported score for every shadow.
    search = pd.read_csv(aggregate / "search_scores.csv")
    diverged = search.loc[search.diverged]
    assert diverged.name.unique().tolist() == [f"s1_{METHOD}_1"]
    assert diverged.primary_relative_mse.tolist() == [1.0] * len(KEYS)
    curves = pd.read_csv(aggregate / "expected_best_curves.csv")
    assert curves.groupby("method_id").size().to_dict() == {METHOD: 2 * len(KEYS)}


def _break(fault, root, parent, derived):
    """Apply one fault to the registered reference; return the message it must raise."""
    grid = yaml.safe_load(_grid_path(root).read_text())
    declared = grid["evaluation"]["reference_evaluation"]
    if fault == "manifest_sha256":
        declared["manifest_sha256"] = "0" * 64
        message = "reference evaluation manifest differs from grid.yaml"
    elif fault == "other_parent":
        # An evaluation on the derived preparation itself, not on its parent.
        declared["manifest_sha256"] = _write_reference(
            root, parent, preparation_hash=sha256_file(derived / SEAL)
        )
        message = "not the completed evaluation of the parent"
    elif fault == "reordered_rows":
        errors = pd.read_parquet(root / REFERENCE / "aggregate/seed_averaged_errors.parquet")
        order = [1, 0, *range(2, len(errors))]
        declared["manifest_sha256"] = _write_reference(
            root, parent, errors=errors.iloc[order].reset_index(drop=True)
        )
        message = "reference errors cover other rows or methods"
    elif fault == "unknown_contrast_method":
        grid["evaluation"]["contrasts"].append(
            ["nersi_profile_vs_siren", METHOD, "siren_real", "unknown"]
        )
        message = "a contrast names a method that is not evaluated or referenced"
    else:
        raise AssertionError(fault)
    _register(root, grid)
    return message


@pytest.mark.parametrize(
    "fault", ["manifest_sha256", "other_parent", "reordered_rows", "unknown_contrast_method"]
)
def test_invalid_reference_is_rejected_before_targets_are_read(scenario, monkeypatch, fault):
    root, parent, derived, path = scenario
    message = _break(fault, root, parent, derived)
    reads = _count_target_reads(monkeypatch)
    with pytest.raises(ValueError, match=message):
        evaluate_forge_fair_grid(root, path)
    assert reads == []
    assert not list((root / "runs" / STUDY).glob("*_final_evaluation"))


def test_reference_with_other_scales_is_rejected_before_target_reads(scenario, monkeypatch):
    root, parent, derived, path = scenario
    grid = yaml.safe_load(_grid_path(root).read_text())
    scales = pd.read_parquet(root / REFERENCE / "aggregate/non_oracle_trace_scales.parquet")
    scales.loc[0, "idw_rms"] *= 1.01
    grid["evaluation"]["reference_evaluation"]["manifest_sha256"] = _write_reference(
        root, parent, scales=scales
    )
    _register(root, grid)
    reads = _count_target_reads(monkeypatch)
    message = "non-oracle restoration scales differ from the reference evaluation"
    with pytest.raises(ValueError, match=message):
        evaluate_forge_fair_grid(root, path)
    # The scales come from observed traces only and are compared before any target read.
    assert reads == []
    (output,) = (root / "runs" / STUDY).glob("*_final_evaluation")
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "failed" and record["test_waveform_reads"] == 0
    assert message in record["failure_reason"]
    assert not (output / "aggregate").exists()
    assert not (output / "runs").exists()
