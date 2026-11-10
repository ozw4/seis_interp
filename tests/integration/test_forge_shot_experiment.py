import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data import forge_trace_experiment as data
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.evaluation import forge_shot_results as scoring
from seis_interp.pipelines import run_forge_trace_experiment as pipeline
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.processing.forge_shot_mask import (
    make_periodic_shot_mask,
    make_random_shot_mask,
    random_shot_support,
    shot_mask_counts,
)
from seis_interp.training.forge_c3_methods import run_forge_c3_method
from seis_interp.visualization.forge_shot_comparison import export_forge_shot_results
from tests.fixtures.forge_mvp import tiny_forge_study

METHODS = ["regsi_real", "ccnet5d_grid", "nersi_profile", "pocs_grid", "drr_grid"]


@pytest.fixture(params=["periodic_shot", "random_shot"])
def shot_study(tmp_path, monkeypatch, request):
    study, contract, _, configs, *_ = tiny_forge_study(tmp_path, source_points=4)
    contract.update(
        methods=METHODS,
        mask_family="periodic_shot",
        interpolation_mask={"id": "tiny_shot50_phase0", "period": 2, "observed_phase": 0},
        validation_holdout=None,
        matched_training={"loss_traces_per_update": 2},
        fixed_settings={method: {"candidate_id": "fixture"} for method in METHODS},
    )
    contract["preprocessing"].update(
        amplitude_scaling="trace_oracle", alignment={"velocity_m_per_s": 10000.0}
    )
    if request.param == "random_shot":
        contract["mask_family"] = "random_shot"
        contract["interpolation_mask"] = {
            "id": "tiny_random_shot50",
            "mask_seed": 7302,
            "observed_shots_per_line": 2,
            "algorithm": "sha256_rank_per_source_line",
        }
    profile = deepcopy(configs["nersi_real"])
    profile.update(method_id="nersi_profile", canvas_time_upsampling=True)
    profile["model"] = dict(
        input_features=3,
        fourier_components=2,
        frequency_base=1.075,
        encoder_width=4,
        latent_channels=2,
        decoder_channels=[2, 2, 2],
        profile_shape=[16, 8],
        upsample_scales=[2, 2, 2],
        kernel_size=1,
        activation="gelu",
        output_activation="linear",
    )
    profile["training"].update(
        supervised_traces_per_update=2, nuclear_norm_weight=0.03, coordinate_jitter_cells=0.5
    )
    profile["prediction"] = {"profile_batch_size": 2}
    configs["nersi_profile"] = profile
    configs["regsi_real"]["training"]["inner_mask_fraction"] = 0.25
    configs["drr_grid"]["execution"] = {"backend": "batched", "workers": 1}
    for method in METHODS:
        cfg = configs[method]
        cfg["checkpoint_rule"] = "fixed_final_ema" if "training" in cfg else "none"
        if "training" in cfg:
            cfg["training"].update(optimizer="adamw", ema_decay=0.999)
        (study / f"{method}.yaml").write_text(yaml.safe_dump(cfg))
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    return tmp_path, study, contract


@pytest.fixture
def shot_prepared(shot_study):
    repo, study, contract = shot_study
    return repo, data.prepare_forge_trace_experiment(repo, study), contract


def test_periodic_mask_uses_original_indices_is_atomic_and_complementary(shot_prepared):
    _, experiment, contract = shot_prepared
    if contract["mask_family"] != "periodic_shot":
        pytest.skip("periodic mask contract")
    manifest = pd.read_parquet(experiment / "input/manifest.parquet")
    phase0 = pd.read_parquet(experiment / "input/partition.parquet")
    phase1_config = deepcopy(contract)
    phase1_config["interpolation_mask"]["observed_phase"] = 1
    phase1 = make_periodic_shot_mask(manifest.iloc[::-1], phase1_config)
    assert phase0.source_grid_point.mod(2).eq(0).equals(phase0.role.eq("train"))
    assert phase0.split.ne(phase1.split).all()
    assert phase0.groupby("source_cell").role.nunique().eq(1).all()
    assert set(phase0.cell_id) == set(manifest.cell_id) == set(range(47))
    assert not phase0.cell_id.eq(47).any()
    counts = shot_mask_counts(phase0, contract["spatial_shape"])
    assert counts["observed_shot_count"] == counts["test_shot_count"] == 4
    assert counts["observed_trace_count"] == 24 and counts["test_trace_count"] == 23
    assert counts["natural_missing_cell_count"] == 1
    assert counts["eligible_trace_missing_fraction"] == 23 / 47
    for line in counts["by_line"]:
        assert line["observed_shot_count"] == line["test_shot_count"] == 2
    bindings = []
    for method in METHODS:
        inputs, _, partition = data.load_forge_trace_inputs(experiment, method)
        np.testing.assert_array_equal(inputs.test_cells, phase0.cell_id[phase0.role.eq("test")])
        np.testing.assert_array_equal(
            inputs.observed_cells, phase0.cell_id[phase0.role.eq("train")]
        )
        assert partition.role.ne("validation").all()
        assert not inputs.excluded_cells.size
        with pytest.raises(ValueError, match="outside outer observed"):
            inputs.read_observed(inputs.test_cells)
        volume = inputs.volume()
        assert not volume.observed_trace_mask.reshape(-1)[inputs.test_cells].any()
        assert not volume.values.reshape(len(inputs.time_s), -1)[:, inputs.test_cells].any()
        bindings.append(inputs.binding)
    assert all(binding == bindings[0] for binding in bindings)


def test_phase_override_creates_a_new_frozen_experiment(shot_study):
    repo, study, contract = shot_study
    if contract["mask_family"] == "random_shot":
        with pytest.raises(ValueError, match="phase override requires periodic"):
            data.prepare_forge_trace_experiment(repo, study, observed_phase=1)
        return
    original = (study / "config.yaml").read_bytes()
    first = data.prepare_forge_trace_experiment(repo, study)
    second = data.prepare_forge_trace_experiment(repo, study, observed_phase=1)
    assert first != second and (study / "config.yaml").read_bytes() == original
    a = pd.read_parquet(first / "input/partition.parquet")
    b = pd.read_parquet(second / "input/partition.parquet")
    assert a.role.ne(b.role).all()
    assert (
        yaml.safe_load((second / "study/config.yaml").read_text())["interpolation_mask"][
            "observed_phase"
        ]
        == 1
    )


def test_physical_reuse_rebuilds_observations_without_reading_raw_or_old_observed(
    shot_study, monkeypatch
):
    repo, study, contract = shot_study
    if contract["mask_family"] != "periodic_shot":
        pytest.skip("phase-based physical reuse contract")
    first = data.prepare_forge_trace_experiment(repo, study)
    source_path = study / "inputs.yaml"
    sources = yaml.safe_load(source_path.read_text())
    sources["physical_reference"] = {
        "experiment": str(first.relative_to(repo)),
        "sha256": sha256_file(first / "experiment.json"),
    }
    source_path.write_text(yaml.safe_dump(sources))
    monkeypatch.setattr(data, "read_selected_traces", lambda *a: pytest.fail("raw data reread"))
    second = data.prepare_forge_trace_experiment(repo, study, observed_phase=1)
    a, _, _ = data.load_forge_trace_inputs(first, "pocs_grid")
    b, _, _ = data.load_forge_trace_inputs(second, "pocs_grid")
    assert set(a.observed_cells).isdisjoint(b.observed_cells)
    assert sha256_file(first / "input/waveforms.npy") == sha256_file(second / "input/waveforms.npy")
    assert sha256_file(first / "input/observed_waveforms.npy") != sha256_file(
        second / "input/observed_waveforms.npy"
    )


def test_zero_observed_waveform_remains_observed(shot_prepared):
    from dataclasses import replace

    _, experiment, _ = shot_prepared
    inputs, _, _ = data.load_forge_trace_inputs(experiment, "pocs_grid")
    values = inputs.observed_values.copy()
    values[0] = 0
    inputs = replace(inputs, observed_values=values)
    volume = inputs.model_space().volume()
    cell = inputs.observed_cells[0]
    assert volume.observed_trace_mask.reshape(-1)[cell]
    assert not volume.values.reshape(len(inputs.time_s), -1)[:, cell].any()


@pytest.mark.parametrize("method", METHODS)
def test_fixed_rms_targets_do_not_enter_any_method_and_whole_shots_are_predicted(
    shot_prepared, monkeypatch, method
):
    _, experiment, _ = shot_prepared
    inputs, config, partition = data.load_forge_trace_inputs(experiment, method)
    first = run_forge_c3_method(inputs, config, reporter=lambda _: None)
    path = experiment / "input/waveforms.npy"
    values = np.load(path).copy()
    values[partition.role.eq("test")] = 1700 + values[partition.role.eq("test")] * -9
    path.chmod(0o644)
    np.save(path, values)
    load = np.load

    def observed_only(path, **kwargs):
        assert Path(path).name != "waveforms.npy"
        return load(path, **kwargs)

    monkeypatch.setattr(np, "load", observed_only)
    second_inputs, second_config, _ = data.load_forge_trace_inputs(
        experiment, method, verify_target_waveforms=False
    )
    np.testing.assert_array_equal(inputs.test_rms, second_inputs.test_rms)
    np.testing.assert_array_equal(inputs.observed_values, second_inputs.observed_values)
    second = run_forge_c3_method(second_inputs, second_config, reporter=lambda _: None)
    np.testing.assert_array_equal(first.cell_ids, second.cell_ids)
    np.testing.assert_array_equal(first.prediction, second.prediction)
    assert set(first.cell_ids) == set(partition.cell_id[partition.role.eq("test")])
    count = int(partition.role.eq("test").sum())
    assert len(first.cell_ids) == len(set(first.cell_ids)) == count
    assert first.prediction.shape == (count, 9) and np.isfinite(first.prediction).all()


def test_all_outcomes_and_predictions_precede_scoring_and_completed_runs_are_reused(
    shot_prepared, monkeypatch
):
    repo, experiment, _ = shot_prepared
    partition = pd.read_parquet(experiment / "input/partition.parquet")
    count = int(partition.role.eq("test").sum())
    read = scoring.read_forge_trace_targets
    calls = []

    def read_finalized(path, domain):
        assert domain == "test"
        for method in METHODS:
            directory = path / "method_runs" / method
            record = json.loads((directory / "manifest.json").read_text())
            assert record["status"] in ("predicted", "completed")
            assert record["prediction_finalized_at"] and record["ended_at"]
            assert (directory / "prediction.npy").is_file()
        calls.append(domain)
        return read(path, domain)

    monkeypatch.setattr(scoring, "read_forge_trace_targets", read_finalized)
    for method in METHODS:
        record = pipeline.execute_forge_trace_method(repo, experiment, method)
        assert record["status"] == "predicted" and record["metrics"] == {}
        assert record["test_scoring_calls"] == 0
        assert "validation_score" not in record
        if method != METHODS[-1]:
            assert not scoring.finalize_forge_shot_experiment(experiment)
            assert not calls
    assert scoring.finalize_forge_shot_experiment(experiment)
    assert calls == ["test"] * 5
    monkeypatch.setattr(pipeline, "run_forge_c3_method", lambda *a, **k: pytest.fail("retrained"))
    for method in METHODS:
        result = pipeline.execute_forge_trace_method(repo, experiment, method)
        assert result["status"] == "completed" and set(result["metrics"]) == {"test"}
        assert result["metrics"]["test"]["all_eligible"]["trace_count"] == count
        if method == "regsi_real":
            train_count = int(partition.role.eq("train").sum())
            assert result["inner_mask_fraction"] == {
                "nominal": 0.25,
                "effective": 2 * np.floor(0.25 * train_count / 2) / train_count,
            }
        if method in METHODS[:3]:
            checkpoint = torch.load(
                experiment / "method_runs" / method / "final.pt", weights_only=True
            )
            assert (
                checkpoint["steps_completed"] == 1
                and checkpoint["selection_rule"] == "fixed_final_ema"
            )
    assert scoring.finalize_forge_shot_experiment(experiment) and len(calls) == 5
    table = pd.read_csv(experiment / "summary.csv")
    assert len(table) == 10 and set(table.method_id) == set(METHODS)
    shots = pd.read_csv(experiment / "per_shot.csv")
    assert len(shots) == 4 and set(shots.source_cell) == set(
        partition.loc[partition.role.eq("test"), "source_cell"]
    )
    output = export_forge_shot_results(experiment, repo / "figures")
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["run"] == str(experiment.relative_to(repo))
    assert set(provenance["method_artifacts"]) == set(METHODS)
    assert provenance["input_hashes"]["input/partition.parquet"]
    assert provenance["method_artifacts"]["regsi_real"]["prediction.npy"]
    chosen = json.loads((experiment / "input/gather_selection.json").read_text())
    assert all((output / f"gather_{name}.png").is_file() for name in chosen)


def test_failure_is_na_and_invalid_success_blocks_every_scoring_read(shot_prepared, monkeypatch):
    repo, experiment, _ = shot_prepared
    for method in METHODS[:-1]:
        pipeline.execute_forge_trace_method(repo, experiment, method)
    output = experiment / "method_runs" / METHODS[-1]
    write_json(
        output / "manifest.json",
        {
            "method_id": METHODS[-1],
            "status": "failed",
            "ended_at": "fixture",
            "failure_reason": "fixture reconstruction failed",
        },
    )
    prediction = experiment / "method_runs" / METHODS[-2] / "prediction.npy"
    values = np.load(prediction)
    np.save(prediction, values[:-1])
    monkeypatch.setattr(
        scoring, "read_forge_trace_targets", lambda *a: pytest.fail("early target read")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        scoring.finalize_forge_shot_experiment(experiment)
    np.save(prediction, values)
    monkeypatch.undo()
    assert scoring.finalize_forge_shot_experiment(experiment)
    table = pd.read_csv(experiment / "summary.csv")
    failed = table.loc[table.method_id.eq(METHODS[-1])]
    assert len(failed) == 2 and failed.mean_trace_snr_db.isna().all()
    assert failed.masked_trace_relative_mse.isna().all()
    assert failed.failure_reason.eq("fixture reconstruction failed").all()
    assert not json.loads((experiment / "outcomes.json").read_text())["all_methods_completed"]


SHOT_STUDY_METHODS = {
    "study_068_forge_shot50": METHODS,
    "study_069_forge_random_shot50": METHODS,
    "study_072_forge_random_shot50_context_field": ["cinr_real"],
}


@pytest.mark.parametrize(
    "study_name,method",
    [(study, method) for study, methods in SHOT_STUDY_METHODS.items() for method in methods],
)
def test_shot_studies_use_forge_selected_nominal_base_plus_grid(study_name, method):
    repo = Path(__file__).resolve().parents[2]
    study = repo / "studies" / study_name
    contract = yaml.safe_load((study / "config.yaml").read_text())
    reference = contract["fixed_settings"][method]
    base = yaml.safe_load((repo / reference["base"]).read_text())
    grid = yaml.safe_load((repo / reference["grid"]).read_text())["methods"][method]
    expected = next(
        c["config"]
        for c in c3_search_candidates(base, grid)
        if c["id"] == reference["candidate_id"]
    )
    actual = yaml.safe_load((study / f"{method}.yaml").read_text())
    actual["device"] = expected["device"]
    assert actual == expected
    sources = yaml.safe_load((study / "inputs.yaml").read_text())
    assert "c3_selections" not in sources
    for name, digest in sources["selection_hashes"].items():
        assert sha256_file(repo / name) == digest
    assert contract["validation_holdout"] is None
    assert contract["workflow"]["total_runs"] == len(SHOT_STUDY_METHODS[study_name])
    assert contract["methods"] == SHOT_STUDY_METHODS[study_name]


def test_random_shot_mask_is_seeded_line_balanced_and_row_order_independent(shot_prepared):
    _, experiment, contract = shot_prepared
    if contract["mask_family"] != "random_shot":
        pytest.skip("random shot mask contract")
    manifest = pd.read_parquet(experiment / "input/manifest.parquet")
    actual = pd.read_parquet(experiment / "input/partition.parquet")
    pd.testing.assert_frame_equal(actual, make_random_shot_mask(manifest.iloc[::-1], contract))
    specification = contract["interpolation_mask"]
    selected = []
    for line in range(2):
        ranks = []
        for cell in range(line * 4, (line + 1) * 4):
            key = [
                contract["study_id"],
                contract["candidate_id"],
                specification["id"],
                7302,
                line,
                cell,
            ]
            digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
            ranks.append((digest, cell))
        selected.extend(cell for _, cell in sorted(ranks)[:2])
    assert set(actual.loc[actual.role.eq("train"), "source_cell"]) == set(selected)
    assert actual.groupby("source_cell").role.nunique().eq(1).all()
    assert set(actual.cell_id) == set(manifest.cell_id) == set(range(47))
    counts = shot_mask_counts(actual, contract["spatial_shape"])
    assert counts["mask_seed"] == 7302 and "observed_phase" not in counts
    assert counts["observed_shot_count"] == counts["test_shot_count"] == 4
    assert all(
        row["observed_shot_count"] == row["test_shot_count"] == 2 for row in counts["by_line"]
    )
    assert counts["eligible_trace_missing_fraction"] == actual.role.eq("test").mean()
    changed = deepcopy(contract)
    changed["interpolation_mask"]["mask_seed"] = 7303
    assert not actual.role.equals(make_random_shot_mask(manifest, changed).role)


def test_mask_study_id_keeps_the_whole_shot_assignment_in_a_renamed_study(shot_study):
    """A re-take under a new ``study_id`` reproduces the historical acquisition.

    The whole-shot key contains the study ID, so without ``mask_study_id`` a
    renamed study silently draws a different acquisition. The non-whole-shot
    branch already namespaces its mask the same way.
    """
    repo, study, contract = shot_study
    if contract["mask_family"] != "random_shot":
        pytest.skip("random shot mask contract")
    original = pd.read_parquet(
        data.prepare_forge_trace_experiment(repo, study) / "input/partition.parquet"
    )

    def _partition(name, **overrides):
        other = shutil.copytree(study, study.parent / name)
        (other / "config.yaml").write_text(yaml.safe_dump({**contract, **overrides}))
        experiment = data.prepare_forge_trace_experiment(repo, other)
        return pd.read_parquet(experiment / "input/partition.parquet")

    renamed = _partition("renamed_study", study_id="tiny_other_shot50")
    assert not renamed.role.equals(original.role)
    namespaced = _partition(
        "namespaced_study",
        study_id="tiny_other_shot50",
        mask_study_id=contract["study_id"],
    )
    pd.testing.assert_frame_equal(namespaced, original)


def test_random_support_identifies_boundary_runs_and_internal_gaps():
    partition = pd.DataFrame(
        {
            "source_cell": range(8),
            "source_grid_line": 0,
            "source_grid_point": range(8),
            "role": ["test", "test", "train", "test", "test", "test", "train", "train"],
        }
    )
    support = random_shot_support(partition, (1, 8, 1, 1)).set_index("source_cell")
    assert support.loc[[0, 1], "shot_support"].eq("edge").all()
    assert support.loc[[3, 4, 5], "shot_support"].eq("internal").all()
    assert support.loc[1, "missing_gap_shot_count"] == 2
    assert not support.loc[1, "endpoint_shot"]
    assert support.loc[4, "missing_gap_shot_count"] == 3
    assert support.loc[4, "nearest_observed_point_distance"] == 2


def test_the_physical_neighbour_study_changes_one_declared_condition():
    """study_073 is study_072's setting with the neighbour ranking declared physical."""
    repo = Path(__file__).resolve().parents[2]
    previous = repo / "studies/study_072_forge_random_shot50_context_field"
    study = repo / "studies/study_073_forge_random_shot50_physical_neighbors"
    before = yaml.safe_load((previous / "cinr_real.yaml").read_text())
    after = yaml.safe_load((study / "cinr_real.yaml").read_text())
    assert after == {**before, "context": {**before["context"], "distance": "physical"}}
    assert before["context"].get("distance", "index") == "index"

    contract = yaml.safe_load((study / "config.yaml").read_text())
    earlier = yaml.safe_load((previous / "config.yaml").read_text())
    assert contract["methods"] == ["cinr_real"]
    assert contract["mask_study_id"] == earlier["mask_study_id"] == "study_069_forge_random_shot50"
    assert contract["interpolation_mask"] == earlier["interpolation_mask"]
    assert contract["fixed_settings"]["cinr_real"]["changes_from_selected"] == [
        "context.distance: physical"
    ]
    for key in ("candidate_id", "base", "grid", "selection", "selection_table"):
        assert (
            contract["fixed_settings"]["cinr_real"][key]
            == earlier["fixed_settings"]["cinr_real"][key]
        )
    sources = yaml.safe_load((study / "inputs.yaml").read_text())
    assert "c3_selections" not in sources
    for name, digest in sources["selection_hashes"].items():
        assert sha256_file(repo / name) == digest
    assert contract["validation_holdout"] is None and contract["workflow"]["total_runs"] == 1


def test_the_gather_excluded_study_changes_one_declared_condition():
    """study_075 is study_072's setting with the training view aligned to the test view."""
    repo = Path(__file__).resolve().parents[2]
    previous = repo / "studies/study_072_forge_random_shot50_context_field"
    study = repo / "studies/study_075_forge_random_shot50_gather_excluded"
    before = yaml.safe_load((previous / "cinr_real.yaml").read_text())
    after = yaml.safe_load((study / "cinr_real.yaml").read_text())
    assert after == {
        **before,
        "context": {**before["context"], "training_exclusion": "source"},
    }
    assert before["context"].get("training_exclusion", "query") == "query"
    assert after["model"]["input_features"] == 4

    contract = yaml.safe_load((study / "config.yaml").read_text())
    earlier = yaml.safe_load((previous / "config.yaml").read_text())
    assert contract["methods"] == ["cinr_real"]
    assert contract["mask_study_id"] == earlier["mask_study_id"] == "study_069_forge_random_shot50"
    assert contract["interpolation_mask"] == earlier["interpolation_mask"]
    assert contract["fixed_settings"]["cinr_real"]["changes_from_selected"] == [
        "context.training_exclusion: source"
    ]
    sources = yaml.safe_load((study / "inputs.yaml").read_text())
    assert "c3_selections" not in sources
    for name, digest in sources["selection_hashes"].items():
        assert sha256_file(repo / name) == digest
    assert contract["validation_holdout"] is None and contract["workflow"]["total_runs"] == 1


def test_the_aligned_baseline_study_changes_one_declared_line_per_method():
    """study_076 is study_069's two data-reading methods with the training view aligned.

    The whole-shot test view gives a query no trace of its own gather. Both methods
    default to re-masking individual traces, so without these two declared lines the
    fit reads context its test never carries. One line each, nothing else moved.
    """
    repo = Path(__file__).resolve().parents[2]
    previous = repo / "studies/study_069_forge_random_shot50"
    study = repo / "studies/study_076_forge_random_shot50_aligned_baselines"
    changes = {
        "regsi_real": ("training", "episode_kind", "random_whole_ffid", "random_trace"),
        "ccnet5d_grid": ("patches", "inner_mask_unit", "source_station", "trace"),
    }
    for method, (section, key, declared, default) in changes.items():
        before = yaml.safe_load((previous / f"{method}.yaml").read_text())
        after = yaml.safe_load((study / f"{method}.yaml").read_text())
        assert before[section].get(key, default) == default
        assert after == {**before, section: {**before[section], key: declared}}

    contract = yaml.safe_load((study / "config.yaml").read_text())
    earlier = yaml.safe_load((previous / "config.yaml").read_text())
    assert contract["methods"] == ["regsi_real", "ccnet5d_grid"]
    assert contract["workflow"]["total_runs"] == 2
    assert contract["mask_study_id"] == "study_069_forge_random_shot50"
    assert contract["interpolation_mask"] == earlier["interpolation_mask"]
    assert contract["matched_training"] == earlier["matched_training"]
    assert contract["validation_holdout"] is None
    for method, (section, key, declared, _) in changes.items():
        reference = contract["fixed_settings"][method]
        assert reference["changes_from_selected"] == [f"{section}.{key}: {declared}"]
        for name in ("candidate_id", "base", "grid", "selection", "selection_table"):
            assert reference[name] == earlier["fixed_settings"][method][name]
    sources = yaml.safe_load((study / "inputs.yaml").read_text())
    assert "c3_selections" not in sources
    for name, digest in sources["selection_hashes"].items():
        assert sha256_file(repo / name) == digest


@pytest.mark.parametrize(
    "method,section,key,declared",
    [
        ("regsi_real", "training", "episode_kind", "random_whole_ffid"),
        ("ccnet5d_grid", "patches", "inner_mask_unit", "source_station"),
    ],
)
def test_the_declared_training_view_reaches_the_fit(shot_prepared, method, section, key, declared):
    """A config-level assertion would miss the wiring; run the fit and read it back."""
    _, experiment, _ = shot_prepared
    inputs, config, partition = data.load_forge_trace_inputs(experiment, method)
    assert key not in config[section]
    default = run_forge_c3_method(inputs, config, reporter=lambda _: None)

    config[section][key] = declared
    aligned = run_forge_c3_method(inputs, config, reporter=lambda _: None)
    assert aligned.diagnostics[key] == declared
    assert default.diagnostics[key] != declared
    count = int(partition.role.eq("test").sum())
    assert aligned.prediction.shape == (count, 9) and np.isfinite(aligned.prediction).all()
    # The declared view changes what the fit sees, so it must change what the fit learns.
    assert not np.array_equal(aligned.prediction, default.prediction)
