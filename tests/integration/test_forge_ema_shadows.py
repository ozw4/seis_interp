import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.forge_mvp_artifacts import load_model_inputs, verify_hashes
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.tune_forge_matched import (
    matched_candidate_config,
    require_holdout_selection,
    search_scoring,
    tune_forge_matched,
)
from seis_interp.training import forge_mvp_methods
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.forge_mvp_methods import (
    _window_loss,
    ema_eval_decays,
    ema_snapshot_updates,
    run_method,
    shadow_key,
)
from tests.fixtures.forge_mvp import tiny_forge_study

NEURAL = ("ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid")
DECAYS = [0.5, 0.8, 0.95]
STEPS = 6


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    study, contract, _, configs, *_ = tiny_forge_study(tmp_path)
    contract["preprocessing"].update(
        amplitude_scaling="trace_oracle", trace_rms=True, alignment={"velocity_m_per_s": 40000.0}
    )
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    for name in NEURAL:
        configs[name]["training"].update(max_steps=STEPS, report_interval=2)
        (study / f"{name}.yaml").write_text(yaml.safe_dump(configs[name]))
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.tune_forge_matched.execution_identity",
        lambda _: {"git_commit": "testcommit", "dirty_worktree": True},
    )
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield tmp_path, prepare_forge_mvp(tmp_path, study)
    torch.set_num_threads(previous)


def _config(preparation, method, **training):
    inputs, config = load_model_inputs(preparation, method)
    config["training"].update(training)
    return inputs, config


def _run(inputs, config):
    return run_method(inputs, config, reporter=lambda message: None)


def _without_timings(history):
    return [{k: v for k, v in row.items() if not k.endswith("seconds")} for row in history]


@pytest.mark.parametrize(
    "method,shared",
    [("ccnet5d_grid", False), ("nersi_real", False), ("regsi_real", False), ("regsi_real", True)],
)
def test_prediction_heartbeats_preserve_primary_shadows_rng_and_order(
    preparation, monkeypatch, method, shared
):
    from seis_interp.training import phase_progress
    from seis_interp.training.exact_resume import rng_state

    _, prepared = preparation
    inputs, config = _config(prepared, method, ema_decay=0.8, ema_eval_decays=DECAYS)
    if shared:
        config["training"]["execution"] = {"shared_prediction": True}
    now = 0.0
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: now)
    calls = []
    advance = False
    original = forge_mvp_methods.initialize_model

    def initialize(config):
        model = original(config)

        def advanced(_, args):
            nonlocal now
            if advance:
                now += 31
            value = args[0]
            if isinstance(value, torch.Tensor):
                calls.append((value.shape, value.detach().numpy().tobytes()))
            else:
                calls.append(
                    tuple(
                        getattr(value, field).numpy().tobytes()
                        for field in ("waveforms", "node_features", "edge_index", "query_indices")
                    )
                )

        model.register_forward_pre_hook(advanced)
        return model

    monkeypatch.setattr(forge_mvp_methods, "initialize_model", initialize)
    baseline = _run(inputs, config)
    before = rng_state()
    baseline_calls = calls.copy()
    calls.clear()
    advance = True
    reports = []
    result = run_method(inputs, config, reporter=reports.append)
    assert result.prediction.tobytes() == baseline.prediction.tobytes()
    assert result.cell_ids.tobytes() == baseline.cell_ids.tobytes()
    for key, values in baseline.shadow_predictions.items():
        assert result.shadow_predictions[key].tobytes() == values.tobytes()
    assert _without_timings(result.history) == _without_timings(baseline.history)
    after = rng_state()
    assert before["python"] == after["python"]
    assert all(np.array_equal(a, b) for a, b in zip(before["numpy"], after["numpy"], strict=True))
    assert torch.equal(before["torch"], after["torch"])
    heartbeats = [
        message for message in reports if isinstance(message, phase_progress.ProgressHeartbeat)
    ]
    for label in ("primary", "ema 0.5", "ema 0.95"):
        assert any(f"({label})" in message for message in heartbeats), heartbeats
    assert calls == baseline_calls


@pytest.mark.parametrize("method", NEURAL)
def test_shadows_equal_separately_trained_ema_and_keep_the_primary(preparation, method):
    _, prepared = preparation
    inputs, plain = _config(prepared, method, ema_decay=0.8)
    baseline = _run(inputs, plain)
    assert baseline.shadow_predictions == {}
    inputs, config = _config(prepared, method, ema_decay=0.8, ema_eval_decays=DECAYS)
    result = _run(inputs, config)
    # The primary prediction, history and checkpoint are those of the run without shadows.
    np.testing.assert_array_equal(result.prediction, baseline.prediction)
    assert _without_timings(result.history) == _without_timings(baseline.history)
    for name, value in baseline.checkpoint["model_state_dict"].items():
        assert torch.equal(result.checkpoint["model_state_dict"][name], value)
    assert list(result.shadow_predictions) == [shadow_key(d) for d in DECAYS]
    assert result.shadow_predictions[shadow_key(0.8)] is result.prediction
    for decay in DECAYS:
        inputs, separate = _config(prepared, method, ema_decay=decay)
        expected = _run(inputs, separate).prediction
        shadow = result.shadow_predictions[shadow_key(decay)]
        assert shadow.dtype == np.float32 and shadow.shape == expected.shape
        np.testing.assert_array_equal(shadow, expected)
    assert not np.array_equal(
        result.shadow_predictions[shadow_key(0.5)], result.shadow_predictions[shadow_key(0.95)]
    )
    diagnostics = result.diagnostics
    assert diagnostics["runtime_training_seconds"] > 0
    assert diagnostics["runtime_prediction_seconds"] > 0
    assert diagnostics["runtime_shadow_prediction_seconds"] > 0
    assert np.isfinite(diagnostics["final_training_loss"])
    assert diagnostics["final_training_loss_window_updates"] == STEPS


@pytest.mark.parametrize("method", ["ccnet5d_grid", "regsi_real"])
def test_primary_ema0999_stays_bitwise_with_round2_shadows(preparation, method):
    _, prepared = preparation
    inputs, plain = _config(prepared, method, ema_decay=0.999)
    if method == "regsi_real":
        plain["graph"]["neighbor_search"] = "exact_index"
        plain["training"]["execution"] = {
            "neighbor_table": True,
            "prefetch": True,
            "deferred_checks": True,
            "foreach_ema": True,
            "shared_prediction": True,
        }
    expected = _run(inputs, plain)
    plain["training"]["ema_eval_decays"] = [0.999, 0.9995, 0.9998]
    actual = _run(inputs, plain)
    np.testing.assert_array_equal(actual.prediction, expected.prediction)
    for name, value in expected.checkpoint["model_state_dict"].items():
        assert torch.equal(value, actual.checkpoint["model_state_dict"][name])
    assert set(actual.shadow_predictions) == {"0.999", "0.9995", "0.9998"}
    assert "would_clip_fraction" in actual.diagnostics
    assert "zero_predictor_loss" in actual.diagnostics["g0"]


def test_memory_probe_fresh_timing_path_runs_on_cpu(preparation):
    from seis_interp.training.ccnet5d_memory_probe import measure_ccnet5d_memory

    _, prepared = preparation
    inputs, config = _config(
        prepared, "ccnet5d_grid", ema_decay=0.999, supervised_traces_per_update=2
    )
    config.setdefault("execution", {}).update(reuse_patch=False, cache_patch_gradients=False)
    result = measure_ccnet5d_memory(inputs, config, "cpu", optimizer_updates=3)
    assert result["training_probe_optimizer_updates"] == 3
    assert result["seconds_per_update"] > 0
    assert result["diagnostics"]["g0"]["status"] == "insufficient_updates"
    assert result["inference_forward_dtype"] == "torch.float32"
    assert result["prediction_calls"] == 1
    assert result["measured_peak_reserved_gib_ceil"] == 0


@pytest.mark.parametrize(
    "training",
    [
        {"ema_decay": 0.8, "ema_eval_decays": [0.5, 0.95]},
        {"ema_decay": 0.8, "ema_eval_decays": [0.8, 0.8]},
        {"ema_decay": 0.8, "ema_eval_decays": []},
        {"ema_decay": 0.8, "ema_eval_decays": [0.8, 1.0]},
        {"ema_decay": None, "ema_eval_decays": [0.8]},
        {"ema_decay": 0.8, "ema_eval_decays": 0.8},
    ],
)
def test_invalid_shadow_decays_are_rejected(preparation, training):
    root, prepared = preparation
    with pytest.raises(ValueError):
        ema_eval_decays(training)
    _, base = load_model_inputs(prepared, "nersi_real")
    with pytest.raises(ValueError):
        matched_candidate_config(base, {"name": "bad", "training": training}, "cpu", None, 10, 7)


@pytest.mark.parametrize("method", NEURAL)
def test_shadow_weights_and_snapshots_equal_separately_trained_ema(preparation, method):
    _, prepared = preparation
    inputs, plain = _config(prepared, method, ema_decay=0.8, ema_eval_decays=DECAYS)
    baseline = _run(inputs, plain)
    inputs, config = _config(
        prepared, method, ema_decay=0.8, ema_eval_decays=DECAYS, ema_snapshot_updates=[3]
    )
    result = _run(inputs, config)
    # Snapshots never touch fitting: predictions and primary weights are unchanged.
    np.testing.assert_array_equal(result.prediction, baseline.prediction)
    for key, values in baseline.shadow_predictions.items():
        np.testing.assert_array_equal(result.shadow_predictions[key], values)
    for name, value in baseline.checkpoint["model_state_dict"].items():
        assert torch.equal(result.checkpoint["model_state_dict"][name], value)
    assert sorted(baseline.ema_states) == sorted(shadow_key(d) for d in DECAYS if d != 0.8)
    assert sorted(result.ema_states) == sorted(shadow_key(d) for d in DECAYS)
    for decay in DECAYS:
        states = result.ema_states[shadow_key(decay)]
        assert set(states) == ({3} if decay == 0.8 else {3, STEPS})
        for update, state in states.items():
            # A constant-rate run of ``update`` steps follows the same trajectory.
            inputs, separate = _config(prepared, method, ema_decay=decay, max_steps=update)
            expected = _run(inputs, separate).checkpoint["model_state_dict"]
            assert list(state) == list(expected)
            for name, value in expected.items():
                assert torch.equal(state[name], value), (decay, update, name)


@pytest.mark.parametrize(
    "training",
    [
        {"ema_decay": 0.8, "max_steps": 6, "ema_snapshot_updates": [0]},
        {"ema_decay": 0.8, "max_steps": 6, "ema_snapshot_updates": [6]},
        {"ema_decay": 0.8, "max_steps": 6, "ema_snapshot_updates": [3, 3]},
        {"ema_decay": 0.8, "max_steps": 6, "ema_snapshot_updates": []},
        {"ema_decay": 0.8, "max_steps": 6, "ema_snapshot_updates": 3},
        {"ema_decay": 0.8, "max_steps": 6, "ema_snapshot_updates": [True]},
        {"ema_decay": None, "max_steps": 6, "ema_snapshot_updates": [3]},
    ],
)
def test_invalid_snapshot_updates_are_rejected(training):
    with pytest.raises(ValueError, match="ema_snapshot_updates"):
        ema_snapshot_updates(training)


@pytest.mark.parametrize("method", NEURAL)
def test_training_loss_window_covers_only_the_last_updates(preparation, method, monkeypatch):
    _, prepared = preparation
    monkeypatch.setattr(forge_mvp_methods, "LOSS_WINDOW_UPDATES", 2)
    inputs, config = _config(prepared, method, report_interval=1)
    result = _run(inputs, config)
    losses = [row["loss"] for row in result.history]
    assert len(losses) == STEPS
    assert result.diagnostics["final_training_loss_window_updates"] == 2
    assert result.diagnostics["final_training_loss_window_mean"] == pytest.approx(
        np.mean(losses[-2:]), rel=1e-12
    )


def test_window_loss_weights_report_spans_and_keeps_a_straddling_span():
    # Reports every 300 updates of 2,500: the last 1,000 updates are rows 1800..2500.
    steps = [*range(300, 2500, 300), 2500]
    history = [{"step": step, "loss": float(step)} for step in steps]
    window = _window_loss(history, 2500)
    assert window["final_training_loss_window_updates"] == 1000
    expected = (1800 * 300 + 2100 * 300 + 2400 * 300 + 2500 * 100) / 1000
    assert window["final_training_loss_window_mean"] == pytest.approx(expected, rel=1e-15)
    # A report span that straddles the window start is counted whole.
    history = [{"step": step, "loss": float(step)} for step in (700, 1400, 2100, 2500)]
    window = _window_loss(history, 2500)
    assert window["final_training_loss_window_updates"] == 700 + 400
    assert window["final_training_loss_window_mean"] == pytest.approx(
        (2100 * 700 + 2500 * 400) / 1100, rel=1e-15
    )
    # Shorter runs average their whole history.
    assert _window_loss([{"step": 4, "loss": 2.0}, {"step": 6, "loss": 5.0}], 6) == {
        "final_training_loss_window_mean": 3.0,
        "final_training_loss_window_updates": 6,
    }


def test_regsi_records_gradient_clip_rate_and_support(preparation):
    _, prepared = preparation
    for clip, fraction in ((1e-9, 1.0), (1e9, 0.0)):
        inputs, config = _config(prepared, "regsi_real", gradient_clip_norm=clip)
        result = _run(inputs, config)
        norms = np.array([row["gradient_norm"] for row in result.history])
        assert len(norms) == STEPS and (norms > 0).all()
        assert result.diagnostics["gradient_clip_fraction"] == fraction
        assert result.diagnostics["mean_gradient_norm"] == pytest.approx(norms.mean())
        support = [row["subgraph_support_node_count"] for row in result.history]
        assert result.diagnostics["mean_subgraph_support_node_count"] == np.mean(support)
        assert result.diagnostics["final_training_loss"] == result.history[-1]["loss"]
        assert result.diagnostics["final_training_loss_window_mean"] == pytest.approx(
            np.mean([row["loss"] for row in result.history])
        )
    inputs, config = _config(prepared, "regsi_real", gradient_clip_norm=None)
    result = _run(inputs, config)
    assert "gradient_norm" not in result.history[0]
    assert "gradient_clip_fraction" not in result.diagnostics


def test_ccnet_records_visible_context_per_update(preparation, monkeypatch):
    _, prepared = preparation
    drawn = []
    sample = CCNet5DObservedPatchSource.sample

    def recorded(self):
        patch = sample(self)
        observed = self._observed_mask[patch.patch_slices[1:]]
        drawn.append((int(patch.visible_observed_mask.sum()), int(observed.sum())))
        return patch

    monkeypatch.setattr(CCNet5DObservedPatchSource, "sample", recorded)
    inputs, config = _config(prepared, "ccnet5d_grid")
    result = _run(inputs, config)
    diagnostics = result.diagnostics
    visible, available = np.array(drawn).T
    # Visible context excludes the hidden pseudo-targets of every drawn patch.
    assert diagnostics["mean_patches_per_update"] == len(drawn) / STEPS
    assert sum(diagnostics["patches_per_update"]) == len(drawn)
    assert len(diagnostics["patches_per_update"]) == STEPS
    assert diagnostics["mean_visible_observed_cells_per_update"] == visible.sum() / STEPS
    assert (visible < available).all()
    assert diagnostics["mean_patches_per_update"] >= 1
    # A patch holds at most every observed cell as visible context.
    assert (
        0
        < diagnostics["mean_visible_observed_cells_per_update"]
        <= (diagnostics["mean_patches_per_update"] * len(inputs.observed_cells))
    )
    losses = [row["loss"] for row in result.history]
    assert diagnostics["final_training_loss_window_mean"] == pytest.approx(np.mean(losses))


def test_candidate_seed_override_changes_initialization(preparation):
    _, prepared = preparation
    inputs, base = load_model_inputs(prepared, "nersi_real")
    config, _ = matched_candidate_config(
        base, {"name": "seed", "model_seed": 7302}, "cpu", None, 10, 7
    )
    assert config["model_seed"] == 7302 and base["model_seed"] != 7302
    assert "Replace the model seed" in config["changes_from_source"][-1]
    same, _ = matched_candidate_config(base, {"name": "same"}, "cpu", None, 10, 7)
    replaced, default = _run(inputs, config), _run(inputs, same)
    assert replaced.checkpoint["model_seed"] == 7302
    assert replaced.diagnostics["initialization_hash"] != default.diagnostics["initialization_hash"]
    for method, seed in (("nersi_real", True), ("nersi_real", 1.5), ("pocs_grid", 7302)):
        _, other = load_model_inputs(prepared, method)
        with pytest.raises(ValueError):
            matched_candidate_config(other, {"name": "bad", "model_seed": seed}, "cpu", None, 10, 7)


def _search(root, method, candidates, **options):
    path = root / f"search_{len(list(root.glob('search_*.yaml')))}.yaml"
    body = {"method_id": method, "device": "cpu", "candidates": candidates, **options}
    path.write_text(yaml.safe_dump(body))
    return path


def test_tuner_scores_every_shadow_on_the_final_test_set(preparation):
    root, prepared = preparation
    candidates = [
        {"name": "shadows", "training": {"ema_eval_decays": [0.5, 0.999, 0.95]}},
        {"name": "seeded", "model_seed": 11},
    ]
    output = tune_forge_matched(
        root, prepared, _search(root, "regsi_real", candidates, target_relative_mse=100.0)
    )
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["selection_domain"] == "final_test" and record["purpose"] == "selection"
    assert record["independent_test_estimate"] is False and record["evaluate"] is True
    rows = pd.read_csv(output / "search_results.csv")
    assert list(zip(rows.name, rows.ema_decay.astype(str), strict=True)) == [
        ("shadows", "0.5"),
        ("shadows", "0.999"),
        ("shadows", "0.95"),
        ("seeded", "0.999"),
    ]
    assert rows.primary_checkpoint.tolist() == [False, True, False, True]
    assert rows.model_seed.tolist() == [7301, 7301, 7301, 11]
    assert (rows.runtime_training_seconds > 0).all()
    best = record["best_candidate"]
    assert best["primary_relative_mse"] == pytest.approx(rows.primary_relative_mse.min())
    assert best["ema_decay"] == str(rows.ema_decay[rows.primary_relative_mse.idxmin()])
    inputs, config = load_model_inputs(prepared, "regsi_real")
    config["training"]["ema_eval_decays"] = [0.5, 0.999, 0.95]
    expected = _run(inputs, config)
    run = json.loads((output / "shadows/run_manifest.json").read_text())
    assert run["ema_decay"] == "0.999" and set(run["ema_shadows"]) == {"0.5", "0.999", "0.95"}
    for key, values in expected.shadow_predictions.items():
        directory = output / "shadows" / f"ema_{key}"
        np.testing.assert_array_equal(np.load(directory / "prediction.npy"), values)
        metrics = json.loads((directory / "metrics.json").read_text())
        score = metrics["clean_target"]["masked_trace_relative_mse"]
        assert run["ema_shadows"][key]["primary_relative_mse"] == score
        assert (directory / "metrics_by_trace.parquet").is_file()
    np.testing.assert_array_equal(
        np.load(output / "shadows/prediction.npy"), expected.shadow_predictions["0.999"]
    )
    assert (output / "shadows/metrics.json").read_bytes() == (
        output / "shadows/ema_0.999/metrics.json"
    ).read_bytes()
    assert not (output / "seeded/ema_0.999").exists()
    resolved = yaml.safe_load((output / "seeded/resolved_config.yaml").read_text())
    assert resolved["model_seed"] == 11


def test_unevaluated_search_saves_every_shadow_without_reading_targets(preparation, monkeypatch):
    root, prepared = preparation

    def forbidden(*args, **kwargs):
        raise AssertionError("an unevaluated search must not read or score targets")

    for name in ("read_selected_traces", "evaluate_traces", "verify_hashes"):
        monkeypatch.setattr(f"seis_interp.pipelines.tune_forge_matched.{name}", forbidden)
    candidates = [
        {
            "name": "final",
            "model_seed": 7311,
            "training": {
                "ema_decay": 0.8,
                "ema_eval_decays": [0.8, 0.95],
                "ema_snapshot_updates": [3],
            },
        }
    ]
    search = _search(root, "ccnet5d_grid", candidates, evaluate=False, purpose="final_evaluation")
    output = tune_forge_matched(root, prepared, search)
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "completed" and record["evaluate"] is False
    assert record["independent_test_estimate"] is True
    assert "best_candidate" not in record and record["target_relative_mse"] is None
    assert not list(output.rglob("metrics*"))
    for relative in ("prediction.npy", "ema_0.8/prediction.npy", "ema_0.95/prediction.npy"):
        assert f"final/{relative}" in record["artifacts"]
    assert (output / "final/final.pt").is_file()
    assert (output / "final/training_log.parquet").is_file()
    # Every shadow's weights are kept: final non-primary shadows and every snapshot.
    saved = {
        "final/ema_0.95/final.pt": (0.95, STEPS),
        "final/update_3/ema_0.8.pt": (0.8, 3),
        "final/update_3/ema_0.95.pt": (0.95, 3),
    }
    assert sorted(name for name in record["artifacts"] if name.endswith(".pt")) == sorted(
        ["final/final.pt", *saved]
    )
    primary = torch.load(output / "final/final.pt", weights_only=False)
    for relative, (decay, update) in saved.items():
        checkpoint = torch.load(output / relative, weights_only=False)
        assert (checkpoint["ema_decay"], checkpoint["steps_completed"]) == (decay, update)
        assert checkpoint["model_seed"] == 7311 == primary["model_seed"]
        assert checkpoint["model_state_dict"].keys() == primary["model_state_dict"].keys()
    rows = pd.read_csv(output / "search_results.csv")
    assert "primary_relative_mse" not in rows and len(rows) == 2


@pytest.mark.parametrize(
    "seal,search,message",
    [
        (
            {"selection_domain": "validation_holdout"},
            {"purpose": "final_evaluation", "evaluate": False},
            "final evaluation requires a final-test preparation",
        ),
        (
            {"selection_domain": "validation_holdout"},
            {"purpose": "final_evaluation", "target_relative_mse": 1.0},
            "final evaluation requires a final-test preparation",
        ),
        (
            {"selection_domain": "original_clean_target_test_set"},
            {"target_relative_mse": 1.0},
            "unsupported selection domain",
        ),
        ({}, {"purpose": "tuning", "target_relative_mse": 1.0}, "unsupported .* search purpose"),
        ({}, {"evaluate": "no", "target_relative_mse": 1.0}, "evaluate must be boolean"),
        ({}, {"evaluate": False, "stop_when_target_met": True}, "requires evaluate: true"),
        ({}, {}, "declares target_relative_mse"),
    ],
)
def test_invalid_scoring_declarations_are_rejected(seal, search, message):
    # Each case is valid apart from the guarded declaration, so only that guard can fire.
    with pytest.raises(ValueError, match=message):
        search_scoring(seal, search)


HOLDOUT = {"validation_holdout": {"fraction": 0.1, "mask_id": "v", "mask_seed": 1}}


@pytest.mark.parametrize(
    "contract,seal,search,allowed",
    [
        (HOLDOUT, {}, {"target_relative_mse": 1.0}, False),
        (HOLDOUT, {}, {"purpose": "final_evaluation", "target_relative_mse": 1.0}, False),
        (HOLDOUT, {}, {"evaluate": False}, False),
        (HOLDOUT, {}, {"purpose": "final_evaluation", "evaluate": False}, True),
        (HOLDOUT, {"selection_domain": "validation_holdout"}, {"target_relative_mse": 1.0}, True),
        ({}, {}, {"target_relative_mse": 1.0}, True),
    ],
)
def test_a_holdout_study_scores_its_final_test_only_in_the_unevaluated_final_runs(
    contract, seal, search, allowed
):
    scoring = search_scoring(seal, search)
    if allowed:
        require_holdout_selection(contract, scoring)
    else:
        with pytest.raises(ValueError, match="selects on its validation holdout"):
            require_holdout_selection(contract, scoring)


def test_scoring_domain_follows_the_seal():
    assert search_scoring({}, {"target_relative_mse": 1.0}) == {
        "selection_domain": "final_test",
        "purpose": "selection",
        "evaluate": True,
        "independent_test_estimate": False,
    }
    validation = search_scoring({"selection_domain": "validation_holdout"}, {"evaluate": False})
    assert validation["selection_domain"] == "validation_holdout"
    assert validation["independent_test_estimate"] is False
