"""Profile-wise NeRSI added to sealed FORGE preparations (the study_061 machinery).

A tiny matched, aligned, trace-oracle parent study with a validation holdout is
prepared once per module (final and validation preparations) and both are
re-sealed for a derived study that adds ``nersi_profile``. At 20 km/s the
geometry-only moveout canvas is 9 + 12 = 21 samples, so the decoder is built on
a 24-sample profile while the study config keeps ``profile_side(9) = 16``.
"""

import json
import shutil
import stat
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.forge_aligned_inputs import align_model_inputs, unalign_predictions
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, verify_hashes
from seis_interp.forge_mvp_config import profile_side
from seis_interp.models.nersi import Nersi
from seis_interp.pipelines import derive_forge_preparation as derivation
from seis_interp.pipelines.derive_forge_preparation import derive_forge_preparation
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.tune_forge_matched import matched_candidate_config, tune_forge_matched
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.processing.forge_nersi_profiles import (
    ExactTraceProfileSampler,
    nersi_profile_layout,
)
from seis_interp.training import forge_mvp_methods
from seis_interp.training.forge_mvp_methods import model_parameter_count, run_method
from tests.fixtures.forge_mvp import tiny_forge_study

VELOCITY = 20000.0
TIME_COUNT = 9
CANVAS = 21
STEPS = 4
TRACES = 2
DECAYS = [0.5, 0.8, 0.95]
PARENT_STUDY = "tiny_forge_mvp"
DERIVED_STUDY = "tiny_forge_nersi_profile"
SEAL = "preparation_manifest.json"
CONTRACT = "input/input_contract.json"
SHARED_BINDING = (
    "amplitude_scaling",
    "trace_scale_hash",
    "idw",
    "alignment",
    "mask_hash",
    "mask_id",
    "mask_seed",
    "global_rms",
)
BUDGET = {
    "max_steps": STEPS,
    "learning_rate": 0.001,
    "loss_traces_per_update": TRACES,
    "parameter_count_target": 1300,
    "parameter_count_tolerance": 0.95,
}
PROFILE_CONFIG = {
    "method_id": "nersi_profile",
    "geometry_mode": "real",
    "model_seed": 7301,
    "loss": "masked_trace_relative_mse",
    "checkpoint_rule": "fixed_final_ema",
    "device": "cpu",
    "model": {
        "input_features": 3,
        "fourier_components": 2,
        "frequency_base": [1.1, 1.1, 1.1],
        "encoder_width": 4,
        "latent_channels": 1,
        "decoder_channels": [2, 2, 2],
        "profile_shape": [profile_side(TIME_COUNT), profile_side(3)],
        "upsample_scales": [2, 2, 2],
        "kernel_size": 1,
        "activation": "gelu",
        "output_activation": "linear",
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "max_steps": STEPS,
        "weight_decay": 0.0,
        "report_interval": 1,
        "learning_rate_schedule": "constant",
        "mixed_precision": "off",
        "supervised_traces_per_update": TRACES,
        "ema_decay": 0.8,
    },
    # Two profiles per prediction batch: the test profiles span several batches.
    "prediction": {"profile_batch_size": 2},
}


def _parent_study(root):
    """Matched study_060-like parent: 21 observed / 14 test, holdout 11 observed / 10 V."""
    study, contract, _, configs, *_ = tiny_forge_study(root)
    contract["preprocessing"].update(
        amplitude_scaling="trace_oracle", trace_rms=True, alignment={"velocity_m_per_s": VELOCITY}
    )
    contract.update(
        mask_study_id=study.name,
        missing_fraction=0.4,
        test_trace_count=14,
        observed_trace_count=21,
        matched_training=BUDGET,
        validation_holdout={"fraction": 0.5, "mask_id": "tiny_validation", "mask_seed": 11},
    )
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    configs["nersi_real"]["training"]["batch_size"] = TRACES
    configs["ccnet5d_grid"]["training"]["supervised_traces_per_update"] = TRACES
    for name in ("regsi_real", "regsi_grid"):
        configs[name]["training"].update(inner_mask_fraction=0.5, query_batch_size=TRACES)
    for name in ("nersi_real", "ccnet5d_grid", "regsi_real", "regsi_grid"):
        configs[name]["training"]["max_steps"] = STEPS
        (study / f"{name}.yaml").write_text(yaml.safe_dump(configs[name]))
    return study, contract


def _write_study(root, name, contract, methods):
    """A study directory adding ``methods`` ({method: yaml body}) on the parent contract."""
    study = root / "studies" / name
    study.mkdir(parents=True)
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    for method, body in methods.items():
        (study / f"{method}.yaml").write_text(yaml.safe_dump(body, sort_keys=False))
    return study


def _identity(commit):
    return lambda _: {"git_commit": commit, "dirty_worktree": False}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("forge_nersi_profile")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
            lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
        )
        patch.setattr(
            "seis_interp.pipelines.derive_forge_preparation.execution_identity",
            _identity("derivecommit"),
        )
        parent_study, contract = _parent_study(root)
        parents = {
            "final": prepare_forge_mvp(root, parent_study),
            "validation": prepare_forge_mvp(root, parent_study, validation_holdout=True),
        }
        derived_contract = {**contract, "study_id": DERIVED_STUDY}
        study = _write_study(
            root, DERIVED_STUDY, derived_contract, {"nersi_profile": PROFILE_CONFIG}
        )
        derived = {
            domain: derive_forge_preparation(root, parent, study, ["nersi_profile"])
            for domain, parent in parents.items()
        }
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (root / "scripts" / script.name).write_bytes(script.read_bytes())
    return SimpleNamespace(
        root=root, contract=derived_contract, parents=parents, derived=derived, study=study
    )


@pytest.fixture(autouse=True)
def execution(monkeypatch):
    monkeypatch.setattr(
        "seis_interp.pipelines.derive_forge_preparation.execution_identity",
        _identity("derivecommit"),
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.tune_forge_matched.execution_identity",
        lambda _: {"git_commit": "tunecommit", "dirty_worktree": True},
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _seal(preparation):
    return json.loads((preparation / SEAL).read_text())


# --- derivation -------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["final", "validation"])
def test_derived_preparation_reseals_identical_inputs_with_one_added_method(runs, domain):
    root, parent, derived = runs.root, runs.parents[domain], runs.derived[domain]
    parent_seal, seal = _seal(parent), _seal(derived)
    suffix = "_validation" if domain == "validation" else ""
    assert derived.parent == root / "runs" / DERIVED_STUDY
    assert derived.name.endswith("_derivecommit" + suffix)
    # Every parent artifact except the input contract is copied byte for byte.
    copied = set(parent_seal["artifacts"]) - {CONTRACT}
    assert CONTRACT in parent_seal["artifacts"] and copied
    for relative in sorted(copied):
        assert (derived / relative).read_bytes() == (parent / relative).read_bytes(), relative
        assert seal["artifacts"][relative] == parent_seal["artifacts"][relative]
    added = {
        "configs/nersi_profile.yaml",
        f"source/studies/{DERIVED_STUDY}/config.yaml",
        f"source/studies/{DERIVED_STUDY}/nersi_profile.yaml",
    }
    assert set(seal["artifacts"]) == set(parent_seal["artifacts"]) | added
    files = {str(path.relative_to(derived)) for path in derived.rglob("*") if path.is_file()}
    assert files == set(seal["artifacts"]) | {SEAL}
    verify_hashes(derived, seal["artifacts"])
    for name in ("config.yaml", "nersi_profile.yaml"):
        copy = derived / "source/studies" / DERIVED_STUDY / name
        assert copy.read_bytes() == (runs.study / name).read_bytes()
    # The contract differs from the parent's only in study_id.
    parent_contract = json.loads((parent / CONTRACT).read_text())
    contract = json.loads((derived / CONTRACT).read_text())
    assert parent_contract["study_id"] == PARENT_STUDY
    assert contract == {**parent_contract, "study_id": DERIVED_STUDY}
    assert (contract.get("selection_domain") == "validation_holdout") == (domain == "validation")
    # The added config is the study yaml bound exactly like the parent's own methods.
    resolved = yaml.safe_load((derived / "configs/nersi_profile.yaml").read_text())
    parent_configs = [yaml.safe_load(p.read_text()) for p in (parent / "configs").glob("*.yaml")]
    binding = {key: parent_configs[0][key] for key in SHARED_BINDING if key in parent_configs[0]}
    assert all({key: c[key] for key in binding} == binding for c in parent_configs)
    assert resolved == {**PROFILE_CONFIG, **binding}
    assert resolved["alignment"] == seal["alignment"] == parent_seal["alignment"]
    assert resolved["mask_hash"] == seal["mask_hash"]
    assert resolved["mask_id"] == ("tiny_validation" if suffix else "m1_random80_mvp_v1")
    # The seal keeps the parent's fields and records where it came from; the parent's
    # code identity moves under derived_from.
    identity = ("git_commit", "dirty_worktree", "implementation_hashes")
    changed = {"study_id", *identity, "start_time", "end_time"}
    changed |= {"parameter_counts", "artifacts"}
    assert set(seal) == (set(parent_seal) - {"implementation_hashes"}) | {"derived_from"}
    assert {k: v for k, v in seal.items() if k not in changed | {"derived_from"}} == {
        k: v for k, v in parent_seal.items() if k not in changed
    }
    assert (seal["study_id"], seal["git_commit"], seal["dirty_worktree"]) == (
        DERIVED_STUDY,
        "derivecommit",
        False,
    )
    assert seal["status"] == "prepared" and seal["canvas_sample_count"] == CANVAS
    assert seal["parameter_counts"] == {
        **parent_seal["parameter_counts"],
        "nersi_profile": model_parameter_count(PROFILE_CONFIG, CANVAS),
    }
    assert seal["derived_from"] == {
        "preparation": str(parent.resolve().relative_to(root.resolve())),
        "manifest_sha256": sha256_file(parent / SEAL),
        "added_methods": ["nersi_profile"],
        "derivation_source_sha256": sha256_file(Path(derivation.__file__)),
        **{f"parent_{key}": parent_seal[key] for key in identity},
    }
    for path in derived.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o444, path


@pytest.mark.parametrize("domain", ["final", "validation"])
def test_derived_inputs_equal_the_parent_inputs(runs, domain):
    parent, derived = runs.parents[domain], runs.derived[domain]
    inputs, config = load_model_inputs(derived, "nersi_profile")
    reference, _ = load_model_inputs(parent, "nersi_real")
    assert config == yaml.safe_load((derived / "configs/nersi_profile.yaml").read_text())
    arrays = ("observed_cells", "test_cells", "excluded_cells", "observed_values")
    for name in (*arrays, "observed_rms", "test_rms"):
        np.testing.assert_array_equal(getattr(inputs, name), getattr(reference, name))
    pd.testing.assert_frame_equal(inputs.geometry, reference.geometry)
    assert inputs.global_rms == reference.global_rms
    assert inputs.spatial_shape == reference.spatial_shape == (2, 3, 2, 3)
    assert inputs.binding == {
        **reference.binding,
        "study_id": DERIVED_STUDY,
        "preparation_hash": sha256_file(derived / SEAL),
    }
    sizes = tuple(len(getattr(inputs, name)) for name in arrays[:3])
    assert sizes == ((11, 10, 14) if domain == "validation" else (21, 14, 0))
    # The parent's six methods stay loadable from the derived preparation.
    for method in METHODS:
        _, parent_config = load_model_inputs(derived, method)
        assert parent_config["method_id"] == method


def _profile(**changes):
    body = json.loads(json.dumps(PROFILE_CONFIG))
    for dotted, value in changes.items():
        *path, key = dotted.split(".")
        target = body
        for part in path:
            target = target[part]
        target[key] = value
    return body


@pytest.mark.parametrize(
    "contract,method,body,message",
    [
        ({"mask_seed": 1}, "nersi_profile", _profile(), "apart from study_id"),
        (
            {"preprocessing.alignment": {"velocity_m_per_s": 40000.0}},
            "nersi_profile",
            _profile(),
            "apart from study_id",
        ),
        ({}, "nersi_real", {"method_id": "nersi_real"}, "already a method"),
        # The study keeps the time-count side; the canvas side is applied at run time.
        ({}, "nersi_profile", _profile(**{"model.profile_shape": [24, 8]}), "complete traces"),
        ({}, "nersi_profile", _profile(**{"model.profile_shape": [16, 16]}), "complete traces"),
        (
            {},
            "nersi_profile",
            _profile(**{"training.supervised_traces_per_update": 3}),
            "exactly the budgeted traces",
        ),
        ({}, "nersi_profile", _profile(**{"training.max_steps": 5}), "match the budget"),
        ({}, "nersi_profile", _profile(**{"model.encoder_width": 400}), "matched model size"),
        ({}, "nersi_profile", _profile(model_seed=11), "model seed mismatch"),
        ({}, "nersi_profile", _profile(geometry_mode="grid"), "unsupported added method"),
        ({}, "nersi_profile", _profile(alignment={"velocity_m_per_s": VELOCITY}), "shared"),
    ],
)
def test_derivation_rejects_drifting_contracts_and_invalid_methods(
    runs, contract, method, body, message
):
    study_contract = json.loads(json.dumps(runs.contract))
    for dotted, value in contract.items():
        *path, key = dotted.split(".")
        target = study_contract
        for part in path:
            target = target[part]
        target[key] = value
    name = f"reject_{len(list((runs.root / 'studies').iterdir()))}"
    study = _write_study(runs.root, name, study_contract, {method: body})
    before = sorted((runs.root / "runs").rglob("*"))
    with pytest.raises(ValueError, match=message):
        derive_forge_preparation(runs.root, runs.parents["final"], study, [method])
    assert sorted((runs.root / "runs").rglob("*")) == before


def test_derivation_requires_a_directly_prepared_verified_parent(runs):
    before = sorted((runs.root / "runs").rglob("*"))
    for derived in runs.derived.values():
        with pytest.raises(ValueError, match="directly prepared"):
            derive_forge_preparation(runs.root, derived, runs.study, ["nersi_profile"])
    # A parent whose sealed artifacts changed is refused before anything is written.
    tampered = runs.root / "runs" / PARENT_STUDY / "tampered"
    shutil.copytree(runs.parents["final"], tampered)
    try:
        path = tampered / "input/observed_index.parquet"
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"\0")
        with pytest.raises(ValueError, match="hash mismatch"):
            derive_forge_preparation(runs.root, tampered, runs.study, ["nersi_profile"])
    finally:
        for item in tampered.rglob("*"):
            item.chmod(0o755 if item.is_dir() else 0o644)
        shutil.rmtree(tampered)
    assert sorted((runs.root / "runs").rglob("*")) == before


# --- profiles and exact-count sampling --------------------------------------------------


def test_profiles_take_unrounded_source_and_receiver_line_positions(runs):
    inputs, _ = load_model_inputs(runs.derived["validation"], "nersi_profile")
    geometry = inputs.geometry.copy()
    lines, points, receiver_lines, receiver_points = inputs.spatial_shape
    source = geometry.source_grid_line * points + geometry.source_grid_point
    receiver = geometry.receiver_grid_line * receiver_points + geometry.receiver_grid_point
    # One actual position per station, off its lattice line by a distinct amount.
    geometry["source_line_continuous"] += 0.01 * (source + 1)
    geometry["receiver_line_continuous"] += 0.001 * (receiver + 1) ** 2
    layout = nersi_profile_layout(geometry, inputs.spatial_shape)
    assert layout.coordinates.shape == (lines * points * receiver_lines, 3)
    assert layout.receiver_points == receiver_points
    profiles, columns = layout.locate(geometry.cell_id.to_numpy())
    np.testing.assert_array_equal(columns, geometry.receiver_grid_point)
    np.testing.assert_array_equal(profiles, source * receiver_lines + geometry.receiver_grid_line)
    # The receiver-line position is the mean over its stations, not over its traces:
    # the missing cell leaves receiver 5 with one trace fewer than the others.
    stations = geometry.drop_duplicates(["receiver_grid_line", "receiver_grid_point"])
    line_mean = stations.groupby("receiver_grid_line").receiver_line_continuous.mean()
    trace_mean = geometry.groupby("receiver_grid_line").receiver_line_continuous.mean()
    assert not np.allclose(line_mean, trace_mean)
    expected = np.column_stack(
        (
            geometry.source_line_continuous / (lines - 1),
            geometry.source_point_continuous / (points - 1),
            line_mean.loc[geometry.receiver_grid_line].to_numpy() / (receiver_lines - 1),
        )
    )
    np.testing.assert_allclose(layout.coordinates[profiles], expected, rtol=0, atol=1e-12)
    for axis, scale in enumerate((lines - 1, points - 1, receiver_lines - 1)):
        unscaled = layout.coordinates[:, axis] * scale
        assert not np.allclose(unscaled, np.round(unscaled)), axis
    # The receiver point is only the profile column: its actual position is never read.
    moved = geometry.assign(receiver_point_continuous=geometry.receiver_point_continuous + 0.4)
    np.testing.assert_array_equal(
        nersi_profile_layout(moved, inputs.spatial_shape).coordinates, layout.coordinates
    )


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda g: g.assign(cell_id=g.cell_id[::-1].to_numpy()), "C order"),
        (
            lambda g: g.assign(
                source_point_continuous=g.source_point_continuous + 0.1 * g.receiver_grid_point
            ),
            "one actual position",
        ),
        (
            lambda g: g.assign(
                receiver_line_continuous=g.receiver_line_continuous + 0.1 * g.source_grid_point
            ),
            "one actual position",
        ),
        (lambda g: g.loc[g.receiver_grid_line.eq(0)], "every source cell and receiver line"),
    ],
)
def test_profile_layout_rejects_inconsistent_geometry(runs, change, message):
    inputs, _ = load_model_inputs(runs.derived["final"], "nersi_profile")
    with pytest.raises(ValueError, match=message):
        nersi_profile_layout(change(inputs.geometry.copy()), inputs.spatial_shape)


class _Visit:
    """Replays fixed profile visiting orders; selection draws stay random."""

    def __init__(self, orders, seed):
        self.orders, self.rng = list(orders), np.random.default_rng(seed)

    def permutation(self, count):
        order = np.array(self.orders.pop(0))
        assert sorted(order) == list(range(count))
        return order

    def choice(self, *args, **kwargs):
        return self.rng.choice(*args, **kwargs)


def test_sampler_stacks_the_shortest_profile_prefix_then_keeps_exactly_k_traces():
    # Profiles 5, 7, 9 hold rows {0, 1}, {2} and {3, 4, 5}.
    sampler = ExactTraceProfileSampler(np.array([5, 5, 7, 9, 9, 9]), traces=3, seed=0)
    np.testing.assert_array_equal(sampler.eligible, [5, 7, 9])
    sampler.rng = _Visit([[1, 0, 2], [2, 1, 0], [0, 1, 2]], seed=0)
    assert sorted(sampler.sample()) == [0, 1, 2] and sampler.profiles_drawn == 2
    assert sorted(sampler.sample()) == [3, 4, 5] and sampler.profiles_drawn == 3
    # Profiles 5 and 7 pool exactly three rows; profile 9 is never visited.
    assert sorted(sampler.sample()) == [0, 1, 2] and sampler.profiles_drawn == 5
    sampler = ExactTraceProfileSampler(np.array([5, 5, 7, 9, 9, 9]), traces=2, seed=0)
    sampler.rng = _Visit([[2, 0, 1]] * 20, seed=1)
    drawn = [sampler.sample() for _ in range(20)]
    # A profile with more rows than needed keeps exactly two of them, uniformly.
    assert all(len(set(rows)) == 2 and set(rows) <= {3, 4, 5} for rows in drawn)
    assert {tuple(sorted(rows)) for rows in drawn} == {(3, 4), (3, 5), (4, 5)}
    assert sampler.profiles_drawn == 20
    with pytest.raises(ValueError, match="fewer observed traces"):
        ExactTraceProfileSampler(np.array([0, 1]), traces=3, seed=0)
    with pytest.raises(ValueError, match="fewer observed traces"):
        ExactTraceProfileSampler(np.array([0, 1]), traces=0, seed=0)


# --- fitting and prediction -------------------------------------------------------------


def test_each_update_fits_exactly_k_sampled_traces_at_their_profile_columns(runs, monkeypatch):
    inputs, config = load_model_inputs(runs.derived["final"], "nersi_profile")
    aligned, _ = align_model_inputs(inputs.model_space(), config["alignment"])
    layout = nersi_profile_layout(aligned.geometry, aligned.spatial_shape)
    coordinates = torch.from_numpy(layout.coordinates.astype(np.float32))
    profiles, columns = layout.locate(aligned.observed_cells)
    models, samples, updates = [], [], []
    build, sample, loss = (
        forge_mvp_methods.initialize_model,
        ExactTraceProfileSampler.sample,
        forge_mvp_methods.masked_trace_relative_mse,
    )

    def recorded_build(model_config):
        models.append(build(model_config))
        return models[-1]

    def recorded_sample(self):
        rows = sample(self)
        samples.append((rows.copy(), self.profiles_drawn))
        return rows

    def recorded_loss(prediction, target):
        # Before the optimizer step: decode every profile with the weights being fitted.
        with torch.no_grad():
            whole = models[-1](coordinates)[:, 0]
        updates.append((prediction.detach().clone(), target.clone(), whole))
        return loss(prediction, target)

    monkeypatch.setattr(forge_mvp_methods, "initialize_model", recorded_build)
    monkeypatch.setattr(ExactTraceProfileSampler, "sample", recorded_sample)
    monkeypatch.setattr(forge_mvp_methods, "masked_trace_relative_mse", recorded_loss)
    result = run_method(inputs, config, reporter=lambda message: None)
    assert len(models) == 1 and len(samples) == len(updates) == STEPS
    assert models[0].profile_shape == (profile_side(CANVAS), 8)
    drawn_before, decoded = 0, []
    for (rows, drawn), (prediction, target, whole) in zip(samples, updates, strict=True):
        assert len(rows) == len(np.unique(rows)) == TRACES
        assert prediction.shape == target.shape == (TRACES, CANVAS)
        np.testing.assert_array_equal(target.numpy(), aligned.observed_values[rows])
        expected = whole[torch.from_numpy(profiles[rows]), :CANVAS, torch.from_numpy(columns[rows])]
        torch.testing.assert_close(prediction, expected, rtol=1e-5, atol=1e-6)
        used = len(np.unique(profiles[rows]))
        assert drawn - drawn_before >= used >= 1
        drawn_before = drawn
        decoded.append(used)
    diagnostics = result.diagnostics
    assert diagnostics["mean_profiles_drawn_per_update"] == samples[-1][1] / STEPS
    assert diagnostics["mean_profiles_decoded_per_update"] == np.mean(decoded)
    assert diagnostics["alignment_canvas_sample_count"] == CANVAS
    assert diagnostics["final_training_loss_window_updates"] == STEPS
    assert [row["step"] for row in result.history] == list(range(1, STEPS + 1))


def test_predictions_decode_each_test_cell_at_its_profile_column(runs):
    inputs, config = load_model_inputs(runs.derived["validation"], "nersi_profile")
    result = run_method(inputs, config, reporter=lambda message: None)
    checkpoint = result.checkpoint
    assert checkpoint["model_config"]["profile_shape"] == [profile_side(CANVAS), 8]
    assert config["model"]["profile_shape"] == [profile_side(TIME_COUNT), 8]
    model = Nersi(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    aligned, canvas = align_model_inputs(inputs.model_space(), config["alignment"])
    layout = nersi_profile_layout(aligned.geometry, aligned.spatial_shape)
    profiles, columns = layout.locate(inputs.test_cells)
    with torch.no_grad():
        whole = model(torch.from_numpy(layout.coordinates.astype(np.float32)))[:, 0].numpy()
    on_canvas = np.ascontiguousarray(whole[profiles, :CANVAS, columns])
    expected = unalign_predictions(on_canvas, inputs.test_cells, canvas) * inputs.test_rms[:, None]
    assert result.prediction.shape == (len(inputs.test_cells), TIME_COUNT) == (10, TIME_COUNT)
    assert result.prediction.dtype == np.float32 and np.isfinite(result.prediction).all()
    np.testing.assert_allclose(result.prediction, expected, rtol=1e-5, atol=1e-7)
    # Profile batching of the prediction does not change any value.
    for batch in (1, 64):
        config["prediction"]["profile_batch_size"] = batch
        other = run_method(inputs, config, reporter=lambda message: None).prediction
        np.testing.assert_allclose(other, result.prediction, rtol=1e-6, atol=1e-8)


# --- matched tuning ---------------------------------------------------------------------


def _search(root, candidates, **options):
    path = root / f"search_{len(list(root.glob('search_*.yaml')))}.yaml"
    body = {
        "method_id": "nersi_profile",
        "device": "cpu",
        "torch_threads": 1,
        "candidates": candidates,
        **options,
    }
    path.write_text(yaml.safe_dump(body))
    return path


def test_matched_search_on_the_derived_validation_preparation(runs, monkeypatch):
    root, preparation = runs.root, runs.derived["validation"]
    batches = []
    loss = forge_mvp_methods.masked_trace_relative_mse

    def recorded_loss(prediction, target):
        batches.append((tuple(prediction.shape), tuple(target.shape)))
        return loss(prediction, target)

    monkeypatch.setattr(forge_mvp_methods, "masked_trace_relative_mse", recorded_loss)
    candidates = [
        {"name": "shadows", "training": {"ema_eval_decays": DECAYS, "ema_snapshot_updates": [2]}},
        {"name": "seeded", "model_seed": 7302, "model": {"encoder_width": 5}},
    ]
    output = tune_forge_matched(
        root, preparation, _search(root, candidates, target_relative_mse=1e9)
    )
    recorded = list(batches)
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert output.parent == root / "runs" / DERIVED_STUDY
    assert record["status"] == "completed" and record["method_id"] == "nersi_profile"
    assert record["selection_domain"] == "validation_holdout" and record["evaluate"] is True
    assert record["preparation"] == str(preparation.relative_to(root))
    assert record["preparation_hash"] == sha256_file(preparation / SEAL)
    assert record["matched_training"] == BUDGET
    assert record["alignment"] == _seal(preparation)["alignment"]
    # Every update of both candidates fits exactly the budgeted traces on the canvas.
    assert recorded == [((TRACES, CANVAS), (TRACES, CANVAS))] * (STEPS * len(candidates))
    inputs, _ = load_model_inputs(preparation, "nersi_profile")
    shape = (len(inputs.test_cells), TIME_COUNT)
    for name in ("shadows", "seeded"):
        prediction = np.load(output / name / "prediction.npy")
        assert prediction.shape == shape and prediction.dtype == np.float32
        assert np.isfinite(prediction).all()
    run = json.loads((output / "shadows/run_manifest.json").read_text())
    keys = [str(decay) for decay in DECAYS]
    assert sorted(run["ema_shadows"]) == sorted(keys) and run["ema_decay"] == "0.8"
    for key in keys:
        values = np.load(output / "shadows" / f"ema_{key}/prediction.npy")
        assert values.shape == shape and np.isfinite(values).all()
        assert np.isfinite(run["ema_shadows"][key]["primary_relative_mse"])
    np.testing.assert_array_equal(
        np.load(output / "shadows/prediction.npy"),
        np.load(output / "shadows/ema_0.8/prediction.npy"),
    )
    assert not np.array_equal(
        np.load(output / "shadows/ema_0.5/prediction.npy"),
        np.load(output / "shadows/ema_0.95/prediction.npy"),
    )
    for relative in ("update_2/ema_0.5.pt", "update_2/ema_0.8.pt", "update_2/ema_0.95.pt"):
        assert f"shadows/{relative}" in record["artifacts"]
    assert run["mean_profiles_drawn_per_update"] >= 1
    assert (
        1
        <= run["mean_profiles_decoded_per_update"]
        <= min(TRACES, run["mean_profiles_drawn_per_update"])
    )
    assert run["alignment_canvas_sample_count"] == CANVAS
    assert run["final_training_loss_window_updates"] == STEPS
    assert run["parameter_count"] == _seal(preparation)["parameter_counts"]["nersi_profile"]
    # The recorded config keeps the study's profile; the model was built on the canvas.
    resolved = yaml.safe_load((output / "shadows/resolved_config.yaml").read_text())
    assert resolved["model"]["profile_shape"] == [profile_side(TIME_COUNT), 8]
    checkpoint = torch.load(output / "shadows/final.pt", weights_only=False)
    assert checkpoint["model_config"]["profile_shape"] == [profile_side(CANVAS), 8]
    Nersi(**checkpoint["model_config"]).load_state_dict(checkpoint["model_state_dict"])
    seeded = yaml.safe_load((output / "seeded/resolved_config.yaml").read_text())
    assert seeded["model_seed"] == 7302 and seeded["model"]["encoder_width"] == 5
    seeded_run = json.loads((output / "seeded/run_manifest.json").read_text())
    assert seeded_run["parameter_count"] == model_parameter_count(seeded, CANVAS)
    # Each shadow equals a separately trained run of that EMA decay.
    _, base = load_model_inputs(preparation, "nersi_profile")
    for decay in DECAYS:
        config, _ = matched_candidate_config(
            base, {"name": "separate", "training": {"ema_decay": decay}}, "cpu", BUDGET, CANVAS, 11
        )
        expected = run_method(inputs, config, reporter=lambda message: None).prediction
        saved = np.load(output / "shadows" / f"ema_{decay}/prediction.npy")
        np.testing.assert_array_equal(saved, expected)


def test_unevaluated_final_run_on_the_derived_final_preparation(runs, monkeypatch):
    root, preparation = runs.root, runs.derived["final"]

    def forbidden(*args, **kwargs):
        raise AssertionError("an unevaluated final run must not read targets")

    monkeypatch.setattr("seis_interp.pipelines.tune_forge_matched.read_selected_traces", forbidden)
    candidates = [{"name": "final_7311", "model_seed": 7311}]
    output = tune_forge_matched(
        root, preparation, _search(root, candidates, purpose="final_evaluation", evaluate=False)
    )
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "completed" and record["independent_test_estimate"] is True
    prediction = np.load(output / "final_7311/prediction.npy")
    assert prediction.shape == (14, TIME_COUNT) and np.isfinite(prediction).all()
    assert not list(output.rglob("metrics*"))
    # The holdout study never scores its final test inside a search.
    with pytest.raises(ValueError, match="selects on its validation holdout"):
        tune_forge_matched(root, preparation, _search(root, candidates, target_relative_mse=1.0))


@pytest.mark.parametrize(
    "candidate,message",
    [
        ({"model": {"profile_shape": [profile_side(CANVAS), 8]}}, "fixed by the matched"),
        ({"model": {"profile_shape": [16, 16]}}, "fixed by the matched"),
        ({"training": {"supervised_traces_per_update": 3}}, "fixed by the matched"),
        ({"training": {"max_steps": STEPS + 1}}, "fixed by the matched"),
        ({"prediction": {"batch_size": 4}}, "unsupported prediction parameters"),
        ({"patches": {"shape": [9]}}, "may be tuned"),
        ({"model": {"encoder_width": 400}}, "matched model size"),
    ],
)
def test_matched_candidates_cannot_change_the_profile_or_the_trace_budget(runs, candidate, message):
    _, base = load_model_inputs(runs.derived["validation"], "nersi_profile")
    with pytest.raises(ValueError, match=message):
        matched_candidate_config(base, {"name": "bad", **candidate}, "cpu", BUDGET, CANVAS, 11)
    unchanged = {
        "name": "same",
        "model": {"profile_shape": [profile_side(TIME_COUNT), 8]},
        "training": {"supervised_traces_per_update": TRACES, "ema_eval_decays": [0.8, 0.9]},
        "prediction": {"profile_batch_size": 8},
    }
    config, count = matched_candidate_config(base, unchanged, "cpu", BUDGET, CANVAS, 11)
    assert count == model_parameter_count(base, CANVAS)
    assert config["model"]["profile_shape"] == base["model"]["profile_shape"]
    assert config["checkpoint_rule"] == "fixed_final_ema"


def test_reference_search_skips_only_the_parameter_band(runs):
    root, preparation = runs.root, runs.derived["validation"]
    _, base = load_model_inputs(preparation, "nersi_profile")
    wide = {"name": "wide", "model": {"encoder_width": 400}}
    with pytest.raises(ValueError, match="matched model size"):
        tune_forge_matched(root, preparation, _search(root, [wide], target_relative_mse=1e9))
    output = tune_forge_matched(
        root, preparation, _search(root, [wide], target_relative_mse=1e9, reference_only=True)
    )
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "completed" and record["reference_only"] is True
    run = json.loads((output / "wide/run_manifest.json").read_text())
    count = model_parameter_count(
        {**base, "model": {**base["model"], "encoder_width": 400}}, CANVAS
    )
    assert run["parameter_count"] == count > BUDGET["parameter_count_target"] * 1.95
    # The other matched conditions still hold for a reference search.
    steps = {"name": "longer", "training": {"max_steps": STEPS + 1}}
    with pytest.raises(ValueError, match="fixed by the matched"):
        tune_forge_matched(
            root, preparation, _search(root, [steps], target_relative_mse=1e9, reference_only=True)
        )
    with pytest.raises(ValueError, match="reference_only must be a boolean of a selection"):
        tune_forge_matched(
            root,
            runs.derived["final"],
            _search(root, [wide], purpose="final_evaluation", evaluate=False, reference_only=True),
        )


def test_time_stage_candidate_runs_on_the_canvas_and_without_alignment(runs):
    root, preparation = runs.root, runs.derived["validation"]
    inputs, base = load_model_inputs(preparation, "nersi_profile")
    stages = {"time_upsample_channels": [2, 2], "time_kernel_size": 1}
    output = tune_forge_matched(
        root,
        preparation,
        _search(root, [{"name": "deep", "model": stages}], target_relative_mse=1e9),
    )
    checkpoint = torch.load(output / "deep/final.pt", weights_only=False)
    assert checkpoint["model_config"]["profile_shape"] == [profile_side(CANVAS, 32), 8]
    assert np.load(output / "deep/prediction.npy").shape == (len(inputs.test_cells), TIME_COUNT)
    # Without alignment the profile time side is padded to the time multiple as well.
    unaligned = {**base, "model": {**base["model"], **stages}, "device": "cpu"}
    unaligned.pop("alignment")
    result = forge_mvp_methods.run_method(inputs, unaligned, reporter=lambda message: None)
    assert result.checkpoint["model_config"]["profile_shape"] == [profile_side(TIME_COUNT, 32), 8]
    assert result.prediction.shape == (len(inputs.test_cells), TIME_COUNT)


def test_regularizers_keep_the_trace_sampling_and_change_only_the_objective(runs, monkeypatch):
    inputs, base = load_model_inputs(runs.derived["validation"], "nersi_profile")
    sampled = []
    sample = ExactTraceProfileSampler.sample

    def recorded(self):
        rows = sample(self)
        sampled[-1].append(rows.copy())
        return rows

    monkeypatch.setattr(ExactTraceProfileSampler, "sample", recorded)
    results = {}
    for name, extra in (
        ("plain", {}),
        ("regularized", {"nuclear_norm_weight": 0.5, "coordinate_jitter_cells": 0.5}),
    ):
        sampled.append([])
        config = {**base, "training": {**base["training"], **extra}, "device": "cpu"}
        results[name] = forge_mvp_methods.run_method(inputs, config, reporter=lambda m: None)
    plain, regularized = results["plain"], results["regularized"]
    assert len(sampled[0]) == len(sampled[1]) == STEPS
    for first, second in zip(*sampled, strict=True):
        np.testing.assert_array_equal(first, second)
    assert all(
        "nuclear_norm" in row and np.isfinite(row["nuclear_norm"]) for row in regularized.history
    )
    assert not any("nuclear_norm" in row for row in plain.history)
    assert not np.array_equal(plain.prediction, regularized.prediction)
    assert np.isfinite(regularized.prediction).all()


def test_c3_canvas_rule_and_support_option_keep_predictions_bitwise(runs):
    inputs, base = load_model_inputs(runs.derived["validation"], "nersi_profile")
    base.pop("alignment")
    base["training"]["ema_decay"] = 0.999
    expected = run_method(inputs, base, reporter=lambda _: None)
    changed = deepcopy(base)
    changed["canvas_time_upsampling"] = True
    changed["training"]["loss_support_mask"] = True
    actual = run_method(inputs, changed, reporter=lambda _: None)
    np.testing.assert_array_equal(actual.prediction, expected.prediction)
    assert actual.history == expected.history
    for name, value in expected.checkpoint["model_state_dict"].items():
        assert torch.equal(value, actual.checkpoint["model_state_dict"][name])


def test_profile_prediction_heartbeats_preserve_primary_shadows_and_rng(runs, monkeypatch):
    from seis_interp.training import phase_progress
    from seis_interp.training.exact_resume import rng_state

    inputs, config = load_model_inputs(runs.derived["validation"], "nersi_profile")
    config["training"]["ema_eval_decays"] = DECAYS
    now = 0.0
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: now)
    original = forge_mvp_methods.initialize_model
    calls = []
    emitting = False

    def initialize(config):
        model = original(config)

        def advance(_, args):
            nonlocal now
            if emitting:
                now += 31
            calls.append(args[0].detach().numpy().tobytes())

        model.register_forward_pre_hook(advance)
        return model

    monkeypatch.setattr(forge_mvp_methods, "initialize_model", initialize)
    baseline = run_method(inputs, config, reporter=lambda _: None)
    before = rng_state()
    baseline_calls = calls.copy()
    calls.clear()
    emitting = True
    reports = []
    actual = run_method(inputs, config, reporter=reports.append)
    assert actual.prediction.tobytes() == baseline.prediction.tobytes()
    assert calls == baseline_calls
    for decay, expected in baseline.shadow_predictions.items():
        assert actual.shadow_predictions[decay].tobytes() == expected.tobytes()
    after = rng_state()
    assert before["python"] == after["python"]
    assert all(np.array_equal(a, b) for a, b in zip(before["numpy"], after["numpy"], strict=True))
    assert torch.equal(before["torch"], after["torch"])
    heartbeats = [m for m in reports if isinstance(m, phase_progress.ProgressHeartbeat)]
    for label in ("primary", "ema 0.5", "ema 0.95"):
        assert any(f"({label})" in message for message in heartbeats)


def test_profile_ema0999_is_bitwise_with_shadows_and_aligned_loss_support_is_exact(
    runs, monkeypatch
):
    inputs, base = load_model_inputs(runs.derived["validation"], "nersi_profile")
    base["training"]["ema_decay"] = 0.999
    baseline = run_method(inputs, base, reporter=lambda _: None)
    changed = deepcopy(base)
    changed["training"]["ema_eval_decays"] = [0.999, 0.9995, 0.9998]
    shadows = run_method(inputs, changed, reporter=lambda _: None)
    np.testing.assert_array_equal(shadows.prediction, baseline.prediction)
    for name, value in baseline.checkpoint["model_state_dict"].items():
        assert torch.equal(value, shadows.checkpoint["model_state_dict"][name])
    batches = []
    original = forge_mvp_methods.masked_trace_relative_mse

    def recorded(prediction, target):
        batches.append((prediction.shape, target.detach().cpu().numpy().copy()))
        return original(prediction, target)

    monkeypatch.setattr(forge_mvp_methods, "masked_trace_relative_mse", recorded)
    changed["training"]["loss_support_mask"] = True
    result = run_method(inputs, changed, reporter=lambda _: None)
    assert len(batches) == STEPS
    assert all(shape == (TRACES, TIME_COUNT) for shape, _ in batches)
    original_targets = (
        inputs.model_space().observed_values.astype(np.float64) / inputs.model_space().global_rms
    ).astype(np.float32)
    for _, values in batches:
        assert all(
            any(np.array_equal(row, original) for original in original_targets) for row in values
        )
    assert result.diagnostics["loss_support_samples_per_trace"] == TIME_COUNT
    assert result.prediction.shape == baseline.prediction.shape
    assert not np.array_equal(result.prediction, baseline.prediction)
