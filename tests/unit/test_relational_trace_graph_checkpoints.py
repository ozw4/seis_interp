"""Relational checkpoints need fixed geometry and time as well as weights."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
    save_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def _fixture(fusion="mean"):
    domain, training, amplitudes = make_relational_trace_domains()
    preprocessing = fit_trace_graph_preprocessing(
        training,
        amplitudes,
        position_scale_m=10,
        offset_scale_m=8,
        azimuth_min_offset_m=0.1,
    )
    torch.manual_seed(12)
    model = RelationalTraceGraphInterpolator(
        width=8, attention_width=5, relation_embedding_dim=3, relation_fusion=fusion
    )
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(std=0.15)
        if fusion == "learned_gate":
            for block in model.rounds:
                block.relation_gate[-1].weight.normal_(std=0.4)
    return model, domain, preprocessing, amplitudes


def _save(path, model, preprocessing, **overrides):
    arguments = {
        "model_config": model.constructor_config(),
        "state_dict": model.state_dict(),
        "preprocessing": preprocessing,
        "graph_settings": TraceGraphSettings(((1.0, 1.0),) * 4, neighbors_per_relation=2),
        "training_mask": {
            "kinds": ["random_trace", "random_whole_ffid"],
            "kind_probabilities": [0.5, 0.5],
            "missing_fractions": [0.5, 0.8],
        },
        "training_provenance": {
            "training_run": {"git_commit": "a" * 40, "git_worktree_dirty": False},
            "training_inputs_lock": preprocessing.fit_domain["inputs_lock"],
            "validation_inputs_lock": {"partition": "validation", "sha256": "b" * 64},
        },
        "training_random_seed": 31,
        "checkpoint_role": "best_validation",
        "global_step": 6,
        "selection_metrics": {"snr_db": None, "snr_status": "perfect_prediction"},
        **overrides,
    }
    save_relational_trace_graph_checkpoint(path, **arguments)


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
@pytest.mark.parametrize("role", ["best_validation", "final"])
@pytest.mark.parametrize("neighbor_search", ["brute_force", "exact_index"])
def test_round_trip_reproduces_nonzero_physical_predictions(
    tmp_path, fusion, role, neighbor_search
):
    model, domain, preprocessing, amplitudes = _fixture(fusion)
    path = tmp_path / "model.pt"
    _save(
        path,
        model,
        preprocessing,
        checkpoint_role=role,
        graph_settings=TraceGraphSettings(
            ((1.0, 1.0),) * 4, neighbors_per_relation=2, neighbor_search=neighbor_search
        ),
    )
    loaded = load_relational_trace_graph_checkpoint(path)
    assert loaded.graph_settings.neighbor_search == neighbor_search
    arguments = {"graph_settings": loaded.graph_settings, "amplitudes": amplitudes}
    original = predict_relational_trace_graph(model, domain, preprocessing, **arguments)
    restored = predict_relational_trace_graph(
        loaded.model, domain, loaded.preprocessing, **arguments
    )
    assert np.abs(original.prediction).max() > 1e-4
    np.testing.assert_array_equal(restored.prediction, original.prediction)
    assert loaded.model.constructor_config() == model.constructor_config()
    assert loaded.preprocessing == preprocessing
    assert loaded.checkpoint_role == role
    assert loaded.global_step == 6
    assert loaded.training_random_seed == 31
    assert (
        loaded.training_provenance["training_inputs_lock"]
        == preprocessing.fit_domain["inputs_lock"]
    )
    assert loaded.selection_metrics == {"snr_db": None, "snr_status": "perfect_prediction"}


def test_saved_best_state_is_cpu_snapshot_and_saving_preserves_rng(tmp_path):
    model, _, preprocessing, _ = _fixture()
    states = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    path = tmp_path / "best.pt"
    _save(path, model, preprocessing, state_dict=states)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
        for tensor in states.values():
            tensor.add_(5)
    payload = torch.load(path, weights_only=True)
    assert "schema_version" not in payload
    assert "optimizer" not in payload
    assert "model" not in payload
    assert payload["checkpoint_role"] == "best_validation"
    for name, tensor in payload["state_dict"].items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor + 5, states[name], rtol=0, atol=0)


def test_successful_save_replaces_best_checkpoint_and_removes_temporary_file(tmp_path):
    model, _, preprocessing, _ = _fixture()
    path = tmp_path / "best.pt"
    _save(path, model, preprocessing, global_step=6)
    with torch.no_grad():
        model.decoder.head[-1].weight.add_(1)
    _save(path, model, preprocessing, global_step=9)

    loaded = load_relational_trace_graph_checkpoint(path)
    assert loaded.global_step == 9
    for name, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("existing_checkpoint", [False, True])
def test_failed_save_preserves_existing_best_and_removes_partial_file(
    tmp_path, monkeypatch, existing_checkpoint
):
    model, _, preprocessing, _ = _fixture()
    path = tmp_path / "best.pt"
    if existing_checkpoint:
        _save(path, model, preprocessing, global_step=6)
        original_bytes = path.read_bytes()
    rng = torch.get_rng_state().clone()
    temporary_paths = []

    def fail_after_partial_write(payload, temporary_path):
        assert temporary_path.parent == path.parent
        assert temporary_path != path
        temporary_paths.append(temporary_path)
        temporary_path.write_bytes(b"partial checkpoint")
        raise OSError("checkpoint write interrupted")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="checkpoint write interrupted"):
        _save(path, model, preprocessing, global_step=9)

    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    if existing_checkpoint:
        assert path.read_bytes() == original_bytes
        loaded = load_relational_trace_graph_checkpoint(path)
        assert loaded.global_step == 6
        for name, tensor in loaded.model.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
        assert list(tmp_path.iterdir()) == [path]
    else:
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("model_type", "trace_graph_interpolator", "model_type"),
        ("relation_names", ["receiver", "source", "cmp", "offset_azimuth"], "relation_names"),
        ("node_feature_names", ["wrong"], "node_feature_names"),
        ("edge_feature_names", ["wrong"], "edge_feature_names"),
        ("checkpoint_role", "best", "checkpoint_role"),
        ("global_step", -1, "global_step"),
        ("training_random_seed", True, "training_random_seed"),
    ],
)
def test_load_rejects_wrong_contract_metadata(tmp_path, field, value, error):
    model, _, preprocessing, _ = _fixture()
    path = tmp_path / "bad.pt"
    _save(path, model, preprocessing)
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=error):
        load_relational_trace_graph_checkpoint(path)


@pytest.mark.parametrize(
    "section,field,value,error",
    [
        ("graph_settings", "relation_scales_m", [[1.0, 0.0]] * 4, "relation_scales_m"),
        ("graph_settings", "radius", -1, "radius"),
        ("graph_settings", "neighbors_per_relation", 0, "neighbors_per_relation"),
        ("preprocessing", "position_scale_m", 0, "position_scale_m"),
        ("preprocessing", "offset_scale_m", -1, "offset_scale_m"),
        ("preprocessing", "amplitude_scale", 0, "amplitude_scale"),
        ("model_config", "width", 16, "state_dict"),
        ("model_config", "temporal_dilations", [1], "temporal_dilations"),
        ("model_config", "relation_fusion", "union", "model_config"),
        ("model_config", "max_edge_time_shift_samples", -1, "max_edge_time_shift_samples"),
        ("model_config", "max_edge_time_shift_samples", True, "max_edge_time_shift_samples"),
        ("model_config", "max_edge_time_shift_samples", 1.5, "max_edge_time_shift_samples"),
        ("time", "sample_count", 4, "sample_count"),
        ("time", "time_downsample_factor", 3, "time_downsample_factor"),
        ("time", "time_s", [1, 2, 3, 4, 5], "time_s"),
        ("training_mask", "kind_probabilities", [0.4, 0.5], "kind_probabilities"),
        ("training_mask", "missing_fractions", [0, 0.8], "missing_fractions"),
    ],
)
def test_load_rejects_geometry_model_and_time_inconsistency(tmp_path, section, field, value, error):
    model, _, preprocessing, _ = _fixture()
    path = tmp_path / "bad.pt"
    _save(path, model, preprocessing)
    payload = torch.load(path, weights_only=True)
    payload[section][field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=error):
        load_relational_trace_graph_checkpoint(path)


def test_weights_without_required_config_are_insufficient(tmp_path):
    model, _, preprocessing, _ = _fixture()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="model_type"):
        load_relational_trace_graph_checkpoint(path)
    _save(path, model, preprocessing)
    payload = torch.load(path, weights_only=True)
    del payload["model_config"]["temporal_dilations"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="all constructor fields"):
        load_relational_trace_graph_checkpoint(path)


def test_missing_weights_cannot_silently_load(tmp_path):
    model, _, preprocessing, _ = _fixture()
    path = tmp_path / "bad.pt"
    _save(path, model, preprocessing)
    payload = torch.load(path, weights_only=True)
    del payload["state_dict"][next(iter(payload["state_dict"]))]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="state_dict does not match"):
        load_relational_trace_graph_checkpoint(path)


def test_save_rejects_training_sample_range_inconsistent_with_time_grid(tmp_path):
    model, _, preprocessing, _ = _fixture()
    preprocessing = replace(
        preprocessing, fit_domain={**preprocessing.fit_domain, "time_samples": [1, 5]}
    )
    with pytest.raises(ValueError, match="time_samples"):
        _save(tmp_path / "bad.pt", model, preprocessing)
    assert not (tmp_path / "bad.pt").exists()


def test_mask_probability_tolerance_matches_training_config(tmp_path):
    model, _, preprocessing, _ = _fixture()
    path = tmp_path / "model.pt"
    mask = {
        "kinds": ["random_trace"],
        "kind_probabilities": [1.0 + 5e-9],
        "missing_fractions": [0.5],
    }
    _save(path, model, preprocessing, training_mask=mask)
    assert load_relational_trace_graph_checkpoint(path).training_mask == mask
