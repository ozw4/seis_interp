"""PoC snapshots reject mismatched inputs and incomplete inference contracts."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import build_poc_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
    load_relational_trace_graph_poc_checkpoint,
    save_relational_trace_graph_poc_checkpoint,
)
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


def _arguments():
    domain, _, _ = make_relational_trace_domains()
    lock = {
        "case_id": "case",
        "volume_id": "volume",
        "benchmark_volume": {"files": {"volume_index.parquet": {"sha256": "a" * 64}}},
    }
    domain = replace(domain, inputs_lock=lock)
    preprocessing = build_poc_trace_graph_preprocessing(
        domain,
        amplitude_scale=2.0,
        position_scale_m=10,
        offset_scale_m=8,
        azimuth_min_offset_m=0.1,
    )
    model = RelationalTraceGraphInterpolator(width=8, attention_width=5, relation_embedding_dim=3)
    return {
        "model_config": model.constructor_config(),
        "state_dict": model.state_dict(),
        "preprocessing": preprocessing,
        "graph_settings": TraceGraphSettings(((1.0, 1.0),) * 4, neighbors_per_relation=2),
        "inputs_lock": lock,
        "metadata": {
            "method": "relational_trace_graph",
            "training_domain": "O_with_inner_pseudo_mask",
            "normalization": {"type": "global_rms", "source": "O_only", "scale": 2.0},
            "loss": "masked_trace_relative_mse",
            "checkpoint_role": "final",
            "output_amplitude_domain": "physical",
            "case_id": "case",
            "volume_id": "volume",
            "model_initialization_seed": 7,
            "episode_seed": 201,
            "steps_completed": 3,
            "inner_mask_fraction": 0.5,
        },
    }


def test_poc_fourier_checkpoint_round_trip_and_fixed_frequency_validation(tmp_path):
    arguments = _arguments()
    model = RelationalTraceGraphInterpolator(**arguments["model_config"], node_fourier_components=4)
    arguments.update(model_config=model.constructor_config(), state_dict=model.state_dict())
    path = tmp_path / "final.pt"
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    restored = load_relational_trace_graph_poc_checkpoint(
        path, inputs_lock=arguments["inputs_lock"]
    )
    assert restored.model.constructor_config() == model.constructor_config()
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])
    payload = torch.load(path, weights_only=True)
    payload["model_state_dict"]["node_fourier_mapping.frequencies"][0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Fourier frequencies"):
        load_relational_trace_graph_poc_checkpoint(path, inputs_lock=arguments["inputs_lock"])


def test_poc_spectral_checkpoint_round_trip(tmp_path):
    arguments = _arguments()
    model = RelationalTraceGraphInterpolator(**arguments["model_config"], spectral_input_block=True)
    arguments.update(model_config=model.constructor_config(), state_dict=model.state_dict())
    path = tmp_path / "final.pt"
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    restored = load_relational_trace_graph_poc_checkpoint(
        path, inputs_lock=arguments["inputs_lock"]
    )
    assert restored.model.constructor_config() == model.constructor_config()
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_schedule_checkpoint_round_trip_and_validation(tmp_path):
    arguments = _arguments()
    schedule = dict(kind="constant_then_cosine", hold_steps=1, minimum_learning_rate=0.00003)
    arguments["metadata"].update(learning_rate=0.001, learning_rate_schedule=schedule)
    path = tmp_path / "final.pt"
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    restored = load_relational_trace_graph_poc_checkpoint(
        path, inputs_lock=arguments["inputs_lock"]
    )
    assert restored.metadata["learning_rate_schedule"] == schedule
    payload = torch.load(path, weights_only=True)
    payload["learning_rate_schedule"]["hold_steps"] = 3
    torch.save(payload, path)
    with pytest.raises(ValueError, match="hold_steps"):
        load_relational_trace_graph_poc_checkpoint(path, inputs_lock=arguments["inputs_lock"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("decay", None),
        ("decay", 1),
        ("updates", 0),
        ("updates", 4),
        ("updates", True),
        ("initialization", "unknown"),
    ],
)
def test_ema_checkpoint_metadata_is_validated(tmp_path, key, value):
    arguments = _arguments()
    arguments["metadata"].update(
        weight_source="ema",
        ema={"decay": 0.999, "updates": 3, "initialization": "first_post_update_weights"},
    )
    path = tmp_path / "ema.pt"
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    payload = torch.load(path, weights_only=True)
    payload["ema"][key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_relational_trace_graph_poc_checkpoint(path, inputs_lock=arguments["inputs_lock"])


@pytest.mark.parametrize(
    "keys,value,error",
    [
        (("model_config", "width"), 16, "state_dict"),
        (("model_state_dict",), {}, "state_dict"),
        (("node_feature_names",), ["wrong"], "node_feature_names"),
        (("edge_feature_names",), ["wrong"], "edge_feature_names"),
        (("relation_names",), ["wrong"], "relation_names"),
        (("graph_settings", "radius"), -1, "radius"),
        (("graph_settings",), {}, "graph_settings"),
        (("time", "sample_count"), 999, "sample_count"),
        (("time", "time_s"), [0, 1], "time_s"),
        (("preprocessing", "amplitude_scale"), 0, "amplitude_scale"),
        (("preprocessing", "midpoint_origin_m"), [999, 999], "midpoint bounds"),
        (("preprocessing", "fit_domain", "amplitude_source"), "T", "fit_domain"),
        (("preprocessing", "fit_domain", "inputs_lock"), {}, "fit_domain"),
        (("normalization", "scale"), 3.0, "normalization"),
        (("checkpoint_role",), "best_validation", "checkpoint_role"),
        (("loss",), "unknown", "loss"),
        (("cudnn_benchmark",), "false", "cudnn_benchmark"),
        (("model_initialization_seed",), True, "model_initialization_seed"),
        (("episode_seed",), -1, "episode_seed"),
        (("steps_completed",), 0, "steps_completed"),
        (("inner_mask_fraction",), 1.0, "inner_mask_fraction"),
    ],
)
def test_poc_loader_rejects_corrupted_contract(tmp_path, keys, value, error):
    arguments = _arguments()
    path = tmp_path / "final.pt"
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    payload = torch.load(path, weights_only=True)
    section = payload
    for key in keys[:-1]:
        section = section[key]
    section[keys[-1]] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=error):
        load_relational_trace_graph_poc_checkpoint(path, inputs_lock=arguments["inputs_lock"])


def test_poc_loader_compares_nested_volume_hash_before_model_construction(tmp_path, monkeypatch):
    arguments = _arguments()
    path = tmp_path / "final.pt"
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    changed = deepcopy(arguments["inputs_lock"])
    changed["benchmark_volume"]["files"]["volume_index.parquet"]["sha256"] = "b" * 64
    monkeypatch.setattr(
        "seis_interp.training.relational_trace_graph_checkpoints.RelationalTraceGraphInterpolator",
        lambda **kwargs: pytest.fail("mismatched lock must fail before model construction"),
    )
    with pytest.raises(ValueError, match="inputs_lock"):
        load_relational_trace_graph_poc_checkpoint(path, inputs_lock=changed)


@pytest.mark.parametrize("loss_name", ["masked_trace_mse", "masked_trace_relative_mse"])
def test_poc_atomic_snapshot_preserves_rng_and_existing_file_on_failure(
    tmp_path, monkeypatch, loss_name
):
    arguments = _arguments()
    path = tmp_path / "final.pt"
    arguments["metadata"]["loss"] = loss_name
    arguments["metadata"]["cudnn_benchmark"] = False
    rng = torch.get_rng_state().clone()
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    assert torch.equal(torch.get_rng_state(), rng)
    original = path.read_bytes()
    restored = load_relational_trace_graph_poc_checkpoint(
        path, inputs_lock=arguments["inputs_lock"]
    )
    assert restored.metadata["loss"] == loss_name
    assert restored.metadata["cudnn_benchmark"] is False
    assert restored.preprocessing == arguments["preprocessing"]
    assert restored.graph_settings == arguments["graph_settings"]
    for key, value in arguments["state_dict"].items():
        assert torch.equal(value, restored.model.state_dict()[key])
    with pytest.raises(ValueError, match="state_dict"):
        load_relational_trace_graph_checkpoint(path)

    def fail(payload, temporary_path):
        temporary_path.write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(torch, "save", fail)
    rng = torch.get_rng_state().clone()
    with pytest.raises(OSError, match="interrupted"):
        save_relational_trace_graph_poc_checkpoint(path, **arguments)
    assert torch.equal(torch.get_rng_state(), rng)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("key", ["attention_pooling", "relation_gate_pooling"])
def test_poc_rms_checkpoint_restores_pooling_and_weights(tmp_path, key):
    arguments = _arguments()
    config = {**arguments["model_config"], "relation_fusion": "learned_gate", key: "rms"}
    model = RelationalTraceGraphInterpolator(**config)
    arguments.update(model_config=model.constructor_config(), state_dict=model.state_dict())
    path = tmp_path / "ema.pt"
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    restored = load_relational_trace_graph_poc_checkpoint(
        path, inputs_lock=arguments["inputs_lock"]
    )
    assert restored.model.constructor_config() == model.constructor_config()
    for block in restored.model.rounds:
        assert getattr(block, key) == "rms"
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
