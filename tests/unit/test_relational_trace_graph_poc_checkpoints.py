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
    rng = torch.get_rng_state().clone()
    save_relational_trace_graph_poc_checkpoint(path, **arguments)
    assert torch.equal(torch.get_rng_state(), rng)
    original = path.read_bytes()
    restored = load_relational_trace_graph_poc_checkpoint(
        path, inputs_lock=arguments["inputs_lock"]
    )
    assert restored.metadata["loss"] == loss_name
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
