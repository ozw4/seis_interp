"""GNN PoC optimizer, mask/label binding, and physical dense scatter contracts."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.data.c3_poc_trace_graph import build_c3_poc_trace_graph_training_data
from seis_interp.data.c3_trace_graph_prediction import scatter_c3_trace_graph_prediction
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config
from seis_interp.training import relational_trace_graph_poc_trainer as trainer
from seis_interp.training.c3_poc_trace_graph_episodes import PocTraceGraphEpisodeGenerator
from tests.fixtures.c3_poc_trace_graph import prepare_poc_trace_graph_inputs


@pytest.mark.parametrize("loss_name", ["masked_trace_mse", "masked_trace_relative_mse"])
@pytest.mark.parametrize("no_context", [False, True])
def test_shared_loss_receives_exact_hidden_rows_and_every_optimizer_step(
    tmp_path, monkeypatch, no_context, loss_name
):
    inputs, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config["training"]["loss"] = loss_name
    settings = validate_relational_trace_graph_poc_config(config)
    training = build_c3_poc_trace_graph_training_data(
        inputs, amplitude_scale=7.0, **settings.geometry
    )
    options = dict(settings.training)
    options.pop("device")
    options.pop("model_initialization_seed")
    options["random_seed"] = options.pop("episode_seed")
    options["max_steps"] = 5
    episode_generator = PocTraceGraphEpisodeGenerator(
        training,
        random_seed=options["random_seed"],
        missing_fraction=options["inner_mask_fraction"],
    )
    expected = []
    positions = {int(value): i for i, value in enumerate(training.domain.trace_ids)}
    while len(expected) < options["max_steps"]:
        episode = episode_generator.next_episode()
        for ids in episode.query_batches(options["query_batch_size"]):
            rows = [positions[int(i)] for i in ids]
            expected.append(
                (training.observed_amplitudes[rows].astype(np.float64) / 7).astype(np.float32)
            )
    original_loss = trainer.masked_trace_loss
    calls = []

    def loss(prediction, target, *, loss_name):
        assert loss_name == options["loss"]
        assert prediction.shape == target.shape == expected[len(calls)].shape
        np.testing.assert_array_equal(target.detach().numpy(), expected[len(calls)])
        assert prediction.requires_grad
        result = original_loss(prediction, target, loss_name=loss_name)
        calls.append(float(result.detach()))
        return result

    monkeypatch.setattr(trainer, "masked_trace_loss", loss)
    torch.manual_seed(4)
    model = RelationalTraceGraphInterpolator(**settings.model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    graph = replace(settings.graph, radius=1e-12) if no_context else settings.graph
    result = trainer.train_relational_trace_graph_poc(
        model, training, graph_settings=graph, **options
    )
    assert result.steps_completed == len(calls) == 5
    assert [row["loss"] for row in result.history] == calls
    assert result.query_count == sum(row["query_count"] for row in result.history)
    assert result.episodes_started > 1
    assert not hasattr(result, "best_state_dict")
    assert not hasattr(result, "validation_history")
    if no_context:
        assert result.no_context_query_count == result.query_count
        if loss_name == "masked_trace_relative_mse":
            np.testing.assert_allclose(calls, 1.0)
        else:
            np.testing.assert_allclose(
                calls, [np.mean(row.astype(np.float64) ** 2) for row in expected[:5]]
            )
    else:
        assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_dense_scatter_preserves_query_id_mapping_and_does_not_rescale(tmp_path):
    inputs, _, _ = prepare_poc_trace_graph_inputs(tmp_path)
    volume = inputs.observed_volume
    target = volume.evaluation_target_trace_mask
    ids = volume.array_rows[target][::-1].copy()
    physical = np.arange(len(ids) * len(volume.time_s), dtype=np.float32).reshape(len(ids), -1)
    dense, coverage = scatter_c3_trace_graph_prediction(inputs, ids, physical)
    np.testing.assert_array_equal(dense[:, target].T, physical[::-1])
    np.testing.assert_array_equal(
        dense[:, volume.observed_trace_mask], volume.values[:, volume.observed_trace_mask]
    )
    np.testing.assert_array_equal(coverage, target)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "amplitude_mode", "observed_trace_rms"),
        ("training", "optimizer", "sgd"),
        ("training", "validation_interval", 1),
        ("training", "max_steps", 0),
        ("training", "episode_seed", -1),
        ("training", "model_initialization_seed", True),
        ("training", "gradient_clip_norm", -1),
        ("prediction", "query_batch_size", 0),
        ("geometry_features", "position_scale_m", 0),
    ],
)
def test_poc_config_rejects_other_protocols_and_invalid_numbers(tmp_path, section, key, value):
    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_relational_trace_graph_poc_config(config)
