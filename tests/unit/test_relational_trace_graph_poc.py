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
    expected_query_ids = []
    positions = {int(value): i for i, value in enumerate(training.domain.trace_ids)}
    while len(expected) < options["max_steps"]:
        episode = episode_generator.next_episode()
        for ids in episode.query_batches(options["query_batch_size"]):
            expected_query_ids.append(ids.copy())
            rows = [positions[int(i)] for i in ids]
            expected.append(
                (training.observed_amplitudes[rows].astype(np.float64) / 7).astype(np.float32)
            )
    original_loss = trainer.masked_trace_loss
    original_build = trainer.FixedTraceGraphSubgraphBuilder.build
    plans = []

    def build(builder, *args, **kwargs):
        plan = original_build(builder, *args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(trainer.FixedTraceGraphSubgraphBuilder, "build", build)
    calls = []

    def loss(prediction, target, *, loss_name):
        assert loss_name == options["loss"]
        assert prediction.shape == target.shape == expected[len(calls)].shape
        np.testing.assert_array_equal(target.detach().numpy(), expected[len(calls)])
        plan = plans[-1]
        np.testing.assert_array_equal(
            plan.trace_ids[plan.query_indices], expected_query_ids[len(calls)]
        )
        assert not np.any(plan.observed_mask[plan.query_indices])
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
    for row, plan in zip(result.history, plans, strict=True):
        assert row["subgraph_node_count"] == len(plan.trace_ids)
        assert row["subgraph_node_count"] == row["query_count"] + row["subgraph_support_node_count"]
        assert row["subgraph_support_node_count"] == plan.diagnostics["support_node_count"]
        assert (
            row["subgraph_edge_count"]
            == plan.diagnostics["typed_edge_count"]
            == plan.edge_index.shape[1]
        )
        assert row["subgraph_max_depth"] == plan.diagnostics["max_depth"]
        for key in ("seconds", "batch_preparation_seconds", "optimization_seconds"):
            assert np.isfinite(row[key]) and row[key] >= 0
        for key in ("subgraph_support_node_count", "subgraph_edge_count", "subgraph_max_depth"):
            assert row[key] >= 0
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


@pytest.mark.parametrize("neighbor_search", ["exact_index", "brute_force"])
def test_training_reuses_one_static_index_with_fresh_episode_builders(
    tmp_path, monkeypatch, neighbor_search
):
    inputs, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    settings = validate_relational_trace_graph_poc_config(config)
    training = build_c3_poc_trace_graph_training_data(
        inputs, amplitude_scale=7.0, **settings.geometry
    )
    options = dict(settings.training)
    options.pop("device")
    options.pop("model_initialization_seed")
    options["random_seed"] = options.pop("episode_seed")
    options["max_steps"] = 5
    indices, builders, masks = [], [], []
    original_index = trainer.TraceGraphSpatialIndex
    original_builder = trainer.FixedTraceGraphSubgraphBuilder.from_spatial_index
    original_build = trainer.FixedTraceGraphSubgraphBuilder.build

    def create_index(*args, **kwargs):
        index = original_index(*args, **kwargs)
        indices.append(index)
        return index

    def create_builder(index, observed_mask):
        assert index is indices[0]
        builder = original_builder(index, observed_mask)
        assert builder.observed_neighbor_cache_info()["cached_sender_count"] == 0
        builders.append(builder)
        masks.append(observed_mask.copy())
        return builder

    def build(builder, geometry, query_ids, *, rounds):
        plan = original_build(builder, geometry, query_ids, rounds=rounds)
        senders = plan.trace_ids[plan.edge_index[0]]
        assert not np.any(np.isin(senders, query_ids))
        assert np.all(np.isin(senders, training.domain.trace_ids[masks[-1]]))
        return plan

    monkeypatch.setattr(trainer, "TraceGraphSpatialIndex", create_index)
    monkeypatch.setattr(
        trainer.FixedTraceGraphSubgraphBuilder, "from_spatial_index", create_builder
    )
    monkeypatch.setattr(trainer.FixedTraceGraphSubgraphBuilder, "build", build)
    torch.manual_seed(4)
    model = RelationalTraceGraphInterpolator(
        **{**settings.model, "message_passing_rounds": 2, "temporal_dilations": (1, 1)}
    )
    result = trainer.train_relational_trace_graph_poc(
        model,
        training,
        graph_settings=replace(settings.graph, neighbor_search=neighbor_search),
        **options,
    )
    assert result.steps_completed == 5
    assert result.episodes_started > 1
    assert all(np.isfinite(row["loss"]) for row in result.history)
    if neighbor_search == "exact_index":
        assert len(indices) == 1
        assert (
            len(builders) == len({id(builder) for builder in builders}) == result.episodes_started
        )
        assert any(not np.array_equal(masks[0], mask) for mask in masks[1:])
    else:
        assert indices == builders == []


@pytest.mark.parametrize("mode", ["off", "fp16", "bf16"])
def test_poc_precision_defaults_and_accepted_modes(tmp_path, mode):
    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    assert validate_relational_trace_graph_poc_config(config).training["mixed_precision"] == "off"
    config["training"]["mixed_precision"] = mode
    assert validate_relational_trace_graph_poc_config(config).training["mixed_precision"] == mode


@pytest.mark.parametrize("mode", [True, False, None, 0, "auto", "float16", [], {}])
def test_poc_rejects_invalid_precision(tmp_path, mode):
    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config["training"]["mixed_precision"] = mode
    with pytest.raises(ValueError, match="mixed_precision"):
        validate_relational_trace_graph_poc_config(config)


def _precision_training_case(tmp_path):
    inputs, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    settings = validate_relational_trace_graph_poc_config(config)
    training = build_c3_poc_trace_graph_training_data(
        inputs, amplitude_scale=7.0, **settings.geometry
    )
    options = dict(settings.training)
    options.pop("device")
    options.pop("model_initialization_seed")
    options.pop("mixed_precision")
    options["random_seed"] = options.pop("episode_seed")
    options["max_steps"] = 2
    torch.manual_seed(4)
    return RelationalTraceGraphInterpolator(**settings.model), training, settings.graph, options


@pytest.mark.parametrize("mode", ["fp16", "bf16"])
def test_cpu_rejects_amp_before_training(tmp_path, mode):
    model, training, graph, options = _precision_training_case(tmp_path)
    with pytest.raises(ValueError, match="requires an available CUDA device"):
        trainer.train_relational_trace_graph_poc(
            model, training, graph_settings=graph, mixed_precision=mode, **options
        )


def test_missing_and_explicit_off_have_identical_history_state_and_rng(tmp_path, monkeypatch):
    model, training, graph, options = _precision_training_case(tmp_path)
    second = RelationalTraceGraphInterpolator(**model.constructor_config())
    second.load_state_dict(model.state_dict())
    initial_rng = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: pytest.fail("off must bypass autocast"))
    first_result = trainer.train_relational_trace_graph_poc(
        model, training, graph_settings=graph, **options
    )
    first_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    second_result = trainer.train_relational_trace_graph_poc(
        second, training, graph_settings=graph, mixed_precision="off", **options
    )
    assert torch.equal(first_rng, torch.get_rng_state())
    for first_row, second_row in zip(first_result.history, second_result.history, strict=True):
        assert {k: v for k, v in first_row.items() if not k.endswith("seconds")} == {
            k: v for k, v in second_row.items() if not k.endswith("seconds")
        }
    for key, value in model.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP")
@pytest.mark.parametrize("mode", ["fp16", "bf16"])
def test_cuda_amp_finite_training_and_unscale_before_clipping(tmp_path, monkeypatch, mode):
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    model, training, graph, options = _precision_training_case(tmp_path)
    events = []
    original_unscale = torch.cuda.amp.GradScaler.unscale_
    original_clip = torch.nn.utils.clip_grad_norm_
    original_loss = trainer.masked_trace_loss

    def unscale(scaler, optimizer):
        result = original_unscale(scaler, optimizer)
        events.append("unscale")
        return result

    def clip(*args, **kwargs):
        if mode == "fp16":
            assert events[-1] == "unscale"
        events.append("clip")
        return original_clip(*args, **kwargs)

    def loss(prediction, target, **kwargs):
        assert not torch.is_autocast_enabled()
        assert prediction.dtype == target.dtype == torch.float32
        objective = original_loss(prediction, target, **kwargs)
        assert objective.dtype == torch.float32
        return objective

    monkeypatch.setattr(torch.cuda.amp.GradScaler, "unscale_", unscale)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(trainer, "masked_trace_loss", loss)
    result = trainer.train_relational_trace_graph_poc(
        model, training, graph_settings=graph, mixed_precision=mode, device="cuda", **options
    )
    assert result.steps_completed == 2
    assert all(np.isfinite(row["loss"]) for row in result.history)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert events == (["unscale", "clip"] * 2 if mode == "fp16" else ["clip"] * 2)


def test_sampling_config_is_absent_by_default_and_retains_valid_mapping(tmp_path):
    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    assert "edge_sampling" not in validate_relational_trace_graph_poc_config(config).training
    config["training"]["edge_sampling"] = {"seed": 0, "fanout_per_relation": 1}
    assert validate_relational_trace_graph_poc_config(config).training["edge_sampling"] == {
        "seed": 0,
        "fanout_per_relation": 1,
    }


@pytest.mark.parametrize(
    "sampling",
    [
        None,
        True,
        {},
        {"seed": 1},
        {"fanout_per_relation": 1},
        {"seed": 1, "fanout_per_relation": 1, "enabled": True},
        {"seed": True, "fanout_per_relation": 1},
        {"seed": -1, "fanout_per_relation": 1},
        {"seed": 1, "fanout_per_relation": True},
        {"seed": 1, "fanout_per_relation": 0},
        {"seed": 1, "fanout_per_relation": 1.5},
        {"seed": 1, "fanout_per_relation": 3},
    ],
)
def test_sampling_config_rejects_invalid_mapping(tmp_path, sampling):
    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config["training"]["edge_sampling"] = sampling
    with pytest.raises(ValueError, match="edge_sampling"):
        validate_relational_trace_graph_poc_config(config)


def test_sampling_rejects_brute_force_and_single_4d(tmp_path):
    from seis_interp.training.trace_graph_sampling import validate_trace_graph_edge_sampling

    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config["training"]["edge_sampling"] = {"seed": 1, "fanout_per_relation": 1}
    config["graph"]["neighbor_search"] = "brute_force"
    with pytest.raises(ValueError, match="exact_index"):
        validate_relational_trace_graph_poc_config(config)
    graph = validate_relational_trace_graph_poc_config(
        {
            **config,
            "training": {k: v for k, v in config["training"].items() if k != "edge_sampling"},
        }
    ).graph
    with pytest.raises(ValueError, match="multi_relation"):
        validate_trace_graph_edge_sampling(
            {"seed": 1, "fanout_per_relation": 1},
            replace(
                graph,
                topology="single_4d",
                common_distance_scales_m=(1, 1),
                neighbor_search="exact_index",
            ),
        )


def test_sampling_seed_reproduces_graphs_losses_without_changing_queries(tmp_path, monkeypatch):
    model, training, graph, options = _precision_training_case(tmp_path)
    options["max_steps"] = 5
    options["query_batch_size"] = 1
    initial_state = {key: value.clone() for key, value in model.state_dict().items()}
    original_build = trainer.FixedTraceGraphSubgraphBuilder.build
    captured = []

    def build(builder, geometry, query_ids, **kwargs):
        plan = original_build(builder, geometry, query_ids, **kwargs)
        captured.append(
            (query_ids.copy(), plan.trace_ids[plan.edge_index].copy(), plan.edge_type.copy())
        )
        return plan

    monkeypatch.setattr(trainer.FixedTraceGraphSubgraphBuilder, "build", build)
    runs = []
    histories = []
    for seed in (47, 47, 48, None):
        model.load_state_dict(initial_state)
        captured.clear()
        result = trainer.train_relational_trace_graph_poc(
            model,
            training,
            graph_settings=graph,
            edge_sampling=None if seed is None else {"seed": seed, "fanout_per_relation": 1},
            **options,
        )
        runs.append((list(captured), [row["loss"] for row in result.history]))
        histories.append(result.history)
    for run, _ in runs[1:]:
        for first, other in zip(runs[0][0], run, strict=True):
            np.testing.assert_array_equal(first[0], other[0])
    assert runs[0][1] == runs[1][1]
    for first, repeated in zip(runs[0][0], runs[1][0], strict=True):
        for array, repeat in zip(first, repeated, strict=True):
            np.testing.assert_array_equal(array, repeat)
    assert any(
        not np.array_equal(first[1], other[1])
        for first, other in zip(runs[0][0], runs[2][0], strict=True)
    )
    assert sum(plan[1].shape[1] for plan in runs[0][0]) < sum(
        plan[1].shape[1] for plan in runs[3][0]
    )

    for key in ("subgraph_node_count", "subgraph_edge_count"):
        assert sum(row[key] for row in histories[0]) < sum(row[key] for row in histories[3])
