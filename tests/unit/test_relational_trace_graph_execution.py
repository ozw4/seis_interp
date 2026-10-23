"""Opt-in ReGSI execution switches against the default path."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.configuration import ConfigurationError
from seis_interp.data.c3_poc_trace_graph import (
    build_c3_poc_trace_graph_domain,
    build_c3_poc_trace_graph_training_data,
)
from seis_interp.data.masked_trace_inputs import MaskedTraceGraphInputs
from seis_interp.models import relational_trace_graph
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import (
    build_trace_graph_edge_features,
    build_trace_graph_node_features,
    compute_trace_graph_geometry,
)
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config
from seis_interp.training import relational_trace_graph_poc_trainer as trainer
from seis_interp.training import trace_graph_execution
from seis_interp.training.relational_trace_graph_prediction import (
    predict_relational_trace_graph,
    predict_relational_trace_graph_states,
)
from seis_interp.training.trace_graph_ema import TraceGraphEMA, update_trace_graph_averages
from seis_interp.training.trace_graph_execution import (
    TraceGraphExecution,
    torch_cpu_threads,
    validate_trace_graph_execution,
)
from tests.fixtures.c3_poc_trace_graph import prepare_poc_trace_graph_inputs

_EXACT_SWITCHES = [
    {"neighbor_table": True},
    {"prefetch": True},
    {"deferred_checks": True},
    {"foreach_ema": True},
    {"neighbor_table": True, "prefetch": True, "deferred_checks": True, "foreach_ema": True},
]
# The C3 v4 ReGSI configuration (width 128) whose parameter count the switches keep.
_C3_MODEL = {
    "width": 128,
    "message_passing_rounds": 2,
    "time_downsample_factor": 2,
    "stem_kernel_size": 7,
    "temporal_kernel_size": 5,
    "temporal_dilations": (1, 2),
    "attention_width": 32,
    "relation_embedding_dim": 8,
    "relation_fusion": "learned_gate",
    "node_fourier_components": 16,
}


def _training_setup(tmp_path, **model):
    inputs, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config["model"].update(message_passing_rounds=2, temporal_dilations=[1, 2], **model)
    settings = validate_relational_trace_graph_poc_config(config)
    training = build_c3_poc_trace_graph_training_data(
        inputs, amplitude_scale=7.0, **settings.geometry
    )
    options = dict(settings.training)
    options.pop("device")
    options.pop("model_initialization_seed")
    options["random_seed"] = options.pop("episode_seed")
    options.update(max_steps=9, report_interval=4)
    return inputs, settings, training, options


def _train(settings, training, options, execution=None, **overrides):
    torch.manual_seed(4)
    model = RelationalTraceGraphInterpolator(**settings.model)
    averages = [TraceGraphEMA(0.9, (3,)), TraceGraphEMA(0.99)]
    result = trainer.train_relational_trace_graph_poc(
        model,
        training,
        graph_settings=settings.graph,
        ema=averages[0],
        ema_shadows=averages[1:],
        execution=execution,
        **{**options, **overrides},
    )
    return result, model, averages


def _without_timings(history):
    return [{k: v for k, v in row.items() if not k.endswith("seconds")} for row in history]


def _assert_same_training(first, second):
    (result, model, averages), (expected, expected_model, expected_averages) = first, second
    assert _without_timings(result.history) == _without_timings(expected.history)
    assert (
        result.steps_completed,
        result.episodes_started,
        result.episodes_completed,
        result.query_count,
        result.no_context_query_count,
    ) == (
        expected.steps_completed,
        expected.episodes_started,
        expected.episodes_completed,
        expected.query_count,
        expected.no_context_query_count,
    )
    state, expected_state = model.state_dict(), expected_model.state_dict()
    assert list(state) == list(expected_state)
    assert all(torch.equal(state[key], expected_state[key]) for key in expected_state)
    for average, reference in zip(averages, expected_averages, strict=True):
        assert average.updates == reference.updates
        for key, value in reference.state_dict().items():
            assert torch.equal(average.state_dict()[key], value)
        assert list(average.snapshots) == list(reference.snapshots)
        for update, snapshot in reference.snapshots.items():
            assert all(torch.equal(average.snapshots[update][k], v) for k, v in snapshot.items())


def test_execution_defaults_off_and_rejects_unknown_or_invalid_switches(tmp_path):
    assert validate_trace_graph_execution(None) == TraceGraphExecution()
    assert validate_trace_graph_execution({}) == TraceGraphExecution()
    options = validate_trace_graph_execution({"prefetch": True, "cpu_threads": 4})
    assert options.prefetch and options.cpu_threads == 4 and not options.deferred_checks
    for invalid in (
        {"prefetch": 1},
        {"cpu_threads": 0},
        {"cpu_threads": True},
        {"speed": True},
        ["prefetch"],
    ):
        with pytest.raises(ValueError):
            validate_trace_graph_execution(invalid)
    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config["training"]["execution"] = {"neighbor_table": True, "cpu_threads": 2}
    settings = validate_relational_trace_graph_poc_config(config)
    assert settings.training["execution"] == {"neighbor_table": True, "cpu_threads": 2}
    config["training"]["execution"] = {"neighbor_table": "yes"}
    with pytest.raises(ConfigurationError, match="neighbor_table must be boolean"):
        validate_relational_trace_graph_poc_config(config)


@pytest.mark.parametrize("execution", _EXACT_SWITCHES)
@pytest.mark.parametrize(
    "model", [{}, {"receptive_field_pruning": True, "activation_checkpointing": True}]
)
def test_exact_switches_reproduce_cpu_training_bit_for_bit(tmp_path, execution, model):
    """The fixture graphs are small enough that CPU index gathers backpropagate serially.

    On larger CPU graphs the default path itself is reproducible only under
    deterministic algorithms (see the large-graph test below).
    """
    _, settings, training, options = _training_setup(tmp_path, **model)
    _assert_same_training(
        _train(settings, training, options, execution),
        _train(settings, training, options),
    )


class _RecordedThreads:
    """``torch`` as seen by ``trace_graph_execution``, with thread changes only recorded.

    A run-time ``torch.set_num_threads`` can crash later CPU kernels of this
    container's PyTorch build, so the tests never change the real thread count.
    """

    def __init__(self, count=8):
        self.count, self.calls = count, []

    def __getattr__(self, name):
        return getattr(torch, name)

    def get_num_threads(self):
        return self.count

    def set_num_threads(self, count):
        self.calls.append(count)
        self.count = count


@pytest.fixture
def recorded_threads(monkeypatch):
    threads = _RecordedThreads()
    monkeypatch.setattr(trace_graph_execution, "torch", threads)
    return threads


def test_cpu_threads_apply_inside_the_block_and_are_restored(recorded_threads):
    with torch_cpu_threads(None):
        assert recorded_threads.count == 8
    assert recorded_threads.calls == []
    with torch_cpu_threads(3):
        assert recorded_threads.count == 3
    assert recorded_threads.calls == [3, 8] and recorded_threads.count == 8


def test_cpu_threads_are_rejected_on_a_cpu_device(tmp_path, recorded_threads):
    for device in ("cpu", torch.device("cpu")):
        with pytest.raises(ValueError, match="cpu_threads is not supported on a CPU device"):
            validate_trace_graph_execution({"cpu_threads": 2}, device=device)
    assert validate_trace_graph_execution({"cpu_threads": 2}, device="cuda:0").cpu_threads == 2
    assert validate_trace_graph_execution({"prefetch": True}, device="cpu").prefetch
    _, settings, training, options = _training_setup(tmp_path)
    for execution in ({"cpu_threads": 2}, TraceGraphExecution(cpu_threads=2)):
        with pytest.raises(ValueError, match="cpu_threads is not supported on a CPU device"):
            _train(settings, training, options, execution)
    assert recorded_threads.calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_training_with_every_exact_switch_is_bit_identical(tmp_path, recorded_threads):
    _, settings, training, options = _training_setup(tmp_path, receptive_field_pruning=True)
    execution = {**_EXACT_SWITCHES[-1], "cpu_threads": 2}
    seen = []
    _assert_same_training(
        _train(
            settings,
            training,
            options,
            execution,
            device="cuda",
            reporter=lambda message: seen.append(recorded_threads.count),
        ),
        _train(settings, training, options, device="cuda"),
    )
    assert seen and set(seen) == {2}
    assert recorded_threads.calls == [2, 8]


def _nonfinite_decoder(model):
    with torch.no_grad():
        model.decoder.head[-1].weight.fill_(float("nan"))


@pytest.mark.parametrize("deferred", [False, True])
def test_deferred_checks_fail_with_the_original_errors(tmp_path, monkeypatch, deferred):
    _, settings, training, options = _training_setup(tmp_path)
    execution = {"deferred_checks": True} if deferred else None
    original_loss = trainer.masked_trace_loss

    # 1. Non-finite predictions.
    original_init = RelationalTraceGraphInterpolator.__init__

    def nonfinite_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _nonfinite_decoder(self)

    monkeypatch.setattr(RelationalTraceGraphInterpolator, "__init__", nonfinite_init)
    with pytest.raises(ValueError, match="^selected prediction and target traces must be finite$"):
        _train(settings, training, options, execution)
    monkeypatch.setattr(RelationalTraceGraphInterpolator, "__init__", original_init)

    # 2. A non-finite loss from finite predictions.
    def nonfinite_loss(prediction, target, **kwargs):
        return original_loss(prediction, target, **kwargs) * math.nan

    monkeypatch.setattr(trainer, "masked_trace_loss", nonfinite_loss)
    with pytest.raises(RuntimeError, match="^non-finite PoC training loss$"):
        _train(settings, training, options, execution)
    monkeypatch.setattr(trainer, "masked_trace_loss", original_loss)

    # 3. A non-finite gradient norm after a finite loss.
    original_clip = torch.nn.utils.clip_grad_norm_

    def nonfinite_gradient(parameters, *args, **kwargs):
        parameters = list(parameters)
        parameters[0].grad.fill_(math.inf)
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", nonfinite_gradient)
    with pytest.raises(RuntimeError) as error:
        _train(settings, training, options, execution)
    assert str(error.value) == trainer.NONFINITE_GRADIENT_MESSAGE
    assert "non-finite" in str(error.value)


def test_deferred_checks_fail_no_later_than_the_next_report_interval(tmp_path, monkeypatch):
    _, settings, training, options = _training_setup(tmp_path)
    original_loss = trainer.masked_trace_loss
    calls = []

    def late_nonfinite_loss(prediction, target, **kwargs):
        calls.append(None)
        value = original_loss(prediction, target, **kwargs)
        return value * math.nan if len(calls) == 2 else value

    monkeypatch.setattr(trainer, "masked_trace_loss", late_nonfinite_loss)
    with pytest.raises(RuntimeError, match="non-finite PoC training loss"):
        _train(settings, training, options, {"deferred_checks": True})
    assert len(calls) == options["report_interval"]


def _graph_inputs(neighbors: int, rounds: int, dtype: torch.dtype, seed: int, samples: int = 12):
    rng = np.random.default_rng(seed)
    source = rng.uniform(0, 4, size=(60, 2))
    candidates = np.column_stack((source, source + rng.uniform(-3, 3, size=(60, 2))))
    query_source = rng.uniform(0.5, 3.5, size=(7, 2))
    queries = np.column_stack((query_source, query_source + rng.uniform(-3, 3, size=(7, 2))))

    def geometry(points):
        return compute_trace_graph_geometry(points[:, :2], points[:, 2:], azimuth_min_offset_m=0.01)

    plan = build_trace_graph_subgraph(
        geometry(queries),
        np.arange(1000, 1007),
        geometry(candidates),
        np.arange(1, 61),
        np.ones(60, dtype=bool),
        rounds=rounds,
        relation_scales_m=np.full((4, 2), 1.5),
        neighbors_per_relation=neighbors,
    )
    node_features = build_trace_graph_node_features(
        plan.geometry,
        midpoint_origin_m=np.zeros(2),
        position_scale_m=10,
        offset_scale_m=10,
        observed_mask=plan.observed_mask,
    )
    edge_features = build_trace_graph_edge_features(
        plan.geometry,
        sender_indices=plan.edge_index[0],
        destination_indices=plan.edge_index[1],
        relation_distances=plan.edge_distances,
        position_scale_m=10,
        offset_scale_m=10,
    )
    waveforms = torch.from_numpy(rng.standard_normal((len(plan.trace_ids), samples)))
    waveforms[~torch.from_numpy(plan.observed_mask)] = 0
    return MaskedTraceGraphInputs(
        waveforms=waveforms.to(dtype),
        node_features=torch.from_numpy(node_features).to(dtype),
        edge_features=torch.from_numpy(edge_features).to(dtype),
        edge_index=torch.from_numpy(plan.edge_index),
        edge_type=torch.from_numpy(plan.edge_type),
        observed_mask=torch.from_numpy(plan.observed_mask),
        query_indices=torch.from_numpy(plan.query_indices),
        coverage=torch.from_numpy(plan.coverage).to(dtype),
        dependency_rounds=plan.dependency_rounds,
    )


def _graph_model(dtype, width=8, **config):
    torch.manual_seed(11)
    model = RelationalTraceGraphInterpolator(
        width=width, stem_kernel_size=3, temporal_kernel_size=3, attention_width=5, **config
    )
    with torch.no_grad():
        model.decoder.head[-1].weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(2))
    return model.to(dtype)


def _forward_backward(model, inputs, **options):
    model.zero_grad(set_to_none=True)
    prediction, context = model(inputs, **options)
    weights = torch.linspace(
        -1, 1, prediction.numel(), dtype=prediction.dtype, device=prediction.device
    )
    (prediction * weights.reshape(prediction.shape)).sum().backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return prediction.detach().clone(), context.clone(), gradients


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    ("neighbors", "rounds", "fusion", "pooling", "memory"),
    [
        (1, 1, "mean", "mean", {}),
        (2, 2, "learned_gate", "mean", {}),
        (3, 2, "learned_gate", "rms", {"receptive_field_pruning": True}),
        (6, 3, "learned_gate", "rms", {"fused_edge_aggregation": True}),
        (
            3,
            2,
            "learned_gate",
            "mean",
            {
                "receptive_field_pruning": True,
                "fused_edge_aggregation": True,
                "activation_checkpointing": True,
            },
        ),
    ],
)
def test_forward_without_host_checks_is_bit_identical(
    dtype, neighbors, rounds, fusion, pooling, memory
):
    inputs = _graph_inputs(neighbors, rounds, dtype, seed=neighbors)
    model = _graph_model(
        dtype,
        message_passing_rounds=rounds,
        temporal_dilations=(1, 2, 1)[:rounds],
        relation_fusion=fusion,
        attention_pooling=pooling,
        relation_gate_pooling=pooling if fusion == "learned_gate" else "mean",
        node_fourier_components=2,
        **memory,
    )
    checked = _forward_backward(model, inputs)
    trusted = _forward_backward(model, inputs, host_checks=False)
    assert torch.equal(checked[0], trusted[0]) and torch.equal(checked[1], trusted[1])
    assert list(checked[2]) == list(trusted[2])
    assert all(torch.equal(checked[2][name], trusted[2][name]) for name in checked[2])


@pytest.fixture
def deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize(
    "memory",
    [
        {},
        {
            "receptive_field_pruning": True,
            "fused_edge_aggregation": True,
            "activation_checkpointing": True,
        },
    ],
)
def test_large_cpu_graphs_are_bit_identical_under_deterministic_algorithms(
    deterministic_algorithms, memory
):
    """CPU bit-identity of the switches is defined under deterministic algorithms.

    With several torch threads, the CPU backward of a large index gather (such as
    ``values[sender]``) accumulates gradients with parallel atomic additions, so
    two passes of the default path itself may differ in the last bits on a graph
    of this size. ``torch.use_deterministic_algorithms(True)`` serializes that
    accumulation; repeated default passes and ``host_checks=False`` then agree.
    """
    inputs = _graph_inputs(6, 2, torch.float32, seed=3, samples=256)
    model = _graph_model(
        torch.float32,
        width=32,
        message_passing_rounds=2,
        temporal_dilations=(1, 2),
        relation_fusion="learned_gate",
        **memory,
    )
    reference = _forward_backward(model, inputs)
    for options in ({}, {"host_checks": False}):
        result = _forward_backward(model, inputs, **options)
        assert torch.equal(result[0], reference[0]) and torch.equal(result[1], reference[1])
        assert list(result[2]) == list(reference[2])
        assert all(torch.equal(result[2][name], reference[2][name]) for name in reference[2])


def test_supplied_receptive_field_is_bit_identical_and_requires_pruning():
    inputs = _graph_inputs(3, 2, torch.float64, seed=9)
    model = _graph_model(
        torch.float64,
        message_passing_rounds=2,
        temporal_dilations=(1, 2),
        relation_fusion="learned_gate",
        receptive_field_pruning=True,
    )
    field = relational_trace_graph.trace_graph_receptive_field(
        inputs.edge_index, inputs.query_indices, 2, inputs.waveforms.shape[0]
    )
    computed = _forward_backward(model, inputs, host_checks=False)
    supplied = _forward_backward(model, inputs, host_checks=False, receptive_field=field)
    assert torch.equal(computed[0], supplied[0]) and torch.equal(computed[1], supplied[1])
    assert all(torch.equal(computed[2][name], supplied[2][name]) for name in computed[2])
    with pytest.raises(ValueError, match="rounds"):
        model(inputs, receptive_field=(field[0][:-1], field[1]))
    with pytest.raises(ValueError, match="requires receptive_field_pruning"):
        _graph_model(torch.float64)(inputs, receptive_field=field)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_foreach_ema_updates_are_bit_identical(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _graph_model(torch.float32, node_fourier_components=2).to(device)
    sequential = [TraceGraphEMA(decay, (2, 4)) for decay in (0.9, 0.999, 0.9998)]
    fused = [TraceGraphEMA(decay, (2, 4)) for decay in (0.9, 0.999, 0.9998)]
    generator = torch.Generator().manual_seed(3)
    for _ in range(5):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator).to(device))
        update_trace_graph_averages(sequential, model)
        update_trace_graph_averages(fused, model, foreach=True)
    for left, right in zip(fused, sequential, strict=True):
        assert left.updates == right.updates == 5
        assert all(torch.equal(left.state_dict()[k], v) for k, v in right.state_dict().items())
        assert list(left.snapshots) == list(right.snapshots) == [2, 4]


@pytest.mark.parametrize("table", [False, True])
def test_shared_state_prediction_equals_separate_predictions(tmp_path, table):
    inputs, settings, training, options = _training_setup(tmp_path)
    _, model, averages = _train(settings, training, options)
    domain = build_c3_poc_trace_graph_domain(inputs)
    arguments = {
        "graph_settings": settings.graph,
        "query_batch_size": 3,
        "observed_waveforms": training.observed_amplitudes,
    }
    states = [average.state_dict() for average in averages] + [model.state_dict()]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    shared = predict_relational_trace_graph_states(
        model,
        states,
        domain,
        training.preprocessing,
        **arguments,
        **(
            {"neighbor_table": trainer.build_training_neighbor_table(training, settings.graph)}
            if table
            else {}
        ),
    )
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
    for state, result in zip(states, shared, strict=True):
        separate_model = RelationalTraceGraphInterpolator(**settings.model)
        separate_model.load_state_dict(state)
        expected = predict_relational_trace_graph(
            separate_model, domain, training.preprocessing, **arguments
        )
        np.testing.assert_array_equal(result.prediction, expected.prediction)
        np.testing.assert_array_equal(result.has_observed_context, expected.has_observed_context)
        np.testing.assert_array_equal(result.query_trace_ids, expected.query_trace_ids)
        for key, value in expected.diagnostics.items():
            if key not in ("timings", "resources"):
                assert result.diagnostics[key] == value, key


def test_switches_keep_parameter_count_and_state_dict_keys():
    torch.manual_seed(0)
    plain = RelationalTraceGraphInterpolator(**_C3_MODEL)
    torch.manual_seed(0)
    switched = RelationalTraceGraphInterpolator(
        **_C3_MODEL,
        receptive_field_pruning=True,
        fused_edge_aggregation=True,
        activation_checkpointing=True,
    )
    assert sum(p.numel() for p in plain.parameters()) == 363_269
    assert sum(p.numel() for p in switched.parameters()) == 363_269
    assert list(switched.state_dict()) == list(plain.state_dict())
    assert all(torch.equal(switched.state_dict()[k], v) for k, v in plain.state_dict().items())


def test_neighbor_table_requires_exact_index(tmp_path):
    _, settings, training, options = _training_setup(tmp_path)
    graph = replace(settings.graph, neighbor_search="brute_force")
    with pytest.raises(ValueError, match="exact_index"):
        trainer.train_relational_trace_graph_poc(
            RelationalTraceGraphInterpolator(**settings.model),
            training,
            graph_settings=graph,
            execution={"neighbor_table": True},
            **options,
        )
    with pytest.raises(ValueError, match="requires execution.neighbor_table"):
        trainer.train_relational_trace_graph_poc(
            RelationalTraceGraphInterpolator(**settings.model),
            training,
            graph_settings=settings.graph,
            neighbor_table=trainer.build_training_neighbor_table(training, settings.graph),
            **options,
        )


def test_config_rejects_neighbor_table_without_exact_index(tmp_path):
    for options in ({"neighbor_table": True}, TraceGraphExecution(neighbor_table=True)):
        with pytest.raises(ValueError, match="requires graph.neighbor_search=exact_index"):
            validate_trace_graph_execution(options, neighbor_search="brute_force")
        assert validate_trace_graph_execution(options, neighbor_search="exact_index").neighbor_table
    other = {"prefetch": True, "deferred_checks": True, "foreach_ema": True}
    assert validate_trace_graph_execution(other, neighbor_search="brute_force").prefetch
    _, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    config["graph"]["neighbor_search"] = "brute_force"
    config["training"]["execution"] = other
    validate_relational_trace_graph_poc_config(config)
    config["training"]["execution"] = {"neighbor_table": True}
    with pytest.raises(ConfigurationError, match="requires graph.neighbor_search=exact_index"):
        validate_relational_trace_graph_poc_config(config)


def test_every_switch_combination_is_accepted():
    for flags in itertools.product((False, True), repeat=5):
        options = dict(
            zip(
                (
                    "neighbor_table",
                    "prefetch",
                    "deferred_checks",
                    "foreach_ema",
                    "shared_prediction",
                ),
                flags,
                strict=True,
            )
        )
        assert validate_trace_graph_execution(options) == TraceGraphExecution(**options)
