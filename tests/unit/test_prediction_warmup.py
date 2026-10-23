"""GPU warm-up planning tested with CPU tensors and forwards only."""

import random

import numpy as np
import pytest
import torch

from seis_interp.data.c3_poc_trace_graph import (
    build_c3_poc_trace_graph_domain,
    build_c3_poc_trace_graph_training_data,
)
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config
from seis_interp.training import phase_progress
from seis_interp.training import prediction_warmup as warmup
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume
from seis_interp.training.exact_resume import restore_rng, rng_state
from tests.fixtures.c3_poc_trace_graph import prepare_poc_trace_graph_inputs


class Recorder(torch.nn.Module):
    kernel_size = 3
    halo_radius = 20
    network = [None] * 6
    message_passing_rounds = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    def forward(self, value):
        self.calls.append(value)
        random.random()
        np.random.random()
        torch.rand(1)
        return value


def test_ccnet_warms_all_halos_without_changing_rng_or_releasing_cache(monkeypatch):
    model = Recorder()
    before = torch.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros", lambda *args, **kwargs: zeros(*args, **(kwargs | {"device": "cpu"}))
    )

    def unexpected_cache_release():
        raise AssertionError("cache must remain allocated through training")

    monkeypatch.setattr(torch.cuda, "empty_cache", unexpected_cache_release)
    ticks = iter(range(0, 10000, 31))
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: next(ticks))
    reports = []
    warmup.warm_ccnet(
        model,
        (11, 17, 1, 1, 1),
        {
            "device": "cuda:0",
            "prediction": {"core_shape": [11, 2, 1, 1, 1]},
        },
        reporter=reports.append,
    )
    assert reports and all("prediction warm-up" in message for message in reports)
    expected = [
        (1, 1, *(s.stop - s.start for s in tile.input_slices))
        for tile in warmup.iter_ccnet5d_tiles((11, 17, 1, 1, 1), (11, 2, 1, 1, 1), halo_radius=6)
    ]
    assert [tuple(value.shape) for value in model.calls] == expected
    assert max(expected, key=np.prod) == (1, 1, 11, 14, 1, 1, 1)
    assert all(torch.count_nonzero(value) == 0 for value in model.calls)
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert python_before == random.getstate()
    assert all(
        np.array_equal(a, b) for a, b in zip(numpy_before, np.random.get_state(), strict=True)
    )


@pytest.mark.parametrize("full_trace", [False, True])
def test_ccnet_warmup_preserves_training_and_prediction_bitwise(monkeypatch, full_trace):
    shape = (13, 1, 11, 1, 1)
    core = (3, 1, 3, 1, 1)
    model = CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=3)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    volume = ObservedC3Volume(
        values=np.ones(shape, dtype=np.float32),
        time_s=np.arange(shape[0]) * 0.004,
        array_rows=np.arange(np.prod(shape[1:])).reshape(shape[1:]),
        observed_trace_mask=np.ones(shape[1:], dtype=bool),
        evaluation_target_trace_mask=np.zeros(shape[1:], dtype=bool),
    )
    volume.observed_trace_mask.flat[::2] = False
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: zeros(*a, **(kw | {"device": "cpu"})))
    monkeypatch.setattr(model, "to", lambda _: model)
    initial_rng = rng_state()
    results = []
    for warmed in (False, True):
        model.load_state_dict(weights)
        restore_rng(initial_rng)
        seen = []
        handle = model.register_forward_pre_hook(
            lambda _, args, seen=seen: seen.append(args[0].shape)
        )
        if warmed:
            warmup.warm_ccnet(
                model,
                shape,
                {
                    "device": "cuda:1",
                    "prediction": {"core_shape": core},
                    "execution": {"full_trace_prediction": full_trace},
                },
            )
        warm_shapes = seen.copy()
        seen.clear()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(torch.rand(1, 1, 5, 2, 3, 1, 4)).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        seen.clear()
        prediction = predict_ccnet5d_volume(
            model,
            volume,
            amplitude_rms=1,
            core_shape=(shape[0], *core[1:]) if full_trace else core,
            device="cpu",
        )
        handle.remove()
        if warmed:
            assert warm_shapes == seen
            assert len(set(warm_shapes)) > 1
        results.append(
            (
                losses,
                prediction.values.tobytes(),
                [p.detach().clone() for p in model.parameters()],
                torch.get_rng_state(),
            )
        )
    assert results[0][:2] == results[1][:2]
    assert all(torch.equal(a, b) for a, b in zip(results[0][2], results[1][2], strict=True))
    assert torch.equal(results[0][3], results[1][3])


def test_graph_warms_the_largest_prediction_plan_once_with_zero_amplitudes(tmp_path, monkeypatch):
    inputs, config, _ = prepare_poc_trace_graph_inputs(tmp_path)
    settings = validate_relational_trace_graph_poc_config(config)
    training = build_c3_poc_trace_graph_training_data(
        inputs, amplitude_scale=7.0, **settings.geometry
    )
    domain = build_c3_poc_trace_graph_domain(inputs)
    model = Recorder()
    plans = []
    build = warmup.FixedTraceGraphSubgraphBuilder.build
    assemble = warmup.assemble_masked_trace_graph_inputs

    def record_plan(*args, **kwargs):
        result = build(*args, **kwargs)
        plans.append(result)
        return result

    def cpu_inputs(*args, **kwargs):
        assert np.count_nonzero(kwargs["observed_waveforms"]) == 0
        return assemble(*args, **(kwargs | {"device": "cpu"}))

    monkeypatch.setattr(warmup.FixedTraceGraphSubgraphBuilder, "build", record_plan)
    monkeypatch.setattr(warmup, "assemble_masked_trace_graph_inputs", cpu_inputs)
    ticks = iter(range(0, 10000, 31))
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: next(ticks))
    reports = []
    warmup.warm_graph(
        model,
        domain,
        training.preprocessing,
        settings.graph,
        domain.trace_ids[~domain.observed_mask],
        2,
        "cuda:0",
        None,
        reporter=reports.append,
    )
    assert reports and all("prediction warm-up" in message for message in reports)
    assert len(model.calls) == 1 and len(plans) > 1
    assert model.calls[0].waveforms.shape[0] == max(len(plan.trace_ids) for plan in plans)
    assert torch.count_nonzero(model.calls[0].waveforms) == 0


def test_coordinate_warmup_reports_completed_work_without_changing_rng(monkeypatch):
    model = Recorder()
    before = rng_state()
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: zeros(*a, **(kw | {"device": "cpu"})))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    ticks = iter(range(0, 10000, 31))
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: next(ticks))
    reports = []
    warmup.warm_coordinates(model, (4, 3), "cuda:0", reporter=reports.append)
    assert reports == ["prediction warm-up complete (coordinates)"]
    after = rng_state()
    assert before["python"] == after["python"]
    assert all(np.array_equal(a, b) for a, b in zip(before["numpy"], after["numpy"], strict=True))
    assert torch.equal(before["torch"], after["torch"])
