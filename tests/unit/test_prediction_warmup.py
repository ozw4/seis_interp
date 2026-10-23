"""GPU warm-up planning tested with CPU tensors and forwards only."""

import numpy as np
import torch

from seis_interp.data.c3_poc_trace_graph import (
    build_c3_poc_trace_graph_domain,
    build_c3_poc_trace_graph_training_data,
)
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config
from seis_interp.training import prediction_warmup as warmup
from tests.fixtures.c3_poc_trace_graph import prepare_poc_trace_graph_inputs


class Recorder(torch.nn.Module):
    kernel_size = 3
    halo_radius = 20
    network = [None] * 6
    message_passing_rounds = 2

    def __init__(self):
        super().__init__()
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    def forward(self, value):
        self.calls.append(value)
        torch.rand(1)
        return value


def test_ccnet_warms_one_maximal_halo_without_changing_rng_or_releasing_cache(monkeypatch):
    model = Recorder()
    before = torch.get_rng_state().clone()
    zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros", lambda *args, **kwargs: zeros(*args, **(kwargs | {"device": "cpu"}))
    )

    def unexpected_cache_release():
        raise AssertionError("cache must remain allocated through training")

    monkeypatch.setattr(torch.cuda, "empty_cache", unexpected_cache_release)
    warmup.warm_ccnet(
        model,
        (11, 17, 17, 17, 17),
        {
            "device": "cuda:0",
            "prediction": {"core_shape": [11, 2, 2, 2, 2]},
        },
    )
    assert len(model.calls) == 1
    assert model.calls[0].shape == (1, 1, 11, 14, 14, 14, 14)
    assert torch.count_nonzero(model.calls[0]) == 0
    assert model.training and torch.equal(before, torch.get_rng_state())


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
    warmup.warm_graph(
        model,
        domain,
        training.preprocessing,
        settings.graph,
        domain.trace_ids[~domain.observed_mask],
        2,
        "cuda:0",
        None,
    )
    assert len(model.calls) == 1 and len(plans) > 1
    assert model.calls[0].waveforms.shape[0] == max(len(plan.trace_ids) for plan in plans)
    assert torch.count_nonzero(model.calls[0].waveforms) == 0
