from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_inputs import validate_masked_trace_graph_inputs
from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from tests.fixtures.relational_trace_graph import (
    make_relational_trace_domains,
    make_relational_trace_plan,
)


def _inputs():
    domain, training, values = make_relational_trace_domains()
    preprocessing = fit_trace_graph_preprocessing(
        training, values, position_scale_m=10.0, offset_scale_m=2.0, azimuth_min_offset_m=0.1
    )
    return MaskedTraceSource(domain, preprocessing, values).inputs(
        make_relational_trace_plan(domain)
    )


def test_validation_and_device_transfer_preserve_inputs() -> None:
    inputs = _inputs()
    originals = {field.name: getattr(inputs, field.name).clone() for field in fields(inputs)}
    assert validate_masked_trace_graph_inputs(inputs) is inputs
    moved = inputs.to("cpu")

    for name, original in originals.items():
        assert torch.equal(getattr(inputs, name), original)
        assert torch.equal(getattr(moved, name), original)


@pytest.mark.parametrize("field_name", ["waveforms", "node_features", "edge_features", "coverage"])
def test_tensor_contract_rejects_nonfinite_values(field_name: str) -> None:
    inputs = _inputs()
    invalid = getattr(inputs, field_name).clone()
    invalid.reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_masked_trace_graph_inputs(replace(inputs, **{field_name: invalid}))


def test_tensor_contract_rejects_query_truth_and_query_senders() -> None:
    inputs = _inputs()
    waveforms = inputs.waveforms.clone()
    waveforms[inputs.query_indices[0], 0] = 1.0
    with pytest.raises(ValueError, match="exactly zero"):
        validate_masked_trace_graph_inputs(replace(inputs, waveforms=waveforms))
    edges = inputs.edge_index.clone()
    edges[0, 0] = inputs.query_indices[0]
    with pytest.raises(ValueError, match="sender must be observed"):
        validate_masked_trace_graph_inputs(replace(inputs, edge_index=edges))


def test_observed_zero_waveform_has_a_distinct_mask_feature() -> None:
    inputs = _inputs()
    zeros = replace(inputs, waveforms=torch.zeros_like(inputs.waveforms))
    validate_masked_trace_graph_inputs(zeros)
    assert torch.all(zeros.node_features[zeros.observed_mask, -1] == 1)
    assert torch.all(zeros.node_features[zeros.query_indices, -1] == 0)
    wrong_features = zeros.node_features.clone()
    wrong_features[:, -1] = 0
    with pytest.raises(ValueError, match="observed feature"):
        validate_masked_trace_graph_inputs(replace(zeros, node_features=wrong_features))


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("edge_index", torch.zeros((3, 1), dtype=torch.int64), "shape"),
        ("edge_type", torch.empty(0, dtype=torch.int64), "shape"),
        ("query_indices", torch.tensor([999]), "out-of-range"),
        ("query_indices", torch.tensor([0.0]), "int64"),
        ("observed_mask", np.ones(3, dtype=bool), "Tensor"),
    ],
)
def test_tensor_contract_rejects_inconsistent_layout(field_name, replacement, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_masked_trace_graph_inputs(replace(_inputs(), **{field_name: replacement}))


def test_coverage_must_agree_with_incoming_relation_availability() -> None:
    inputs = _inputs()
    with pytest.raises(ValueError, match="selected incoming edges"):
        validate_masked_trace_graph_inputs(
            replace(inputs, coverage=torch.zeros_like(inputs.coverage))
        )
    coverage = inputs.coverage.clone()
    empty = coverage[:, :, 0] == 0
    coverage[:, :, 1][empty] = 2.0
    with pytest.raises(ValueError, match="empty relation"):
        validate_masked_trace_graph_inputs(replace(inputs, coverage=coverage))
