"""Label-free tensors for a masked, directed trace dependency graph."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import torch


@dataclass(frozen=True)
class MaskedTraceGraphInputs:
    """A graph batch with zero query waveforms and observed-only senders.

    Coverage has shape ``[N, 4, 2]``: selected degree divided by k, followed
    by minimum relation distance. Both entries are zero for empty relations.
    ``dependency_rounds`` is the number of synchronous updates whose incoming
    dependencies were collected, independent of the graph's realized depth.
    Array-row identities and training/evaluation labels are deliberately absent.
    """

    waveforms: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    observed_mask: torch.Tensor
    query_indices: torch.Tensor
    coverage: torch.Tensor
    dependency_rounds: int

    def to(self, device: torch.device | str) -> MaskedTraceGraphInputs:
        """Move this batch's tensors together without changing their dtypes."""
        return MaskedTraceGraphInputs(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
                if field.name != "dependency_rounds"
            },
            dependency_rounds=self.dependency_rounds,
        )


def validate_masked_trace_graph_inputs(inputs: MaskedTraceGraphInputs) -> MaskedTraceGraphInputs:
    """Check shapes, visibility and finite values without changing the batch."""
    if not isinstance(inputs, MaskedTraceGraphInputs):
        raise TypeError("inputs must be a MaskedTraceGraphInputs object")
    if (
        isinstance(inputs.dependency_rounds, bool)
        or not isinstance(inputs.dependency_rounds, Integral)
        or inputs.dependency_rounds < 1
    ):
        raise ValueError("dependency_rounds must be a positive integer")
    tensors = {
        field.name: getattr(inputs, field.name)
        for field in fields(inputs)
        if field.name != "dependency_rounds"
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if len({tensor.device for tensor in tensors.values()}) != 1:
        raise ValueError("all input tensors must be on the same device")
    if inputs.waveforms.ndim != 2 or inputs.waveforms.shape[1] <= 0:
        raise ValueError("waveforms must have shape [N, T] with positive T")
    node_count = inputs.waveforms.shape[0]
    if inputs.edge_index.ndim != 2 or inputs.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    edge_count = inputs.edge_index.shape[1]
    for name, shape in (
        ("node_features", (node_count, 9)),
        ("edge_features", (edge_count, 15)),
        ("edge_type", (edge_count,)),
        ("observed_mask", (node_count,)),
        ("coverage", (node_count, 4, 2)),
    ):
        if tensors[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if inputs.query_indices.ndim != 1:
        raise ValueError("query_indices must have shape [Q]")
    floats = (inputs.waveforms, inputs.node_features, inputs.edge_features, inputs.coverage)
    if any(not value.is_floating_point() for value in floats):
        raise TypeError("waveforms, features and coverage must have floating-point dtypes")
    if len({value.dtype for value in floats}) != 1:
        raise TypeError("waveforms, features and coverage must share a dtype")
    if any(not bool(torch.isfinite(value).all()) for value in floats):
        raise ValueError("waveforms, features and coverage must be finite")
    if inputs.observed_mask.dtype != torch.bool:
        raise TypeError("observed_mask must have dtype torch.bool")
    for name in ("edge_index", "edge_type", "query_indices"):
        if tensors[name].dtype != torch.int64:
            raise TypeError(f"{name} must have dtype torch.int64")
    for name in ("edge_index", "query_indices"):
        if bool(((tensors[name] < 0) | (tensors[name] >= node_count)).any()):
            raise ValueError(f"{name} contains an out-of-range node index")
    if bool(((inputs.edge_type < 0) | (inputs.edge_type >= 4)).any()):
        raise ValueError("edge_type must be in [0, 4)")
    if len(torch.unique(inputs.query_indices)) != len(inputs.query_indices):
        raise ValueError("query_indices must be unique")
    if bool(inputs.observed_mask[inputs.query_indices].any()):
        raise ValueError("query nodes must be unobserved")
    if bool((inputs.waveforms[~inputs.observed_mask] != 0).any()):
        raise ValueError("unobserved waveforms must be exactly zero")
    if not torch.equal(
        inputs.node_features[:, -1], inputs.observed_mask.to(inputs.waveforms.dtype)
    ):
        raise ValueError("node observed feature must match observed_mask")
    sender, destination = inputs.edge_index
    if not bool(inputs.observed_mask[sender].all()):
        raise ValueError("every edge sender must be observed")
    if bool((sender == destination).any()):
        raise ValueError("self edges are not permitted")
    if bool((inputs.coverage < 0).any()) or bool((inputs.coverage[:, :, 0] > 1).any()):
        raise ValueError("coverage must be nonnegative and degree/k must not exceed 1")
    groups = destination * 4 + inputs.edge_type
    degree = torch.bincount(groups, minlength=node_count * 4).reshape(node_count, 4)
    available = degree > 0
    if not torch.equal(inputs.coverage[:, :, 0] > 0, available):
        raise ValueError("coverage must describe the selected incoming edges")
    if bool((inputs.coverage[:, :, 1][~available] != 0).any()):
        raise ValueError("empty relation minimum distance must be zero")
    return inputs
