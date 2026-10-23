"""CCNet-5D outputs restricted to the spatial dependencies of selected traces."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from seis_interp.models.ccnet5d import CCNet5D


@dataclass(frozen=True)
class QueryConvolutions:
    source_indices: torch.Tensor
    receiver_indices: torch.Tensor


@dataclass(frozen=True)
class CCNet5DTraceQueries:
    input_cells: np.ndarray
    query_cells: np.ndarray
    layers: tuple[QueryConvolutions, ...]


def _neighbors(cells, shape, axes, width):
    coordinates = np.array(np.unravel_index(cells, shape)).T
    offsets = np.array(list(product(range(-(width // 2), width // 2 + 1), repeat=2)))
    neighbors = np.broadcast_to(coordinates[:, None], (len(cells), len(offsets), len(shape))).copy()
    neighbors[:, :, axes[0]] += offsets[None, :, 0]
    neighbors[:, :, axes[1]] += offsets[None, :, 1]
    valid = ((neighbors >= 0) & (neighbors < np.asarray(shape))).all(axis=2)
    result = np.full(valid.shape, -1, dtype=np.int64)
    result[valid] = np.ravel_multi_index(neighbors[valid].T, shape)
    return result


def _local_indices(neighbors, cells, device):
    valid = neighbors >= 0
    indices = np.full(neighbors.shape, len(cells), dtype=np.int64)
    indices[valid] = np.searchsorted(cells, neighbors[valid])
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def prepare_ccnet5d_trace_queries(model: CCNet5D, trace_mask, *, device):
    """Plan each Conv3D/Conv2D backward from the requested (batch, spatial) cells."""
    mask = np.asarray(trace_mask)
    if mask.ndim != 5 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("trace queries require a nonempty Boolean (batch, four spatial axes) mask")
    queries = np.flatnonzero(mask)
    outputs = queries
    layers = []
    for module in reversed(model.network):
        receiver = _neighbors(outputs, mask.shape, (3, 4), module.conv2d.kernel_size[0])
        intermediate = np.unique(receiver[receiver >= 0])
        source = _neighbors(intermediate, mask.shape, (1, 2), module.conv3d.kernel_size[1])
        inputs = np.unique(source[source >= 0])
        layers.append(
            QueryConvolutions(
                _local_indices(source, inputs, device),
                _local_indices(receiver, intermediate, device),
            )
        )
        outputs = inputs
    return CCNet5DTraceQueries(outputs, queries, tuple(reversed(layers)))


def _convolve_cells(values, indices, weights, bias, padding, dilation, relu):
    gathered = values[indices.clamp(max=len(values) - 1)]
    gathered = gathered.masked_fill((indices == len(values))[:, :, None, None], 0)
    columns = gathered.permute(0, 2, 1, 3).flatten(1, 2)
    result = F.conv1d(columns, weights, bias, padding=padding, dilation=dilation)
    return F.relu(result) if relu else result


def _convolve_spatial_chunks(
    values, indices, weights, bias, *, padding, dilation, relu, chunk_size, recompute
):
    outputs = []
    for start in range(0, len(indices), chunk_size):
        arguments = (
            values,
            indices[start : start + chunk_size],
            weights,
            bias,
            padding,
            dilation,
            relu,
        )
        if recompute and torch.is_grad_enabled():
            output = checkpoint(_convolve_cells, *arguments, use_reentrant=False)
        else:
            output = _convolve_cells(*arguments)
        outputs.append(output)
    return torch.cat(outputs)


def forward_ccnet5d_trace_queries(
    model: CCNet5D,
    values,
    queries: CCNet5DTraceQueries,
    *,
    spatial_chunk_size=64,
    recompute=True,
):
    """Return complete traces, bounding gathers along spatial cells only.

    Checkpointing recomputes spatial convolution chunks during backward. Every
    convolution receives the entire time axis, with the original time padding.
    """
    if values.ndim != 3 or values.shape[:2] != (len(queries.input_cells), 1):
        raise ValueError("query inputs must be (dependency cells, one channel, time)")
    if (
        isinstance(spatial_chunk_size, bool)
        or not isinstance(spatial_chunk_size, int)
        or spatial_chunk_size < 1
    ):
        raise ValueError("spatial_chunk_size must be a positive integer")
    for module, layer in zip(model.network, queries.layers, strict=True):
        weights = module.conv3d.weight.permute(0, 1, 3, 4, 2).flatten(1, 3)
        values = _convolve_spatial_chunks(
            values,
            layer.source_indices,
            weights,
            module.conv3d.bias,
            padding=module.conv3d.padding[0],
            dilation=module.conv3d.dilation[0],
            relu=True,
            chunk_size=spatial_chunk_size,
            recompute=recompute,
        )
        values = _convolve_spatial_chunks(
            values,
            layer.receiver_indices,
            module.conv2d.weight.flatten(1).unsqueeze(-1),
            module.conv2d.bias,
            padding=0,
            dilation=1,
            relu=module.output_activation == "relu",
            chunk_size=spatial_chunk_size,
            recompute=recompute,
        )
    return values[:, 0]
