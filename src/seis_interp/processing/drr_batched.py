"""Opt-in batched double-precision DRR for parameter searches."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from itertools import islice, product

import numpy as np
import torch

from seis_interp.processing.drr import (
    select_drr_frequencies,
    validate_drr_block_inputs,
    validate_drr_parameters,
)
from seis_interp.processing.drr_windows import (
    WindowedDrrResult,
    _axis_window_starts,
    _positive_hann_weights,
    _validated_window_spec,
)
from seis_interp.processing.level_four_hankel import hankelize_level_four


def _hankel_indices(shape):
    cells = int(np.prod(shape))
    indices = hankelize_level_four(
        np.arange(cells).reshape(shape).astype(np.complex128)
    ).real.astype(int)
    counts = np.bincount(indices.ravel(), minlength=cells)
    inverse = np.full((cells, int(counts.max())), indices.size, dtype=np.int64)
    positions = np.zeros(cells, dtype=int)
    # Match the reference's C-order start-column accumulation.
    for column in range(indices.shape[1]):
        for row in range(indices.shape[0]):
            cell = indices[row, column]
            inverse[cell, positions[cell]] = row * indices.shape[1] + column
            positions[cell] += 1
    return indices, inverse, counts


def _reduce_numpy(current, indices, inverse, counts, rank, power):
    matrices = current[:, indices]
    left, singular, right = np.linalg.svd(matrices, full_matrices=False)
    kept = singular[:, :rank]
    ratios = np.divide(singular[:, rank, None], kept, out=np.zeros_like(kept), where=kept > 0)
    damped = kept * np.clip(1 - ratios**power, 0, 1)
    reduced = (left[:, :, :rank] * damped[:, None]) @ right[:, :rank]
    flat = np.pad(reduced.reshape(len(current), -1), ((0, 0), (0, 1)))
    result = np.zeros_like(current)
    for contribution in range(inverse.shape[1]):
        result += flat[:, inverse[:, contribution]]
    return result / counts


def _reduce_torch(current, indices, inverse, counts, rank, power):
    """Compute the same damped projection from the smaller Hermitian Gram matrix."""
    matrices = current[:, indices]
    eigenvalues, right = torch.linalg.eigh(matrices.mH @ matrices)
    singular = eigenvalues.clamp_min(0).sqrt()
    kept = singular[:, -rank:]
    ratios = torch.where(kept > 0, singular[:, -rank - 1, None] / kept, 0.0)
    weights = (1 - ratios**power).clamp(0, 1)
    right = right[:, :, -rank:]
    reduced = ((matrices @ right) * weights[:, None]) @ right.mH
    flat = torch.nn.functional.pad(reduced.reshape(len(current), -1), (0, 1))
    result = torch.zeros_like(current)
    for contribution in range(inverse.shape[1]):
        result += flat[:, inverse[:, contribution]]
    return result / counts


def _reconstruct_blocks(
    values, masks, selection, maps, rank, power, iterations, device, matrix_limit
):
    shape = values[0].shape
    mask = np.stack(masks).reshape(len(masks), 1, -1)
    observed = np.stack(values).reshape(len(values), shape[0], -1).astype(np.float64)
    observed = np.where(mask, observed, 0.0)
    transformed = np.fft.rfft(observed, n=selection.fft_length, axis=1, norm="ortho")
    predicted = np.zeros_like(transformed)
    # Bound the stacked Hankel matrices even when one block's complete spectrum
    # exceeds the limit. One matrix is the irreducible minimum allocation.
    bins_per_batch = max(1, matrix_limit // (len(values) * maps[0].size))
    cpu = torch.device(device).type == "cpu"
    tensor_maps = None if cpu else tuple(torch.as_tensor(value, device=device) for value in maps)
    for start in range(0, len(selection.bin_indices), bins_per_batch):
        bins = selection.bin_indices[start : start + bins_per_batch]
        frequency = np.ascontiguousarray(transformed[:, bins]).reshape(-1, mask.shape[-1])
        expanded_mask = np.broadcast_to(mask, (len(mask), len(bins), mask.shape[-1])).reshape(
            frequency.shape
        )
        if cpu:
            current = frequency.copy()
            for _ in range(iterations):
                current = _reduce_numpy(current, *maps, rank, power)
                current = np.where(expanded_mask, frequency, current)
        else:
            frequency = torch.as_tensor(frequency, dtype=torch.complex128, device=device)
            expanded_mask = torch.as_tensor(np.array(expanded_mask), device=device)
            current = frequency.clone()
            for _ in range(iterations):
                current = _reduce_torch(current, *tensor_maps, rank, power)
                current = torch.where(expanded_mask, frequency, current)
            current = current.cpu().numpy()
        predicted[:, bins] = current.reshape(len(values), len(bins), -1)
    predicted[:, 0] = predicted[:, 0].real
    predicted[:, -1] = predicted[:, -1].real
    result = np.fft.irfft(predicted, n=selection.fft_length, axis=1, norm="ortho")[:, : shape[0]]
    return np.where(mask, observed, result).reshape(len(values), *shape).astype(values[0].dtype)


def _reconstruct_batch(
    batch, *, values, mask, shape, selection, maps, rank, power, iterations, device, matrix_limit
):
    slices, batch_values, batch_masks = [], [], []
    for location in batch:
        region = tuple(slice(a, a + width) for a, width in zip(location, shape, strict=True))
        if mask[region].any():
            slices.append(region)
            batch_values.append(values[(slice(None), *region)])
            batch_masks.append(mask[region])
    predictions = []
    if slices:
        predictions = _reconstruct_blocks(
            batch_values,
            batch_masks,
            selection,
            maps,
            rank,
            power,
            iterations,
            device,
            matrix_limit,
        )
    return slices, predictions, len(batch)


def interpolate_drr_volume_batched(
    observed,
    observed_trace_mask,
    time_s,
    *,
    rank,
    damping_power,
    n_iterations,
    frequency_min_hz,
    frequency_max_hz,
    spatial_window_shape,
    spatial_overlap,
    overlap_policy="explicit",
    boundary_rule="clip_and_shift",
    device="cpu",
    batch_matrix_limit=2_000_000,
    workers=1,
    reporter=None,
):
    """Clip windows to each axis and shift the last window inside the volume.

    Nonempty blocks use positive interior-Hann synthesis weights. With
    ``overlap_policy='point_half'``, line axes have zero overlap and point axes
    have half the requested window width. ``batch_matrix_limit`` bounds Hankel
    elements per worker and SVD call, except for one irreducible matrix.
    CPU uses NumPy SVD. CUDA computes the damped projection via a double-precision
    Hermitian Gram eigendecomposition; final candidates can be verified on CPU.
    """
    values, mask = validate_drr_block_inputs(observed, observed_trace_mask, time_s)
    if boundary_rule != "clip_and_shift":
        raise ValueError("DRR boundary_rule must be clip_and_shift")
    if overlap_policy not in ("explicit", "point_half"):
        raise ValueError("DRR overlap_policy must be explicit or point_half")
    if overlap_policy == "point_half":
        requested, _ = _validated_window_spec(spatial_window_shape, (0, 0, 0, 0))
        if requested is None:
            raise ValueError("point_half overlap requires spatial_window_shape")
        spatial_overlap = (0, requested[1] // 2, 0, requested[3] // 2)
    window, overlap = _validated_window_spec(spatial_window_shape, spatial_overlap)
    if window is None:
        window, overlap = values.shape[1:], (0,) * 4
    shape = tuple(min(n, width) for n, width in zip(mask.shape, window, strict=True))
    rank, power, iterations = validate_drr_parameters(rank, damping_power, n_iterations, shape)
    selection = select_drr_frequencies(
        time_s, frequency_min_hz=frequency_min_hz, frequency_max_hz=frequency_max_hz
    )
    if (
        isinstance(batch_matrix_limit, bool)
        or not isinstance(batch_matrix_limit, int)
        or batch_matrix_limit < 1
    ):
        raise ValueError("batch_matrix_limit must be a positive integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers > 1 and torch.device(device).type != "cpu":
        raise ValueError("multiple workers require CPU execution")
    starts = [
        _axis_window_starts(n, width, shared)
        for n, width, shared in zip(mask.shape, window, overlap, strict=True)
    ]
    maps = _hankel_indices(shape)
    batch_size = max(1, batch_matrix_limit // (len(selection.bin_indices) * maps[0].size))
    locations = iter(product(*starts))
    weighted = np.zeros(values.shape, dtype=np.float64)
    weights = np.zeros(mask.shape, dtype=np.float64)
    taper = _positive_hann_weights(shape)
    count = empty = 0
    block_total = int(np.prod([len(axis) for axis in starts]))
    reconstruct = partial(
        _reconstruct_batch,
        values=values,
        mask=mask,
        shape=shape,
        selection=selection,
        maps=maps,
        rank=rank,
        power=power,
        iterations=iterations,
        device=device,
        matrix_limit=batch_matrix_limit,
    )
    batches = iter(lambda: list(islice(locations, batch_size)), [])
    with ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext() as executor:
        results = executor.map(reconstruct, batches) if executor else map(reconstruct, batches)
        for slices, predictions, batch_count in results:
            count += batch_count
            empty += batch_count - len(slices)
            # Consume in original window order, independent of worker completion order.
            for region, prediction in zip(slices, predictions, strict=True):
                weighted[(slice(None), *region)] += taper[None] * prediction
                weights[region] += taper
            if reporter is not None and (
                count == block_total or count // 100 != (count - batch_count) // 100
            ):
                reporter(f"DRR blocks {count}/{block_total}")
    covered = weights > 0
    np.divide(weighted, weights[None], out=weighted, where=covered[None])
    result = weighted.astype(values.dtype)
    result[:, mask] = values[:, mask]
    return WindowedDrrResult(result, count, empty, int((~covered).sum()))
