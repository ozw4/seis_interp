"""Nonoverlapping DRR windows with unknown padding at the crop boundaries."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np

from seis_interp.processing.drr import (
    select_drr_frequencies,
    validate_drr_block_inputs,
    validate_drr_parameters,
)
from seis_interp.processing.drr_batched import _hankel_indices, _reduce_numpy
from seis_interp.processing.drr_windows import WindowedDrrResult


def interpolate_drr_nonoverlapping(
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
    workers=1,
    batch_matrix_limit=2_000_000,
    reporter=None,
):
    """Use full-sized windows and bounded batches of the reference NumPy SVD.

    Padding is missing data, never an observation. Windows step by their full
    width and each physical cell is copied from exactly one reconstruction.
    Frequency batching changes neither the projection nor the iteration count.
    """
    values, mask = validate_drr_block_inputs(observed, observed_trace_mask, time_s)
    window = tuple(spatial_window_shape)
    if len(window) != 4 or tuple(spatial_overlap) != (0, 0, 0, 0):
        raise ValueError("nonoverlapping DRR requires four window lengths and zero overlap")
    rank, power, iterations = validate_drr_parameters(rank, damping_power, n_iterations, window)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(batch_matrix_limit, int) or batch_matrix_limit < 1:
        raise ValueError("batch_matrix_limit must be a positive integer")
    selection = select_drr_frequencies(
        time_s, frequency_min_hz=frequency_min_hz, frequency_max_hz=frequency_max_hz
    )
    maps = _hankel_indices(window)
    bins_per_batch = max(1, batch_matrix_limit // maps[0].size)
    locations = list(
        product(*(range(0, n, width) for n, width in zip(mask.shape, window, strict=True)))
    )

    def reconstruct(location):
        lengths = tuple(
            min(width, n - start)
            for start, width, n in zip(location, window, mask.shape, strict=True)
        )
        region = tuple(
            slice(start, start + length) for start, length in zip(location, lengths, strict=True)
        )
        if not mask[region].any():
            return region, None
        block_mask = np.zeros(window, dtype=bool)
        block = np.zeros((len(time_s), *window), dtype=np.float64)
        inside = tuple(slice(0, length) for length in lengths)
        block_mask[inside] = mask[region]
        block[(slice(None), *inside)] = np.where(
            mask[region][None], values[(slice(None), *region)], 0.0
        )
        frequency = np.fft.rfft(
            block.reshape(len(time_s), -1), n=selection.fft_length, axis=0, norm="ortho"
        )
        predicted = np.zeros_like(frequency)
        flat_mask = block_mask.reshape(1, -1)
        for start in range(0, len(selection.bin_indices), bins_per_batch):
            bins = selection.bin_indices[start : start + bins_per_batch]
            measured = frequency[bins]
            current = measured.copy()
            for _ in range(iterations):
                current = _reduce_numpy(current, *maps, rank, power)
                current = np.where(flat_mask, measured, current)
            predicted[bins] = current
        predicted[0] = predicted[0].real
        predicted[-1] = predicted[-1].real
        restored = np.fft.irfft(predicted, n=selection.fft_length, axis=0, norm="ortho")[
            : len(time_s)
        ].reshape(block.shape)
        restored[:, block_mask] = block[:, block_mask]
        return region, restored[(slice(None), *inside)]

    result = np.zeros_like(values)
    empty = uncovered = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, (region, prediction) in enumerate(executor.map(reconstruct, locations), 1):
            if prediction is None:
                empty += 1
                uncovered += int(mask[region].size)
            else:
                result[(slice(None), *region)] = prediction
            if reporter is not None:
                reporter(f"DRR blocks {completed}/{len(locations)}")
    result[:, mask] = values[:, mask]
    return WindowedDrrResult(result, len(locations), empty, uncovered)
