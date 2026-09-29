"""Batched double-precision POCS for parameter searches on CPU or CUDA."""

from itertools import islice, product

import numpy as np
import torch

from seis_interp.processing.pocs import _validated_block_inputs, _validated_pocs_parameters
from seis_interp.processing.pocs_windows import (
    WindowedPocsResult,
    _axis_window_starts,
    _positive_hann_weights,
    _validated_window_spec,
)


def _reconstruct_blocks(values, masks, iterations, start, end, device):
    """Batch independent windows and frequencies, keeping their thresholds separate."""
    if torch.device(device).type == "cpu":
        return _reconstruct_numpy_blocks(values, masks, iterations, start, end)
    observed = torch.as_tensor(np.stack(values), dtype=torch.float64, device=device)
    mask = torch.as_tensor(np.stack(masks), device=device).unsqueeze(1)
    observed = torch.where(mask, observed, 0.0)
    frequency = torch.fft.rfft(observed, dim=1, norm="ortho")
    reconstructed = frequency
    axes = (2, 3, 4, 5)
    spectrum = torch.fft.fftn(frequency, dim=axes, norm="ortho")
    maximum = spectrum.abs().amax(dim=axes, keepdim=True)
    for iteration, ratio in enumerate(np.geomspace(start, end, iterations)):
        if iteration:
            spectrum = torch.fft.fftn(reconstructed, dim=axes, norm="ortho")
        spectrum = torch.where(spectrum.abs() < float(ratio) * maximum, 0.0, spectrum)
        reconstructed = torch.fft.ifftn(spectrum, dim=axes, norm="ortho")
        reconstructed = torch.where(mask, frequency, reconstructed)
    reconstructed[:, 0] = reconstructed[:, 0].real.clone()
    if observed.shape[1] % 2 == 0:
        reconstructed[:, -1] = reconstructed[:, -1].real.clone()
    result = torch.fft.irfft(reconstructed, n=observed.shape[1], dim=1, norm="ortho")
    result = torch.where(mask, observed, result)
    return result.cpu().numpy().astype(values[0].dtype)


def _reconstruct_numpy_blocks(values, masks, iterations, start, end):
    """Keep the reference FFT implementation for final CPU verification."""
    mask = np.stack(masks)[:, None]
    observed = np.where(mask, np.stack(values).astype(np.float64), 0.0)
    frequency = np.fft.rfft(observed, axis=1, norm="ortho")
    reconstructed = frequency
    axes = (2, 3, 4, 5)
    spectrum = np.fft.fftn(frequency, axes=axes, norm="ortho")
    maximum = np.abs(spectrum).max(axis=axes, keepdims=True)
    for iteration, ratio in enumerate(np.geomspace(start, end, iterations)):
        if iteration:
            spectrum = np.fft.fftn(reconstructed, axes=axes, norm="ortho")
        spectrum[np.abs(spectrum) < ratio * maximum] = 0.0
        reconstructed = np.fft.ifftn(spectrum, axes=axes, norm="ortho")
        reconstructed = np.where(mask, frequency, reconstructed)
    reconstructed[:, 0] = reconstructed[:, 0].real
    if observed.shape[1] % 2 == 0:
        reconstructed[:, -1] = reconstructed[:, -1].real
    result = np.fft.irfft(reconstructed, n=observed.shape[1], axis=1, norm="ortho")
    return np.where(mask, observed, result).astype(values[0].dtype)


def interpolate_pocs_volume_batched(
    observed,
    observed_trace_mask,
    *,
    n_iterations,
    threshold_start,
    threshold_end,
    window_shape,
    overlap,
    device="cpu",
    batch_sample_limit=2_000_000,
):
    """Use the existing POCS schedule and positive-Hann overlap assembly in batches.

    This opt-in search implementation leaves the reference NumPy path unchanged.
    No target values, frequency filtering, padding, or analysis taper are used.
    """
    values, mask = _validated_block_inputs(observed, observed_trace_mask)
    iterations, start, end = _validated_pocs_parameters(
        n_iterations, threshold_start, threshold_end
    )
    window, window_overlap = _validated_window_spec(window_shape, overlap)
    if not isinstance(batch_sample_limit, int) or batch_sample_limit < 1:
        raise ValueError("batch_sample_limit must be a positive integer")
    if window is None:
        window, window_overlap = values.shape, (0,) * 5
    window = tuple(min(length, width) for length, width in zip(values.shape, window, strict=True))
    starts = [
        _axis_window_starts(length, width, shared)
        for length, width, shared in zip(values.shape, window, window_overlap, strict=True)
    ]
    blocks = iter(product(*starts))
    batch_size = max(1, batch_sample_limit // int(np.prod(window)))
    weighted = np.zeros(values.shape, dtype=np.float64)
    weights = np.zeros(values.shape, dtype=np.float64)
    taper = _positive_hann_weights(window)
    count = empty = 0
    while locations := list(islice(blocks, batch_size)):
        slices, batch_values, batch_masks = [], [], []
        for location in locations:
            count += 1
            region = tuple(slice(a, a + b) for a, b in zip(location, window, strict=True))
            local_mask = mask[region[1:]]
            if not local_mask.any():
                empty += 1
                continue
            slices.append(region)
            batch_values.append(values[region])
            batch_masks.append(local_mask)
        if not slices:
            continue
        predictions = _reconstruct_blocks(batch_values, batch_masks, iterations, start, end, device)
        for region, prediction in zip(slices, predictions, strict=True):
            weighted[region] += taper * prediction
            weights[region] += taper
    covered = weights > 0
    np.divide(weighted, weights, out=weighted, where=covered)
    result = weighted.astype(values.dtype)
    result[:, mask] = values[:, mask]
    return WindowedPocsResult(result, count, empty, int(np.count_nonzero(~covered)))
