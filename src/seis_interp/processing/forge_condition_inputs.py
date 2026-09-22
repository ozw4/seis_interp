"""Time filtering and nested missing-trace masks for diagnostic condition searches."""

import numpy as np

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs


def downsample_traces(values, factor):
    """Decimate each trace after a reflection-padded Hamming-windowed sinc FIR.

    The low-pass cutoff is 0.45/factor cycles per original sample, with
    20*factor+1 taps. Output sample zero stays at original time zero.
    """
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite [trace, time] values")
    if factor == 1:
        return values.astype(np.float32, copy=True)
    half = 10 * factor
    positions = np.arange(-half, half + 1)
    cutoff = 0.45 / factor
    kernel = 2 * cutoff * np.sinc(2 * cutoff * positions) * np.hamming(len(positions))
    kernel /= kernel.sum()
    nfft = 1 << (values.shape[1] + 4 * half - 1).bit_length()
    spectrum = np.fft.rfft(kernel, n=nfft)
    output = np.empty((len(values), len(range(0, values.shape[1], factor))), np.float32)
    for start in range(0, len(values), 128):
        padded = np.pad(
            values[start : start + 128].astype(float), ((0, 0), (half, half)), mode="reflect"
        )
        filtered = np.fft.irfft(np.fft.rfft(padded, n=nfft, axis=1) * spectrum, n=nfft, axis=1)
        output[start : start + 128] = filtered[:, 2 * half : 2 * half + values.shape[1] : factor]
    return output


def nested_condition_mask(mask, missing_fraction):
    """Retain the original hash ranking; smaller test sets are strict subsets."""
    if not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be between zero and one")
    result = mask.copy()
    count = round(len(result) * missing_fraction)
    if count < 1 or count >= len(result):
        raise ValueError("mask must leave both observed and test traces")
    result["split"] = np.where(result.hash_rank <= count, "test", "observed")
    return result


def condition_model_inputs(geometry, mask, values, spatial_shape, time_s, normalization, binding):
    """Return a runtime object holding only observed amplitudes, plus evaluation scales.

    Oracle test RMS is returned separately, never as a model feature or label.
    """
    if not np.array_equal(geometry.cell_id, mask.cell_id) or len(values) != len(mask):
        raise ValueError("geometry, mask and waveform rows must align")
    observed = mask.split.eq("observed").to_numpy()
    raw = values[observed].astype(np.float64)
    if normalization == "trace_oracle":
        scales = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), 128):
            scales[start : start + 128] = np.sqrt(
                np.square(values[start : start + 128].astype(float)).mean(axis=1)
            )
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("trace RMS must be finite and positive")
        raw /= scales[observed, None]
        global_rms = 1.0
    elif normalization == "observed_global":
        global_rms = float(np.sqrt(np.mean(raw**2)))
        scales = np.full(len(values), global_rms)
    else:
        raise ValueError("unknown diagnostic normalization")
    inputs = ForgeModelInputs(
        geometry=geometry,
        observed_cells=mask.loc[observed, "cell_id"].to_numpy(),
        test_cells=mask.loc[~observed, "cell_id"].to_numpy(),
        observed_values=raw.astype(np.float32),
        spatial_shape=tuple(spatial_shape),
        time_s=time_s,
        global_rms=global_rms,
        binding=binding,
    )
    return inputs, scales
