"""Invertible time alignment along the receiver-y grid."""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def validate_time_alignment(value: object) -> dict[str, object]:
    """Validate the explicit receiver-y alignment and boundary contract."""
    if not isinstance(value, Mapping) or set(value) != {
        "receiver_y_shift_samples_per_cell",
        "boundary",
    }:
        raise ValueError("time_alignment requires receiver_y_shift_samples_per_cell and boundary")
    boundary = value["boundary"]
    if boundary not in ("circular", "zero_pad", "fourier_periodic"):
        raise ValueError("time_alignment.boundary must be circular, zero_pad or fourier_periodic")
    shift = value["receiver_y_shift_samples_per_cell"]
    if boundary == "fourier_periodic":
        if (
            isinstance(shift, bool)
            or not isinstance(shift, Real)
            or not math.isfinite(shift)
            or shift == 0
        ):
            raise ValueError("receiver_y_shift_samples_per_cell must be finite and nonzero")
        shift = float(shift)
    else:
        if isinstance(shift, bool) or not isinstance(shift, Integral) or shift == 0:
            raise ValueError("receiver_y_shift_samples_per_cell must be a nonzero integer")
        shift = int(shift)
    return {"receiver_y_shift_samples_per_cell": shift, "boundary": boundary}


def alignment_time_padding(receiver_y_count: int, alignment: Mapping[str, object]) -> int:
    """Return zero padding rounded to the NeRSI decoder's eight-sample grid."""
    options = validate_time_alignment(alignment)
    if (
        isinstance(receiver_y_count, bool)
        or not isinstance(receiver_y_count, Integral)
        or receiver_y_count < 1
    ):
        raise ValueError("receiver_y_count must be a positive integer")
    if options["boundary"] != "zero_pad":
        return 0
    span = abs(options["receiver_y_shift_samples_per_cell"]) * (receiver_y_count - 1)
    return ((span + 7) // 8) * 8


def align_receiver_y_time(
    values: np.ndarray, alignment: Mapping[str, object], *, inverse: bool = False
) -> np.ndarray:
    """Align traces by ``step * (local_y - floor(ny / 2))`` time samples.

    Time is axis zero and receiver-y is the last axis of a five-dimensional
    volume. Circular boundaries preserve every sample and trace energy exactly;
    they impose a periodic endpoint condition. Zero padding instead adds the
    shift span rounded up to eight samples and offsets every shift by the minimum
    shift. Inversion removes only that padding; original samples are never cropped.
    Fourier-periodic alignment rotates resolved Fourier phases in float64 and
    leaves DC and the real Nyquist coefficient unchanged, preserving trace energy
    and invertibility up to the output dtype's rounding.
    """
    options = validate_time_alignment(alignment)
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 5
        or not values.size
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
    ):
        raise ValueError("alignment values must be a finite nonempty floating 5D volume")
    if not isinstance(inverse, bool):
        raise ValueError("inverse must be boolean")
    step = options["receiver_y_shift_samples_per_cell"]
    shifts = [step * (y - values.shape[-1] // 2) for y in range(values.shape[-1])]
    if options["boundary"] == "fourier_periodic":
        result = np.empty_like(values)
        frequencies = np.fft.rfftfreq(values.shape[0])
        for receiver_y, shift in enumerate(shifts):
            signed_shift = -shift if inverse else shift
            phase = np.exp(-2j * np.pi * frequencies * signed_shift)
            if values.shape[0] % 2 == 0:
                # A real Nyquist coefficient cannot take an arbitrary unitary phase.
                phase[-1] = 1
            spectrum = np.fft.rfft(values[..., receiver_y].astype(np.float64), axis=0)
            spectrum *= phase[:, None, None, None]
            result[..., receiver_y] = np.fft.irfft(spectrum, n=values.shape[0], axis=0)
        return np.ascontiguousarray(result)
    if options["boundary"] == "zero_pad":
        padding = alignment_time_padding(values.shape[-1], options)
        time_count = values.shape[0] - padding if inverse else values.shape[0]
        if time_count < 1 or time_count % 8:
            raise ValueError(
                "zero_pad alignment requires a positive physical time count divisible by 8"
            )
        output_count = time_count if inverse else time_count + padding
        result = np.zeros((output_count, *values.shape[1:]), dtype=values.dtype)
        minimum_shift = min(shifts)
        for receiver_y, shift in enumerate(shifts):
            offset = shift - minimum_shift
            if inverse:
                result[..., receiver_y] = values[offset : offset + time_count, ..., receiver_y]
            else:
                result[offset : offset + time_count, ..., receiver_y] = values[..., receiver_y]
        return result
    result = np.empty_like(values)
    for receiver_y, shift in enumerate(shifts):
        result[..., receiver_y] = np.roll(
            values[..., receiver_y], -shift if inverse else shift, axis=0
        )
    return np.ascontiguousarray(result)
