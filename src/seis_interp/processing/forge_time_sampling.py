"""Validate native/output FORGE sampling and apply the shared anti-alias filter."""

from numbers import Integral

from seis_interp.processing.forge_condition_inputs import downsample_traces


def native_time_contract(contract):
    """Return the SEG-Y sampling contract, retaining the declared output separately."""
    sampling = contract.get("time_sampling")
    if sampling is None:
        return contract
    keys = {"native_sample_count", "native_sample_interval_us", "decimation_factor"}
    if not isinstance(sampling, dict) or set(sampling) != keys:
        raise ValueError(
            "time_sampling requires native sample count, interval and decimation factor"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in sampling.values()
    ):
        raise ValueError("time_sampling values must be positive integers")
    count = sampling["native_sample_count"]
    interval = sampling["native_sample_interval_us"]
    factor = sampling["decimation_factor"]
    if (
        contract["time_sample_count"] != len(range(0, count, factor))
        or contract["sample_interval_us"] != interval * factor
    ):
        raise ValueError("output time sampling does not match native sampling and decimation")
    return {**contract, "time_sample_count": count, "sample_interval_us": interval}


def resample_forge_traces(values, contract):
    """Filter and decimate before computing either observed or oracle trace RMS."""
    native = native_time_contract(contract)
    if values.ndim != 2 or values.shape[1] != native["time_sample_count"]:
        raise ValueError("waveform length does not match native time sampling")
    if contract.get("time_sampling") is None:
        return values
    return downsample_traces(values, contract["time_sampling"]["decimation_factor"])
