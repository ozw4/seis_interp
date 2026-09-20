"""Evidence about channel multiplicity without guessing vendor component codes."""

from collections import Counter

import pandas as pd


def summarize_component_evidence(
    headers: pd.DataFrame, file_headers: pd.DataFrame, waveforms: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Retain opaque bytes 209-212 as groups, never label them Z/N/E or sensor types."""
    candidate = headers[headers.header_eligible].copy()
    candidate["vendor_bytes_209_212_hex"] = candidate.raw_header.map(lambda raw: raw[208:212].hex())
    multiplicity = candidate.groupby(["source_file", "receiver_line", "receiver_point"]).size()
    declarations = Counter()
    for raw in file_headers.raw_file_header:
        text = raw[:3200].decode("ascii", errors="replace")
        for start in range(0, 3200, 80):
            line = text[start : start + 80].strip()
            if "Geophone" in line:
                declarations[line] += 1
    joined = candidate[["source_file", "trace_index", "vendor_bytes_209_212_hex"]].merge(
        waveforms,
        on=["source_file", "trace_index"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise ValueError("waveform evidence does not cover all header candidates")
    groups = (
        joined.groupby("vendor_bytes_209_212_hex")
        .agg(
            trace_count=("trace_index", "size"),
            all_zero_count=("all_zero", "sum"),
            median_rms=("rms", "median"),
            amplitude_review_count=("amplitude_review", "sum"),
            median_sweep_power_fraction=("power_fraction_sweep_4_84", "median"),
        )
        .reset_index()
    )
    return {
        "sensor_declarations_file_counts": dict(declarations),
        "candidate_channel_multiplicity_counts": {
            str(k): int(v) for k, v in multiplicity.value_counts().items()
        },
        "orientation": "unconfirmed",
        "polarity": "unconfirmed",
        "physical_amplitude_calibration": "unconfirmed",
        "vendor_group_meanings": "unknown; raw bytes are not component labels",
        "interpretation": "One recorded channel per source/receiver station pair; "
        "this does not establish its physical component or a uniform sensor response.",
    }, groups
