"""Artifact-only comparison using the fixed v3 mean trace SNR contract."""

import csv
import json
import math
from pathlib import Path

from seis_interp.evaluation.c3_poc_comparison import (
    SUMMARY_COLUMNS,
    read_poc_json,
    validate_poc_run_artifacts,
)

V3_COLUMNS = (
    "method",
    "status",
    "mean_trace_snr_db",
    "mean_trace_snr_status",
    "global_snr_db",
    "global_snr_status",
    *(key for key in SUMMARY_COLUMNS if key not in ("method", "status", "snr_db", "snr_status")),
)


def validate_v3_run_artifacts(method: str, run: Path, expected_lock: dict) -> dict:
    row, lock = validate_poc_run_artifacts(method, run)
    if lock != expected_lock:
        raise ValueError("v3 methods used different verified input locks")
    metadata = read_poc_json(run / "metadata.json")
    if any(
        metadata.get(key) != value
        for key, value in {
            "condition_id": "c3_random80_v3",
            "input_protocol": "c3_random80_v3",
            "primary_metric": "mean_trace_snr_db",
        }.items()
    ):
        raise ValueError("run metadata does not declare the v3 primary metric and input protocol")
    metrics = read_poc_json(run / "metrics.json")["evaluation_target"]
    if metrics.get("trace_snr_trace_count") != lock["target_trace_count"]:
        raise ValueError("mean trace SNR must include every target trace")
    counts = [
        metrics.get("trace_snr_" + name + "_count")
        for name in ("positive_infinity", "negative_infinity", "undefined")
    ]
    if (
        any(type(count) is not int or count < 0 for count in counts)
        or sum(counts) > lock["target_trace_count"]
    ):
        raise ValueError("invalid nonfinite trace SNR counts")
    positive, negative, undefined = counts
    expected_status = (
        "undefined"
        if undefined or (positive and negative)
        else "positive_infinity"
        if positive
        else "negative_infinity"
        if negative
        else "finite"
    )
    value = metrics.get("mean_trace_snr_db")
    if metrics.get("mean_trace_snr_status") != expected_status:
        raise ValueError("mean trace SNR status disagrees with trace counts")
    if expected_status == "finite":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("mean trace SNR must be finite")
    elif value is not None:
        raise ValueError("nonfinite mean trace SNR must be recorded as null with explicit status")
    row.update(
        mean_trace_snr_db=value,
        mean_trace_snr_status=expected_status,
        global_snr_db=row.pop("snr_db"),
        global_snr_status=row.pop("snr_status"),
    )
    return row


def write_v3_comparison(output: Path, summary: dict) -> None:
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        stream.write(serialized)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=V3_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["runs"])
