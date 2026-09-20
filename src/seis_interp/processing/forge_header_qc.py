"""Conservative header-only classification for the FORGE correlated gathers."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.forge_headers import TRACE_FIELDS
from seis_interp.processing.geometry import apply_coordinate_scalar

TRACE_CODE_LABELS = {
    1: "seismic_data",
    2: "dead",
    3: "dummy",
    4: "time_break",
    5: "uphole",
    6: "sweep",
    7: "timing",
    8: "water_break",
}
STATION_FIELDS = {
    "source_line": "vendor_189_raw",
    "source_point": "vendor_193_raw",
    "receiver_line": "vendor_181_raw",
    "receiver_point": "vendor_185_raw",
}


def classify_forge_headers(traces: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Keep all rows; mark only code-1 rows with consistent headers as candidates."""
    table = traces.copy()
    code = table.trace_identification_code
    table["trace_code_meaning"] = code.map(TRACE_CODE_LABELS).fillna("optional_or_unknown")
    table["trace_class"] = np.select(
        [code.eq(1), code.eq(2), code.eq(3), code.isin([4, 5, 6, 7, 8])],
        ["seismic_candidate", "dead", "dummy", "auxiliary"],
        default="unknown_code",
    )
    for column, raw in STATION_FIELDS.items():
        table[column] = (table[raw] / 100).where(table[raw].mod(100).eq(0))
    unit_factor = {1: 1.0, 2: 0.3048}.get(metadata["measurement_system"], np.nan)
    length_units = table.coordinate_units.eq(1)
    for role in ("source", "receiver"):
        for axis in ("x", "y"):
            name = f"{role}_{axis}"
            table[name + "_m"] = (
                apply_coordinate_scalar(table[name + "_raw"], table.coordinate_scalar) * unit_factor
            )
            table.loc[~length_units, name + "_m"] = np.nan
        table[role + "_elevation_m"] = (
            apply_coordinate_scalar(table[role + "_elevation_raw"], table.elevation_scalar)
            * unit_factor
        )
        table[role + "_xy_zero"] = table[role + "_x_raw"].eq(0) & table[role + "_y_raw"].eq(0)
    allowed_scalars = [0, 1, -1, 10, -10, 100, -100, 1000, -1000, 10000, -10000]
    checks = {
        "unsupported_coordinate_units": ~length_units | ~np.isfinite(unit_factor),
        "invalid_coordinate_scalar": ~table.coordinate_scalar.isin(allowed_scalars),
        "source_xy_zero": table.source_xy_zero,
        "receiver_xy_zero": table.receiver_xy_zero,
        "sample_count_mismatch": table.sample_count.ne(metadata["binary_sample_count"]),
        "sample_interval_mismatch": table.sample_interval_us.ne(
            metadata["binary_sample_interval_us"]
        ),
        "nonproduction_data": table.data_use.ne(1),
        "nonpositive_trace_number": table.trace_number.le(0),
        "invalid_station_id": table[list(STATION_FIELDS)].isna().any(axis=1)
        | table[list(STATION_FIELDS)].le(0).any(axis=1),
    }
    table["header_issues"] = ""
    for reason, mask in checks.items():
        table.loc[mask, "header_issues"] += reason + ";"
    table["header_issues"] = table.header_issues.str.rstrip(";")
    table["header_eligible"] = code.eq(1) & table.header_issues.eq("")
    table["waveform_qc_status"] = "not_performed"
    table["component_status"] = "not_resolved_by_header"
    return table


def attach_navigation(
    table: pd.DataFrame, navigation: pd.DataFrame, role: str, tolerance_m: float
) -> pd.DataFrame:
    """Join station IDs without replacing coordinates or filling missing receivers."""
    nav = navigation.rename(
        columns={
            "line": role + "_line",
            "point": role + "_point",
            "x_m": role + "_nav_x_m",
            "y_m": role + "_nav_y_m",
            "elevation_m": role + "_nav_elevation_m",
            "file_line_number": role + "_nav_file_line_number",
        }
    )
    result = table.merge(
        nav, how="left", on=[role + "_line", role + "_point"], validate="many_to_one", sort=False
    )
    found = result[role + "_nav_x_m"].notna()
    valid = (
        ~result[role + "_xy_zero"] & result[role + "_x_m"].notna() & result[role + "_y_m"].notna()
    )
    distance = np.hypot(
        result[role + "_x_m"] - result[role + "_nav_x_m"],
        result[role + "_y_m"] - result[role + "_nav_y_m"],
    ).where(found & valid)
    result[role + "_nav_distance_m"] = distance
    result[role + "_nav_status"] = np.select(
        [~found, ~valid, distance.gt(tolerance_m)],
        ["not_listed", "header_coordinates_invalid", "coordinate_mismatch"],
        default="matched",
    )
    mismatch = result[role + "_nav_status"].eq("coordinate_mismatch")
    result.loc[mismatch, "header_eligible"] = False
    result.loc[mismatch, "header_issues"] = (
        result.loc[mismatch, "header_issues"] + ";" + role + "_navigation_mismatch"
    ).str.strip(";")
    return result


def mark_duplicate_headers(table: pd.DataFrame) -> pd.DataFrame:
    """Retain and quarantine ambiguous duplicates, including across files."""
    result = table.copy()
    result["duplicate_ffid_trace"] = result.duplicated(["ffid", "trace_number"], keep=False)
    result["duplicate_seismic_station_pair"] = False
    seismic = result.trace_identification_code.eq(1)
    result.loc[seismic, "duplicate_seismic_station_pair"] = result.loc[seismic].duplicated(
        list(STATION_FIELDS), keep=False
    )
    for column in ("duplicate_ffid_trace", "duplicate_seismic_station_pair"):
        mask = result[column]
        result.loc[mask, "header_eligible"] = False
        result.loc[mask, "header_issues"] = (
            result.loc[mask, "header_issues"] + ";" + column
        ).str.strip(";")
    return result


def summarize_header_columns(table: pd.DataFrame, columns: list[str]) -> dict:
    """Bounded profiles; exact raw values remain in the trace table."""
    profiles = {}
    for name in columns:
        values = table[name].dropna()
        counts = values.value_counts().sort_index()
        profile = {"unique_count": len(counts), "missing_count": int(table[name].isna().sum())}
        if pd.api.types.is_numeric_dtype(values):
            profile.update(
                minimum=float(values.min()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None,
            )
        if len(counts) <= 32:
            profile["counts"] = {str(key): int(value) for key, value in counts.items()}
        profiles[name] = profile
    return profiles


def check_forge_file_headers(table: pd.DataFrame, metadata: dict, path: Path) -> list[str]:
    issues = []
    fields = path.stem.split("_")
    if len(fields) != 7 or not all(value.isdigit() for value in fields):
        issues.append("unrecognized_filename")
    else:
        for column, expected in zip(
            ["ffid", "vendor_189_raw", "vendor_193_raw"], map(int, fields[1:4]), strict=True
        ):
            if not table[column].eq(expected).all():
                issues.append("filename_" + column + "_mismatch")
    for column in ["ffid", "source_x_raw", "source_y_raw", "vendor_189_raw", "vendor_193_raw"]:
        if table[column].nunique() != 1:
            issues.append("multiple_" + column)
    if metadata["binary_data_trace_count"] + metadata["binary_aux_trace_count"] != len(table):
        issues.append("binary_trace_count_mismatch")
    if "LINE AND POINT NUMBERS ARE X 100" not in metadata["text_header"]:
        issues.append("station_scaling_text_not_confirmed")
    if metadata["endian"] == "little" and "Little Endian" not in metadata["text_header"]:
        issues.append("endian_text_not_confirmed")
    for pattern, expected, label in [
        (r"SAMPLE INTERVAL:\s*(\d+)\s*usec", metadata["binary_sample_interval_us"], "interval"),
        (r"SAMPLES/TRACE:\s*(\d+)", metadata["binary_sample_count"], "samples"),
        (r"TRACE/RECORD:\s*(\d+)", len(table), "traces"),
    ]:
        match = re.search(pattern, metadata["text_header"])
        if match and int(match[1]) != expected:
            issues.append("text_binary_" + label + "_mismatch")
    return issues


def summarize_forge_audit(
    table: pd.DataFrame, files: list[dict], expected_range: list[int]
) -> dict:
    ffids = set(map(int, table.ffid.unique()))
    counts = table.groupby(["trace_identification_code", "trace_class"]).size()
    candidates = table[table.trace_identification_code.eq(1)]
    return {
        "file_count": len(files),
        "trace_count": len(table),
        "ffid_count": len(ffids),
        "missing_ffids_in_expected_range": sorted(
            set(range(expected_range[0], expected_range[1] + 1)) - ffids
        ),
        "unexpected_ffids": sorted(ffids - set(range(expected_range[0], expected_range[1] + 1))),
        "header_eligible_count": int(table.header_eligible.sum()),
        "classification": [
            {"code": int(code), "class": label, "count": int(count)}
            for (code, label), count in counts.items()
        ],
        "seismic_header_issue_counts": candidates.header_issues.str.split(";")
        .explode()
        .loc[lambda s: s.ne("")]
        .value_counts()
        .astype(int)
        .to_dict(),
        "all_trace_header_issue_counts": table.header_issues.str.split(";")
        .explode()
        .loc[lambda s: s.ne("")]
        .value_counts()
        .astype(int)
        .to_dict(),
        "file_issue_counts": pd.Series(
            [issue for row in files for issue in row["issues"]], dtype=str
        )
        .value_counts()
        .astype(int)
        .to_dict(),
        "profiles": summarize_header_columns(
            table, list(TRACE_FIELDS) + ["trace_class", "source_nav_status", "receiver_nav_status"]
        ),
        "navigation": {
            role: {
                "seismic_status_counts": candidates[role + "_nav_status"].value_counts().to_dict(),
                "max_valid_coordinate_difference_m": (
                    float(candidates[role + "_nav_distance_m"].max())
                    if candidates[role + "_nav_distance_m"].notna().any()
                    else None
                ),
            }
            for role in ("source", "receiver")
        },
        "limitations": [
            "Header eligibility is provisional; no amplitude or waveform QC was performed.",
            "Trace code 1 does not resolve component, sensor type or polarity.",
            "Codes >=9 are optional/vendor-specific in revision 0; meanings are not inferred.",
            "Station IDs at bytes 181/185/189/193 are an inferred FORGE mapping "
            "checked against navigation.",
            "Missing navigation records do not imply missing seismic traces; "
            "coordinates are not filled.",
            "XLS and ZIP are inventoried and hashed, not parsed as acquisition documentation.",
        ],
    }


def summarize_forge_stations(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Summarize station availability; do not fill coordinates of dead traces."""
    seismic = table[table.trace_identification_code.eq(1)]
    receiver_counts = seismic.groupby(["receiver_line", "receiver_point"]).agg(
        seismic_shot_count=("ffid", "nunique"),
        x_unique=("receiver_x_m", "nunique"),
        y_unique=("receiver_y_m", "nunique"),
        x_m=("receiver_x_m", "first"),
        y_m=("receiver_y_m", "first"),
    )
    receiver_domain = table[table.trace_number.gt(0)][
        ["receiver_line", "receiver_point"]
    ].drop_duplicates()
    receiver_summary = receiver_domain.merge(
        receiver_counts.reset_index(), how="left", on=["receiver_line", "receiver_point"]
    )
    receiver_summary["seismic_shot_count"] = receiver_summary.seismic_shot_count.fillna(0).astype(
        int
    )
    source_stations = table[
        ["ffid", "source_line", "source_point", "source_x_m", "source_y_m"]
    ].drop_duplicates()
    counts = {
        "source_lines": source_stations.groupby("source_line").size().to_dict(),
        "receiver_station_count": len(receiver_domain),
        "receiver_line_count": int(receiver_domain.receiver_line.nunique()),
        "seismic_receiver_station_count": len(receiver_counts),
        "seismic_receivers_in_every_ffid": int(
            receiver_counts.seismic_shot_count.eq(table.ffid.nunique()).sum()
        ),
        "receiver_stations_with_varying_coordinates": int(
            (receiver_counts.x_unique.gt(1) | receiver_counts.y_unique.gt(1)).sum()
        ),
    }
    return source_stations, receiver_summary, counts
