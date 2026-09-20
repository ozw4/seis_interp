"""Station support and sparse four-dimensional occupancy for FORGE geometry EDA."""

import math

import numpy as np
import pandas as pd

from seis_interp.processing.geometry import compute_trace_geometry

STATUS_LABELS = {
    0: "no_record",
    1: "header_excluded",
    2: "waveform_unusable",
    3: "numeric_usable_review",
    4: "passed_numeric_checks",
}


def station_inventory(table: pd.DataFrame, role: str) -> pd.DataFrame:
    """Keep every recorded station ID, with coordinates from eligible headers only."""
    keys = [f"{role}_line", f"{role}_point"]
    xy = [f"{role}_x_m", f"{role}_y_m"]
    if table[keys].isna().any().any() or (table[keys] % 1).ne(0).any().any():
        raise ValueError("station IDs must be finite integers")
    stations = table[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    coords = table.loc[table.header_eligible, keys + xy].drop_duplicates()
    if coords.duplicated(keys).any():
        raise ValueError(f"varying {role} station coordinates")
    if not np.isfinite(coords[xy].to_numpy()).all():
        raise ValueError("eligible coordinates must be finite")
    stations = stations.merge(coords, on=keys, how="left", validate="one_to_one")
    stations[keys] = stations[keys].astype(int)
    stations.insert(0, f"{role}_id", np.arange(len(stations)))
    return stations


def prepare_geometry(table: pd.DataFrame) -> tuple:
    """Return recorded seismic rows, station tables and usable trace geometry.

    Auxiliary traces are outside the station-pair domain. Dead header coordinates
    remain untouched; station tables may learn a coordinate from another shot.
    """
    domain = table[table.trace_identification_code.isin([1, 2])].copy()
    keys = ["source_line", "source_point", "receiver_line", "receiver_point"]
    if domain.duplicated(keys).any():
        raise ValueError("duplicate recorded source/receiver pair")
    sources = station_inventory(domain, "source")
    receivers = station_inventory(domain, "receiver")
    for role, stations in [("source", sources), ("receiver", receivers)]:
        domain = domain.merge(
            stations[[f"{role}_id", f"{role}_line", f"{role}_point"]],
            on=[f"{role}_line", f"{role}_point"],
            validate="many_to_one",
        )
    domain["geometry_usable"] = domain.header_eligible & domain.numerically_usable
    domain["availability_code"] = np.select(
        [~domain.header_eligible, ~domain.numerically_usable, domain.waveform_status.eq("review")],
        [1, 2, 3],
        default=4,
    ).astype(np.uint8)
    observed = domain[domain.geometry_usable].copy()
    if observed.empty:
        raise ValueError("no numerically usable seismic traces")
    sx, sy, rx, ry = (
        observed[c].to_numpy() for c in ["source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m"]
    )
    mx, my, offset, reverse_azimuth = compute_trace_geometry(sx, sy, rx, ry)
    observed["midpoint_x_m"], observed["midpoint_y_m"] = mx, my
    observed["offset_x_m"], observed["offset_y_m"] = rx - sx, ry - sy
    observed["offset_m"] = offset
    # Clockwise from north, source toward receiver. Zero offset has no direction.
    observed["azimuth_deg"] = np.where(offset > 0, (reverse_azimuth + 180) % 360, np.nan)
    observed["axial_azimuth_deg"] = observed.azimuth_deg % 180
    return domain, sources, receivers, observed


def availability_matrix(domain: pd.DataFrame, n_source: int, n_receiver: int) -> np.ndarray:
    matrix = np.zeros((n_source, n_receiver), dtype=np.uint8)
    matrix[domain.source_id.to_numpy(), domain.receiver_id.to_numpy()] = domain.availability_code
    return matrix


def station_spacing(stations: pd.DataFrame, role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure successive points and adjacent observed lines at matched point IDs.

    Line distances are Euclidean matched-point separations, not a claim of
    perpendicular spacing. Gaps in station IDs remain explicit.
    """
    line, point = f"{role}_line", f"{role}_point"
    x, y = f"{role}_x_m", f"{role}_y_m"
    valid = stations.dropna(subset=[x, y]).sort_values([line, point])
    along = []
    for key, group in valid.groupby(line):
        values = group[[point, x, y]].to_numpy()
        for a, b in zip(values[:-1], values[1:], strict=True):
            along.append(
                {
                    "line": int(key),
                    "point_a": int(a[0]),
                    "point_b": int(b[0]),
                    "point_id_step": int(b[0] - a[0]),
                    "distance_m": float(np.hypot(*(b[1:] - a[1:]))),
                }
            )
    across = []
    lines = sorted(valid[line].unique())
    for a, b in zip(lines[:-1], lines[1:], strict=True):
        pair = valid[valid[line].eq(a)].merge(valid[valid[line].eq(b)], on=point)
        for _, row in pair.iterrows():
            across.append(
                {
                    "line_a": int(a),
                    "line_b": int(b),
                    "point": int(row[point]),
                    "line_id_step": int(b - a),
                    "distance_m": float(
                        np.hypot(row[x + "_y"] - row[x + "_x"], row[y + "_y"] - row[y + "_x"])
                    ),
                }
            )
    return (
        pd.DataFrame(along, columns=["line", "point_a", "point_b", "point_id_step", "distance_m"]),
        pd.DataFrame(across, columns=["line_a", "line_b", "point", "line_id_step", "distance_m"]),
    )


def fit_station_lattice(stations: pd.DataFrame, role: str) -> tuple[dict, pd.DataFrame]:
    """Fit XY = origin + line_delta * a + point_delta * b without snapping."""
    line, point = f"{role}_line", f"{role}_point"
    xy_cols = [f"{role}_x_m", f"{role}_y_m"]
    valid = stations.dropna(subset=xy_cols).copy()
    ids = valid[[line, point]].to_numpy()
    anchor = ids.min(axis=0)
    design = np.column_stack([np.ones(len(ids)), ids - anchor])
    coefficients, _, rank, _ = np.linalg.lstsq(design, valid[xy_cols].to_numpy(), rcond=None)
    if rank != 3:
        raise ValueError(f"{role} station layout does not span two ID axes")
    residual = valid[xy_cols].to_numpy() - design @ coefficients
    valid["fit_residual_x_m"], valid["fit_residual_y_m"] = residual.T
    valid["fit_residual_m"] = np.linalg.norm(residual, axis=1)
    summary = {
        "anchor_line_point": anchor.tolist(),
        "coefficients_xy_m": coefficients.tolist(),
        "residual_m": numeric_quantiles(valid.fit_residual_m),
    }
    return summary, valid


def numeric_quantiles(values) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {}
    return dict(
        zip(
            ["min", "p01", "p50", "p95", "p99", "max"],
            np.quantile(finite, [0, 0.01, 0.5, 0.95, 0.99, 1]).tolist(),
            strict=True,
        )
    )


def reference_pairs(sources: pd.DataFrame, receivers: pd.DataFrame) -> pd.DataFrame:
    """Known-station Cartesian support, including QC-excluded pairs, not new data."""
    src = sources.dropna(subset=["source_x_m", "source_y_m"])
    rec = receivers.dropna(subset=["receiver_x_m", "receiver_y_m"])
    return src.merge(rec, how="cross")


def spatial_coordinates(
    table: pd.DataFrame, representation: str, origin_xy: list, east_axis: list
) -> np.ndarray:
    """Shared orthonormal frame; offset is the full receiver-minus-source vector."""
    east = np.asarray(east_axis, dtype=float)
    east = east / np.linalg.norm(east)
    north = np.array([-east[1], east[0]])
    basis = np.column_stack([east, north])
    source = (table[["source_x_m", "source_y_m"]].to_numpy() - origin_xy) @ basis
    receiver = (table[["receiver_x_m", "receiver_y_m"]].to_numpy() - origin_xy) @ basis
    if representation == "source_receiver":
        return np.column_stack([source, receiver])
    if representation == "midpoint_offset":
        return np.column_stack([(source + receiver) / 2, receiver - source])
    raise ValueError(f"unknown representation: {representation}")


def sparse_grid_occupancy(
    reference: np.ndarray, observed: np.ndarray, widths: list, phase: float = 0.0
) -> tuple[dict, pd.DataFrame]:
    """Count sparse 4D cells; distinguish bounding-box voids from QC support loss.

    Bins are floor(coordinate / width - phase), with half-open boundaries.
    Never allocate the bounding-box tensor or average colliding traces.
    """
    widths = np.asarray(widths, dtype=float)
    if widths.shape != (4,) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("four finite positive bin widths are required")
    for coords in (reference, observed):
        if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
            raise ValueError("finite N by 4 coordinates are required")
    if not len(reference) or not len(observed):
        raise ValueError("nonempty reference and observed coordinates are required")
    ref_indices = np.floor(reference / widths - phase).astype(np.int64)
    obs_indices = np.floor(observed / widths - phase).astype(np.int64)
    columns = ["i0", "i1", "i2", "i3"]
    ref = (
        pd.DataFrame(ref_indices, columns=columns)
        .value_counts(sort=False)
        .rename("reference_pairs")
    )
    obs = (
        pd.DataFrame(obs_indices, columns=columns)
        .value_counts(sort=False)
        .rename("observed_traces")
    )
    cells = ref.to_frame().join(obs, how="outer")
    if cells.reference_pairs.isna().any() or (cells.observed_traces > cells.reference_pairs).any():
        raise ValueError("observations must be a subset of reference support")
    cells["observed_traces"] = cells.observed_traces.fillna(0).astype(int)
    counts = cells.observed_traces.to_numpy()
    shape = ref_indices.max(axis=0) - ref_indices.min(axis=0) + 1
    bbox = math.prod(map(int, shape))
    occupied = int((counts > 0).sum())
    collisions = counts[counts > 1]
    error = observed - (obs_indices + phase + 0.5) * widths
    summary = {
        "widths_m": widths.tolist(),
        "phase_bins": phase,
        "index_min": ref_indices.min(axis=0).tolist(),
        "shape": shape.tolist(),
        "bbox_cells": bbox,
        "reference_cells": len(cells),
        "observed_cells": occupied,
        "bbox_fill_fraction": occupied / bbox,
        "reference_fill_fraction": occupied / len(cells),
        "outside_reference_cells": bbox - len(cells),
        "unobserved_reference_cells": len(cells) - occupied,
        "observed_traces": len(observed),
        "collision_cells": len(collisions),
        "traces_in_collision_cells": int(collisions.sum()),
        "collision_trace_fraction": float(collisions.sum() / len(observed)),
        "extra_traces_beyond_one_per_cell": len(observed) - occupied,
        "max_traces_per_cell": int(counts.max()),
        "quantization_rms_m_per_axis": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "quantization_p95_abs_m_per_axis": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
    }
    return summary, cells.reset_index()


def station_id_occupancy(
    sources: pd.DataFrame, receivers: pd.DataFrame, observed: pd.DataFrame, steps: list
) -> dict:
    """Diagnostic ID rectangle only; ID spacings do not assert a physical grid."""
    axes = [
        sources.source_line,
        sources.source_point,
        receivers.receiver_line,
        receivers.receiver_point,
    ]
    shape = []
    for values, step in zip(axes, steps, strict=True):
        if step <= 0 or ((values - values.min()) % step).ne(0).any():
            raise ValueError("station IDs do not follow configured index steps")
        shape.append(int((values.max() - values.min()) / step) + 1)
    size = math.prod(shape)
    return {
        "steps": steps,
        "shape": shape,
        "rectangle_cells": size,
        "observed_pairs": len(observed),
        "fill_fraction": len(observed) / size,
        "recorded_station_pairs": len(sources) * len(receivers),
    }
