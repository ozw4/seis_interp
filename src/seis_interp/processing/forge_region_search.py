"""Deterministic local orthogonal lattices and QC-aware FORGE region comparison."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StationRegion:
    """A finite grid and every known station in its declared line/along window."""

    metrics: dict
    stations: pd.DataFrame


def along_direction(stations: pd.DataFrame) -> np.ndarray:
    """Median axial direction of consecutive points, with deterministic orientation."""
    vectors = []
    for _, group in stations.sort_values(["line", "point"]).groupby("line"):
        delta = np.diff(group[["x_m", "y_m"]].to_numpy(), axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        vectors.extend(delta[lengths > 0] / lengths[lengths > 0, None])
    if not vectors:
        raise ValueError("not enough distinct consecutive stations")
    vectors = np.asarray(vectors)
    # Axial directions are unchanged when acquisition point numbering reverses.
    reference = vectors[0]
    vectors *= np.where(vectors @ reference < 0, -1, 1)[:, None]
    direction = np.median(vectors, axis=0)
    direction /= np.linalg.norm(direction)
    if direction[np.argmax(abs(direction))] < 0:
        direction *= -1
    return direction


def lattice_phase(values: np.ndarray, spacing: float, config: dict) -> float:
    """Minimize squared periodic distances without assigning independent line shifts."""

    def loss(phases):
        residual = (values[:, None] - phases + spacing / 2) % spacing - spacing / 2
        return np.mean(residual**2, axis=0)

    phases = np.arange(0, spacing, config["phase_step_m"])
    best = phases[np.argmin(loss(phases))]
    phases = best + np.arange(
        -config["phase_step_m"],
        config["phase_step_m"] + config["phase_refinement_m"] / 2,
        config["phase_refinement_m"],
    )
    return float(phases[np.argmin(loss(phases))] % spacing)


def fit_station_region(block, role, shape, center_xy, seed_direction, config):
    """Fit one shared orthogonal grid; preserve curved/outlying stations in the crop.

    Membership is all known points on the line block inside the final along-axis
    half-open cell envelope. Across-axis outliers remain and map to the nearest
    finite grid node, including edge nodes; their complete displacement is counted.
    """
    n_line, n_point = shape
    spacing = config["along_spacing_m"]
    origin = np.asarray(config["origin_xy_m"])
    xy = block[["x_m", "y_m"]].to_numpy() - origin
    center = np.asarray(center_xy) - origin
    seed_u = (xy - center) @ seed_direction
    seed = block[(seed_u >= -n_point * spacing / 2) & (seed_u < n_point * spacing / 2)]
    if seed.line.nunique() != n_line or len(seed) < 2 * n_line:
        return None
    direction = along_direction(seed)
    normal = np.array([-direction[1], direction[0]])
    u, v = xy @ direction, xy @ normal
    center_u = center @ direction
    seed_mask = (u >= center_u - n_point * spacing / 2) & (u < center_u + n_point * spacing / 2)
    if not seed_mask.any():
        return None
    phase = lattice_phase(u[seed_mask], spacing, config)
    proposed_first = center_u - (n_point - 1) * spacing / 2
    first_u = phase + np.floor((proposed_first - phase) / spacing + 0.5) * spacing
    lower, upper = first_u - spacing / 2, first_u + (n_point - 0.5) * spacing
    selected = (u >= lower) & (u < upper)
    points = block[selected].copy().reset_index(drop=True)
    if points.line.nunique() != n_line:
        return None
    u, v = u[selected], v[selected]
    centers = pd.Series(v).groupby(points.line).median().sort_values()
    ranks = np.arange(n_line)
    across = (
        float(np.polyfit(ranks, centers.to_numpy(), 1)[0])
        if role == "source"
        else float(config["receiver_across_spacing_m"])
    )
    if not np.isfinite(across) or across <= 0:
        return None
    first_v = float(np.mean(centers.to_numpy() - ranks * across))
    raw_i = np.floor((v - first_v) / across + 0.5).astype(int)
    i = np.clip(raw_i, 0, n_line - 1)
    j = np.floor((u - first_u) / spacing + 0.5).astype(int)
    if (j < 0).any() or (j >= n_point).any():
        raise ValueError("along membership disagrees with finite grid envelope")
    grid_u, grid_v = first_u + j * spacing, first_v + i * across
    grid_xy = origin + grid_u[:, None] * direction + grid_v[:, None] * normal
    distance = np.hypot(u - grid_u, v - grid_v)
    normalized = np.hypot((u - grid_u) / spacing, (v - grid_v) / across)
    points["grid_line"], points["grid_point"] = i, j
    points["cell"] = i * n_point + j
    points["grid_x_m"], points["grid_y_m"] = grid_xy.T
    points["distance_m"], points["normalized_distance"] = distance, normalized
    points["edge_clamped"] = raw_i != i
    counts = points.cell.value_counts()
    metrics = {
        "role": role,
        "line_ids": sorted(int(x) for x in block.line.unique()),
        "physical_line_order": [int(x) for x in centers.index],
        "shape": list(shape),
        "station_count": len(points),
        "cell_count": n_line * n_point,
        "fill_fraction": len(counts) / (n_line * n_point),
        "collision_fraction": float(counts[counts > 1].sum() / len(points)),
        "max_cell_count": int(counts.max()),
        "distance_median_m": float(np.median(distance)),
        "distance_p95_m": float(np.quantile(distance, 0.95)),
        "distance_max_m": float(distance.max()),
        "normalized_p95": float(np.quantile(normalized, 0.95)),
        "edge_clamped_count": int(points.edge_clamped.sum()),
        "direction": direction.tolist(),
        "normal": normal.tolist(),
        "first_u_m": float(first_u),
        "first_v_m": first_v,
        "along_lower_m": float(lower),
        "along_upper_m": float(upper),
        "along_spacing_m": spacing,
        "across_spacing_m": across,
        "line_spacing_min_m": float(np.diff(centers).min()),
        "line_spacing_max_m": float(np.diff(centers).max()),
        "angle_degrees": float(np.degrees(np.arctan2(direction[1], direction[0]))),
        "area_km2": n_line * across * n_point * spacing / 1e6,
        "origin_xy_m": origin.tolist(),
    }
    return StationRegion(metrics, points)


def diverse_station_shortlist(regions, count):
    """Round-robin three rankings, suppressing identical station memberships only."""
    rankings = [
        sorted(
            regions,
            key=lambda x: (
                x.metrics[key] * sign,
                -x.metrics["fill_fraction"],
                x.metrics["normalized_p95"],
                x.metrics["collision_fraction"],
                x.metrics["candidate_id"],
            ),
        )
        for key, sign in [("fill_fraction", -1), ("normalized_p95", 1), ("collision_fraction", 1)]
    ]
    selected, seen = [], set()
    for position in range(len(regions)):
        for ranking in rankings:
            region = ranking[position]
            membership = tuple(sorted(region.stations.station_id))
            if membership not in seen:
                seen.add(membership)
                selected.append(region)
                if len(selected) == count:
                    return selected
    return selected


def search_station_regions(stations, role, shape, config):
    """Coarse scan plus local 50 m refinement; no quality-dependent point removal."""
    stations = stations.rename(
        columns={f"{role}_{key}": key for key in ["line", "point", "x_m", "y_m"]}
        | {f"{role}_id": "station_id"}
    ).dropna(subset=["x_m", "y_m"])
    lines = sorted(stations.line.unique())
    n_line, n_point = shape
    origin = np.asarray(config["origin_xy_m"])
    regions, blocks, visited = [], {}, set()

    def visit(block_index, center_u, stage):
        key = (block_index, round(float(center_u), 6))
        if key in visited:
            return
        visited.add(key)
        block, direction, center_v = blocks[block_index]
        normal = np.array([-direction[1], direction[0]])
        center = origin + direction * center_u + normal * center_v
        region = fit_station_region(block, role, shape, center, direction, config)
        if region is not None:
            region.metrics.update(
                candidate_id=f"{role}_b{block_index:02d}_u{center_u:.1f}",
                block_index=block_index,
                scan_center_m=float(center_u),
                stage=stage,
            )
            regions.append(region)

    for start in range(len(lines) - n_line + 1):
        block_lines = lines[start : start + n_line]
        if not np.all(np.diff(block_lines) == config["line_id_steps"][role]):
            continue
        block = stations[stations.line.isin(block_lines)]
        direction = along_direction(block)
        xy = block[["x_m", "y_m"]].to_numpy() - origin
        u = xy @ direction
        center_v = float(np.median(xy @ np.array([-direction[1], direction[0]])))
        blocks[start] = block, direction, center_v
        # Crop centers are anchored to a common metric origin, not station IDs.
        half = n_point * config["along_spacing_m"] / 2
        low = np.ceil((u.min() + half - config["along_spacing_m"] / 2) / config["coarse_step_m"])
        high = np.floor((u.max() - half + config["along_spacing_m"] / 2) / config["coarse_step_m"])
        for center in np.arange(low, high + 1) * config["coarse_step_m"]:
            visit(start, center, "coarse")
    seeds = diverse_station_shortlist(regions, config["refinement_seeds"])
    for seed in seeds:
        for delta in np.arange(
            -config["coarse_step_m"],
            config["coarse_step_m"] + config["fine_step_m"] / 2,
            config["fine_step_m"],
        ):
            visit(seed.metrics["block_index"], seed.metrics["scan_center_m"] + delta, "fine")
    return regions, diverse_station_shortlist(regions, config["station_shortlist"])


def pair_cell_counts(source, receiver, status):
    """Count recorded and usable observations on the finite 4D grid, never average."""
    s, r = source.stations, receiver.stations
    sub = status[np.ix_(s.station_id, r.station_id)]
    cells = (
        s.cell.to_numpy()[:, None] * receiver.metrics["cell_count"] + r.cell.to_numpy()
    ).ravel()
    n = source.metrics["cell_count"] * receiver.metrics["cell_count"]
    retained = sub.ravel() >= 3
    available = np.bincount(cells[retained], minlength=n)
    recorded = np.bincount(cells[sub.ravel() > 0], minlength=n)
    support = np.bincount(cells, minlength=n)
    return sub, cells, available, recorded, support


def compare_region_pair(source, receiver, status, config, samples):
    sub, _, counts, recorded, support = pair_cell_counts(source, receiver, status)
    n = len(counts)
    usable = int(counts.sum())
    collision = float(counts[counts > 1].sum() / usable) if usable else 0.0
    p95 = max(source.metrics["normalized_p95"], receiver.metrics["normalized_p95"])
    fill = float(np.count_nonzero(counts) / n)
    limit = config["limits"]
    return {
        "source_candidate": source.metrics["candidate_id"],
        "receiver_candidate": receiver.metrics["candidate_id"],
        "grid_cells": n,
        "retained_traces": usable,
        "fill_fraction": fill,
        "collision_fraction": collision,
        "max_cell_count": int(counts.max()),
        "normalized_p95": p95,
        "source_p95_m": source.metrics["distance_p95_m"],
        "receiver_p95_m": receiver.metrics["distance_p95_m"],
        "source_fill": source.metrics["fill_fraction"],
        "receiver_fill": receiver.metrics["fill_fraction"],
        "structural_empty_cells": int(np.count_nonzero(support == 0)),
        "unrecorded_supported_cells": int(np.count_nonzero((support > 0) & (recorded == 0))),
        "qc_empty_cells": int(np.count_nonzero((recorded > 0) & (counts == 0))),
        "header_excluded_traces": int(np.count_nonzero(sub == 1)),
        "fixed_qc_excluded_traces": int(np.count_nonzero(sub == 2)),
        "review_fraction": float(np.count_nonzero(sub == 3) / usable) if usable else None,
        "dense_float32_bytes": n * samples * 4,
        "meets_guidelines": bool(
            usable
            and p95 <= limit["normalized_p95"]
            and fill >= limit["fill_fraction"]
            and collision <= limit["collision_fraction"]
        ),
    }


def rank_region_pairs(table):
    return table.sort_values(
        [
            "meets_guidelines",
            "fill_fraction",
            "normalized_p95",
            "collision_fraction",
            "candidate_id",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_finalists(table, source_regions, receiver_regions, status, config):
    """Exact retained-pair Jaccard, using separable station sets and the QC matrix."""
    selected = []
    for row in rank_region_pairs(table).to_dict("records"):
        source = set(source_regions[row["source_candidate"]].stations.station_id)
        receiver = set(receiver_regions[row["receiver_candidate"]].stations.station_id)
        duplicate = False
        for prior, ps, pr in selected:
            a, b = sorted(source & ps), sorted(receiver & pr)
            intersection = int(np.count_nonzero(status[np.ix_(a, b)] >= 3))
            union = row["retained_traces"] + prior["retained_traces"] - intersection
            if union and intersection / union > config["maximum_jaccard"]:
                duplicate = True
                break
        if not duplicate and row["retained_traces"]:
            selected.append((row, source, receiver))
        if len(selected) == config["finalists_per_size"]:
            break
    return pd.DataFrame([row for row, _, _ in selected], columns=table.columns)


def region_trace_mapping(domain, source, receiver):
    """Keep original keys, QC, actual XY and finite-grid assignments for all records."""
    result = domain[
        domain.source_id.isin(source.stations.station_id)
        & domain.receiver_id.isin(receiver.stations.station_id)
    ].copy()
    columns = [
        "station_id",
        "x_m",
        "y_m",
        "grid_line",
        "grid_point",
        "cell",
        "grid_x_m",
        "grid_y_m",
        "distance_m",
        "normalized_distance",
        "edge_clamped",
    ]
    for role, region in [("source", source), ("receiver", receiver)]:
        result = result.rename(
            columns={f"{role}_{axis}_m": f"{role}_header_{axis}_m" for axis in ["x", "y"]}
        )
        mapping = region.stations[columns].rename(
            columns={k: f"{role}_{k}" for k in columns} | {"station_id": f"{role}_id"}
        )
        result = result.merge(mapping, on=f"{role}_id", validate="many_to_one")
    result["grid_cell"] = result.source_cell * receiver.metrics["cell_count"] + result.receiver_cell
    return result.sort_values(["source_file", "trace_index"]).reset_index(drop=True)


def pair_characteristics(source, receiver, status, flags):
    """Summarize geometry and diagnostic flags without scanning the full trace table."""
    s, r = source.stations, receiver.stations
    selection = np.ix_(s.station_id, r.station_id)
    keep = status[selection] >= 3
    dx = r.x_m.to_numpy()[None, :] - s.x_m.to_numpy()[:, None]
    dy = r.y_m.to_numpy()[None, :] - s.y_m.to_numpy()[:, None]
    offset = np.hypot(dx, dy)[keep]
    azimuth = (np.degrees(np.arctan2(dx, dy)) % 360)[keep]
    result = {
        f"{key}_fraction": float(values[selection][keep].mean()) if keep.any() else None
        for key, values in flags.items()
    }
    for name, q in [("min", 0), ("p05", 0.05), ("median", 0.5), ("p95", 0.95), ("max", 1)]:
        result[f"offset_{name}_m"] = float(np.quantile(offset, q)) if len(offset) else None
    result["azimuth_30deg_counts"] = np.histogram(azimuth[offset > 0], bins=np.arange(0, 361, 30))[
        0
    ].tolist()
    result["zero_offset_count"] = int(np.count_nonzero(offset == 0))
    return result
