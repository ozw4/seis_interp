"""One-to-one FORGE candidate mapping, stable outer mask, and isolated geometry."""

import hashlib
import json

import numpy as np
import pandas as pd

AXES = [
    "source_grid_line",
    "source_grid_point",
    "receiver_grid_line",
    "receiver_grid_point",
]
QC_FLAGS = ["dc_review", "amplitude_review", "near_constant_review"]
METHODS = ("pocs_grid", "drr_grid", "ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid")


def grid_coordinates(indices: np.ndarray, definition: dict) -> np.ndarray:
    """Invert [line, point] indices using only the saved candidate basis."""
    u = definition["first_u_m"] + indices[:, 1] * definition["along_spacing_m"]
    v = definition["first_v_m"] + indices[:, 0] * definition["across_spacing_m"]
    return (
        np.asarray(definition["origin_xy_m"])
        + u[:, None] * np.asarray(definition["direction"])
        + v[:, None] * np.asarray(definition["normal"])
    )


def continuous_indices(xy: np.ndarray, definition: dict) -> np.ndarray:
    """Preserve actual coordinates, including subcell residuals; never round."""
    delta = xy - np.asarray(definition["origin_xy_m"])
    line = (delta @ definition["normal"] - definition["first_v_m"]) / definition["across_spacing_m"]
    point = (delta @ definition["direction"] - definition["first_u_m"]) / definition[
        "along_spacing_m"
    ]
    return np.column_stack((line, point))


def validate_mapping(mapping, stations, cells, grids, contract):
    """Fail before reading amplitudes if any canonical mapping invariant differs."""
    shape = tuple(contract["spatial_shape"])
    if contract["axis_order"] != AXES or shape != tuple(
        grids["source"]["shape"] + grids["receiver"]["shape"]
    ):
        raise ValueError("axis order or grid shape mismatch")
    size = int(np.prod(shape))
    if len(cells) != size or not np.array_equal(np.sort(cells.grid_cell), np.arange(size)):
        raise ValueError("grid cell coverage mismatch")
    if mapping.duplicated(["source_file", "trace_index"]).any():
        raise ValueError("duplicate original trace key")
    eligible = mapping.loc[mapping.eligible_after_fixed_qc].copy()
    if len(eligible) != contract["eligible_trace_count"]:
        raise ValueError("eligible trace count mismatch")
    if eligible.grid_cell.duplicated().any():
        raise ValueError("eligible cell collision")
    if size - len(eligible) != contract["natural_missing_cell_count"]:
        raise ValueError("natural missing count mismatch")
    if (
        not eligible.geometry_usable.all()
        or not eligible.sample_count.eq(contract["time_sample_count"]).all()
        or not eligible.sample_interval_us.eq(contract["sample_interval_us"]).all()
    ):
        raise ValueError("geometry or time contract mismatch")
    if eligible.fixed_qc_excluded.any() or not eligible.header_eligible.all():
        raise ValueError("fixed exclusion entered eligible population")
    indices = eligible[AXES].to_numpy(dtype=np.int64)
    if (
        not np.array_equal(indices, eligible[AXES].to_numpy())
        or np.any(indices < 0)
        or np.any(indices >= np.asarray(shape))
    ):
        raise ValueError("invalid grid indices")
    flat = np.ravel_multi_index(indices.T, shape)
    if not np.array_equal(flat, eligible.grid_cell):
        raise ValueError("axis flattening mismatch")
    if not np.array_equal(
        eligible.source_cell * (shape[2] * shape[3]) + eligible.receiver_cell, flat
    ):
        raise ValueError("source/receiver cell mismatch")
    actual_counts = np.bincount(flat, minlength=size)
    if not np.array_equal(
        cells.set_index("grid_cell").loc[np.arange(size), "retained_count"], actual_counts
    ):
        raise ValueError("saved cell counts mismatch")
    for role, role_shape in [("source", shape[:2]), ("receiver", shape[2:])]:
        table = stations[role].set_index("station_id", verify_integrity=True)
        saved = table.loc[eligible[f"{role}_id"]]
        role_indices = eligible[[f"{role}_grid_line", f"{role}_grid_point"]].to_numpy()
        if not np.array_equal(role_indices, saved[["grid_line", "grid_point"]].to_numpy()):
            raise ValueError("station-to-trace grid mismatch")
        if not np.array_equal(
            eligible[f"{role}_cell"], np.ravel_multi_index(role_indices.T, role_shape)
        ):
            raise ValueError("station cell mismatch")
        xy = eligible[[f"{role}_x_m", f"{role}_y_m"]].to_numpy()
        projected = grid_coordinates(role_indices, grids[role])
        if not np.isfinite(xy).all() or not np.array_equal(xy, saved[["x_m", "y_m"]].to_numpy()):
            raise ValueError("actual station coordinates mismatch")
        if not np.allclose(
            projected, eligible[[f"{role}_grid_x_m", f"{role}_grid_y_m"]], atol=1e-8, rtol=0
        ):
            raise ValueError("projected station coordinates mismatch")
        distance = np.linalg.norm(xy - projected, axis=1)
        normalized = np.linalg.norm(continuous_indices(xy, grids[role]) - role_indices, axis=1)
        if not np.allclose(
            distance, eligible[f"{role}_distance_m"], atol=1e-8, rtol=0
        ) or not np.allclose(
            normalized, eligible[f"{role}_normalized_distance"], atol=1e-9, rtol=0
        ):
            raise ValueError("projection distance mismatch")
    eligible = eligible.sort_values("grid_cell").reset_index(drop=True)
    eligible["trace_id"] = eligible.source_file + "#" + eligible.trace_index.astype(str)
    eligible["cell_id"] = eligible.grid_cell
    eligible["candidate_id"] = contract["candidate_id"]
    eligible["clean_target"] = ~eligible[QC_FLAGS].any(axis=1)
    for role in ("source", "receiver"):
        eligible[f"{role}_projection_distance_m"] = eligible[f"{role}_distance_m"]
        eligible[f"{role}_projection_distance_normalized"] = eligible[f"{role}_normalized_distance"]
    return eligible


def make_outer_mask(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    """SHA256 of a canonical JSON list, ordered by digest then integer cell ID."""
    if not manifest.eligible_after_fixed_qc.all() or manifest.cell_id.duplicated().any():
        raise ValueError("outer mask requires unique eligible cells")
    count = config["test_trace_count"]
    if len(manifest) != config["eligible_trace_count"] or not 0 < count < len(manifest):
        raise ValueError("outer mask population/count mismatch")
    columns = [
        "candidate_id",
        "trace_id",
        "cell_id",
        "source_cell",
        "receiver_cell",
        "eligible_after_fixed_qc",
        *QC_FLAGS,
        "clean_target",
    ]
    columns += [
        f"{r}_projection_distance{s}" for r in ("source", "receiver") for s in ("_m", "_normalized")
    ]
    mask = manifest[columns].sort_values("cell_id").reset_index(drop=True).copy()
    prefix = [config["study_id"], config["candidate_id"], config["mask_id"], config["mask_seed"]]
    if not mask.candidate_id.eq(config["candidate_id"]).all():
        raise ValueError("mask candidate mismatch")
    mask["stable_hash"] = [
        hashlib.sha256(
            json.dumps([*prefix, int(cell)], ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        for cell in mask.cell_id
    ]
    order = mask.sort_values(["stable_hash", "cell_id"]).index
    mask["hash_rank"] = 0
    mask.loc[order, "hash_rank"] = np.arange(1, len(mask) + 1)
    mask["split"] = np.where(mask.hash_rank <= count, "test", "observed")
    mask["mask_id"], mask["mask_seed"] = config["mask_id"], config["mask_seed"]
    if int(mask.split.eq("observed").sum()) != config["observed_trace_count"]:
        raise ValueError("outer observed count mismatch")
    return mask


def make_geometry(manifest: pd.DataFrame, grids: dict, mode: str) -> pd.DataFrame:
    """Return a whitelist of model geometry; grid mode never reads actual UTM."""
    if mode not in ("real", "grid"):
        raise ValueError("geometry mode must be real or grid")
    result = manifest[["cell_id", *AXES]].copy()
    xy = {}
    for role in ("source", "receiver"):
        indices = result[[f"{role}_grid_line", f"{role}_grid_point"]].to_numpy()
        xy[role] = (
            manifest[[f"{role}_x_m", f"{role}_y_m"]].to_numpy().copy()
            if mode == "real"
            else grid_coordinates(indices, grids[role])
        )
        local = (
            continuous_indices(xy[role], grids[role])
            if mode == "real"
            else indices.astype(np.float64)
        )
        result[[f"{role}_x_m", f"{role}_y_m"]] = xy[role]
        result[[f"{role}_line_continuous", f"{role}_point_continuous"]] = local
    delta = xy["receiver"] - xy["source"]
    result[["offset_x_m", "offset_y_m"]] = delta
    result["offset_m"] = np.linalg.norm(delta, axis=1)
    # Clockwise from north, matching compute_trace_graph_geometry.
    result["azimuth_rad"] = np.arctan2(delta[:, 0], delta[:, 1])
    result[["midpoint_x_m", "midpoint_y_m"]] = (xy["source"] + xy["receiver"]) / 2
    return result


def validate_regsi_pair(configs: dict) -> None:
    """Reject any ablation change besides method identity and geometry."""
    left, right = (
        {k: v for k, v in configs[name].items() if k not in ("method_id", "geometry_mode")}
        for name in ("regsi_real", "regsi_grid")
    )
    if left != right:
        raise ValueError("ReGSI pair differs beyond geometry")
