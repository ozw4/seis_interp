"""Whole-shot acquisition masks on original FORGE source grid indices."""

import numpy as np
import pandas as pd

from seis_interp.processing.forge_mvp_contract import stable_cell_hashes


def make_periodic_shot_mask(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign all eligible receivers of a source cell together, without renumbering."""
    specification = config["interpolation_mask"]
    if (
        specification["period"] != 2
        or type(specification["observed_phase"]) is not int
        or specification["observed_phase"] not in (0, 1)
    ):
        raise ValueError("periodic shot50 requires period=2 and observed_phase=0 or 1")
    shape = config["spatial_shape"]
    if shape[1] % 2:
        raise ValueError("shot50 requires an even original source point count")
    if (
        len(manifest) != config["eligible_trace_count"]
        or not manifest.eligible_after_fixed_qc.all()
        or manifest.cell_id.duplicated().any()
        or not manifest.candidate_id.eq(config["candidate_id"]).all()
    ):
        raise ValueError("periodic mask requires unique eligible candidate cells")
    mask = validated_shot_geometry(manifest, shape)
    mask["split"] = np.where(
        mask.source_grid_point % 2 == specification["observed_phase"], "observed", "test"
    )
    mask["role"] = mask.split.map({"observed": "train", "test": "test"})
    mask["mask_id"] = specification["id"]
    mask["period"] = 2
    mask["observed_phase"] = specification["observed_phase"]
    return mask


def validated_shot_geometry(manifest, shape):
    """Check the complete original source domain and retain original trace cell IDs."""
    mask = manifest.sort_values("cell_id").reset_index(drop=True).copy()
    source = mask.source_grid_line * shape[1] + mask.source_grid_point
    indices = mask[["source_grid_line", "source_grid_point"]].to_numpy()
    if (
        not np.array_equal(indices, indices.astype(np.int64))
        or np.any(indices < 0)
        or np.any(indices >= np.asarray(shape[:2]))
    ):
        raise ValueError("invalid original source grid indices")
    cell = source * (shape[2] * shape[3]) + mask.receiver_cell
    if not np.array_equal(source, mask.source_cell) or not np.array_equal(cell, mask.cell_id):
        raise ValueError("original source/cell identity mismatch")
    shots = mask[["source_cell", "source_grid_line", "source_grid_point"]].drop_duplicates()
    if not np.array_equal(np.sort(shots.source_cell), np.arange(shape[0] * shape[1])):
        raise ValueError("original source grid coverage mismatch")
    return mask


def make_random_shot_mask(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rank original source cells independently within each line by a seeded SHA256."""
    specification, shape = config["interpolation_mask"], config["spatial_shape"]
    if specification["algorithm"] != "sha256_rank_per_source_line":
        raise ValueError("unknown random shot ranking algorithm")
    seed, count = specification["mask_seed"], specification["observed_shots_per_line"]
    if type(seed) is not int or seed < 0:
        raise ValueError("random shot mask_seed must be a nonnegative integer")
    if type(count) is not int or shape[1] % 2 or count != shape[1] // 2:
        raise ValueError("random shot50 must observe exactly half the original points per line")
    if (
        len(manifest) != config["eligible_trace_count"]
        or not manifest.eligible_after_fixed_qc.all()
        or manifest.cell_id.duplicated().any()
        or not manifest.candidate_id.eq(config["candidate_id"]).all()
    ):
        raise ValueError("random shot mask requires unique eligible candidate cells")
    mask = validated_shot_geometry(manifest, shape)
    shots = mask[["source_cell", "source_grid_line"]].drop_duplicates().copy()
    prefix = [config["study_id"], config["candidate_id"], specification["id"], seed]
    shots["shot_stable_hash"] = [
        stable_cell_hashes([*prefix, int(line)], [cell])[0]
        for cell, line in shots.itertuples(index=False, name=None)
    ]
    shots = shots.sort_values(["source_grid_line", "shot_stable_hash", "source_cell"])
    shots["shot_hash_rank"] = shots.groupby("source_grid_line").cumcount() + 1
    lookup = shots.set_index("source_cell")
    for column in ("shot_stable_hash", "shot_hash_rank"):
        mask[column] = mask.source_cell.map(lookup[column])
    mask["split"] = np.where(mask.shot_hash_rank <= count, "observed", "test")
    mask["role"] = mask.split.map({"observed": "train", "test": "test"})
    mask["mask_id"], mask["mask_seed"] = specification["id"], seed
    mask["mask_family"] = "random_shot"
    return mask


def shot_mask_counts(mask: pd.DataFrame, spatial_shape) -> dict:
    """Record shot fractions separately from eligible trace fractions and fixed QC."""
    lines = []
    for line, group in mask.groupby("source_grid_line", sort=True):
        counts = {"source_grid_line": int(line)}
        for role in ("observed", "test"):
            rows = group.loc[group.split.eq(role)]
            counts[f"{role}_shot_count"] = int(rows.source_cell.nunique())
            counts[f"{role}_trace_count"] = len(rows)
        if (
            counts["observed_shot_count"] != spatial_shape[1] // 2
            or counts["test_shot_count"] != spatial_shape[1] // 2
        ):
            raise ValueError("each source line must retain exactly half its shots")
        lines.append(counts)
    identity = (
        {
            "mask_family": "random_shot",
            "mask_seed": int(mask.mask_seed.iloc[0]),
            "algorithm": "sha256_rank_per_source_line",
        }
        if "mask_family" in mask and mask.mask_family.eq("random_shot").all()
        else {"period": 2, "observed_phase": int(mask.observed_phase.iloc[0])}
    )
    return {
        "mask_id": str(mask.mask_id.iloc[0]),
        **identity,
        "by_line": lines,
        **{key: sum(row[key] for row in lines) for key in lines[0] if key != "source_grid_line"},
        "eligible_trace_missing_fraction": float(mask.split.eq("test").mean()),
        "natural_missing_cell_count": int(np.prod(spatial_shape)) - len(mask),
    }


def selected_shot_gathers(partition, spatial_shape):
    """Choose lower central line, nearest central missing point and its missing edge."""
    if "mask_family" in partition and partition.mask_family.eq("random_shot").all():
        shots = random_shot_support(partition, spatial_shape)
        shots["line_distance"] = abs(shots.source_grid_line - (spatial_shape[0] - 1) / 2)
        shots["point_distance"] = abs(shots.source_grid_point - (spatial_shape[1] - 1) / 2)
        return {
            support: int(
                group.sort_values(
                    ["line_distance", "source_grid_line", "point_distance", "source_cell"]
                ).source_cell.iloc[0]
            )
            for support, group in shots.groupby("shot_support", sort=True)
        }
    shots = partition.loc[
        partition.role.eq("test"), ["source_cell", "source_grid_line", "source_grid_point"]
    ].drop_duplicates()
    line = (spatial_shape[0] - 1) / 2
    shots = shots.assign(line_distance=abs(shots.source_grid_line - line))
    shots = shots.loc[shots.line_distance.eq(shots.line_distance.min())]
    shots = shots.loc[shots.source_grid_line.eq(shots.source_grid_line.min())]
    edge = shots.source_grid_point.isin([0, spatial_shape[1] - 1])
    internal = shots.loc[~edge].copy()
    internal["point_distance"] = abs(internal.source_grid_point - (spatial_shape[1] - 1) / 2)
    if internal.empty or not edge.any():
        raise ValueError("gather selection requires internal and edge missing shots")
    return {
        "internal": int(
            internal.sort_values(["point_distance", "source_cell"]).source_cell.iloc[0]
        ),
        "edge": int(shots.loc[edge].sort_values("source_cell").source_cell.iloc[0]),
    }


def random_shot_support(partition, spatial_shape):
    """Describe source-axis support and contiguous missing gaps using geometry alone."""
    rows = []
    for line, group in partition.groupby("source_grid_line", sort=True):
        observed = np.sort(group.loc[group.role.eq("train"), "source_grid_point"].unique())
        test = group.loc[
            group.role.eq("test"), ["source_cell", "source_grid_point"]
        ].drop_duplicates()
        for source, point in test.itertuples(index=False, name=None):
            left, right = observed[observed < point], observed[observed > point]
            lower = int(left[-1]) if len(left) else -1
            upper = int(right[0]) if len(right) else spatial_shape[1]
            rows.append(
                {
                    "source_cell": int(source),
                    "source_grid_line": int(line),
                    "source_grid_point": int(point),
                    "shot_support": "internal" if len(left) and len(right) else "edge",
                    "endpoint_shot": point in (0, spatial_shape[1] - 1),
                    "observed_left_point": lower if len(left) else None,
                    "observed_right_point": upper if len(right) else None,
                    "missing_gap_shot_count": upper - lower - 1,
                    "nearest_observed_point_distance": int(np.min(abs(observed - point))),
                }
            )
    return pd.DataFrame(rows).sort_values("source_cell").reset_index(drop=True)
