"""The SEG C3 NA crop on the FORGE fair-grid contract: geometry, trace manifest, outer mask.

The dense crop of the C3 volume index maps onto the FORGE contract axes as
``source_grid_line`` <- source line, ``source_grid_point`` <- shot in line,
``receiver_grid_line`` <- relative receiver x and ``receiver_grid_point`` <-
relative receiver y. A cell ID is the C-order ravel over these four axes, which
is the row order of a validated volume index. Receivers are relative to their
source, so a receiver cell is an offset bin rather than a fixed station. C3 has
no trace QC flags: every cell is eligible and every target is clean.
"""

import numpy as np
import pandas as pd

from seis_interp.processing.forge_mvp_contract import AXES, stable_cell_hashes

C3_DATASET_ID = "seg_c3_na"
# Volume-index columns behind the four FORGE contract axes, in ``AXES`` order.
C3_INDEX_AXES = (
    "source_line_index",
    "shot_in_line_index",
    "relative_receiver_x_index",
    "relative_receiver_y_index",
)
MASK_COLUMNS = ["candidate_id", "trace_id", "cell_id", "source_cell", "receiver_cell"]


def c3_cell_ids(volume_index: pd.DataFrame, spatial_shape) -> np.ndarray:
    """Cell ID of each volume-index row; the rows must cover the grid once, in C order."""
    shape = tuple(int(n) for n in spatial_shape)
    indices = volume_index[list(C3_INDEX_AXES)].to_numpy(dtype=np.int64)
    if (
        len(shape) != 4
        or len(indices) != int(np.prod(shape))
        or np.any(indices < 0)
        or np.any(indices >= np.asarray(shape))
    ):
        raise ValueError("volume index does not cover the spatial grid")
    cells = np.ravel_multi_index(tuple(indices.T), shape)
    if not np.array_equal(cells, np.arange(len(cells))):
        raise ValueError("volume index rows must follow C order over the spatial grid")
    return cells


def c3_geometry_table(volume_index: pd.DataFrame, spatial_shape) -> pd.DataFrame:
    """Model geometry of every cell with the columns, order and dtypes of FORGE ``make_geometry``.

    Receiver coordinates are source plus relative receiver coordinates. The
    continuous line/point coordinates are the integer grid indices, the index
    coordinates of the C3 v3 profile NeRSI; offset is receiver minus source and
    the azimuth is ``atan2(offset_x, offset_y)`` (clockwise from north), as in
    FORGE. The crop is an exact lattice and the grid methods read cell IDs only,
    so a preparation writes this one table as both its real and grid geometry.
    """
    cells = c3_cell_ids(volume_index, spatial_shape)
    table = pd.DataFrame({"cell_id": cells})
    for axis, column in zip(AXES, C3_INDEX_AXES, strict=True):
        table[axis] = volume_index[column].to_numpy(dtype=np.int64)
    source = volume_index[["source_x_m", "source_y_m"]].to_numpy(dtype=np.float64)
    relative = volume_index[["relative_receiver_x_m", "relative_receiver_y_m"]]
    xy = {"source": source, "receiver": source + relative.to_numpy(dtype=np.float64)}
    for role, (line, point) in (("source", AXES[:2]), ("receiver", AXES[2:])):
        table[[f"{role}_x_m", f"{role}_y_m"]] = xy[role]
        table[[f"{role}_line_continuous", f"{role}_point_continuous"]] = table[
            [line, point]
        ].to_numpy(dtype=np.float64)
    delta = xy["receiver"] - xy["source"]
    table[["offset_x_m", "offset_y_m"]] = delta
    table["offset_m"] = np.linalg.norm(delta, axis=1)
    table["azimuth_rad"] = np.arctan2(delta[:, 0], delta[:, 1])
    table[["midpoint_x_m", "midpoint_y_m"]] = (xy["source"] + xy["receiver"]) / 2
    return table


def c3_trace_manifest(volume_index: pd.DataFrame, spatial_shape, candidate_id: str) -> pd.DataFrame:
    """Cell -> interim ``array_row`` mapping; the row never enters model geometry."""
    cells = c3_cell_ids(volume_index, spatial_shape)
    rows = volume_index.array_row.to_numpy(dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("duplicate array_row in the volume index")
    receivers = int(spatial_shape[2]) * int(spatial_shape[3])
    return pd.DataFrame(
        {
            "cell_id": cells,
            "array_row": rows,
            "ffid": volume_index.ffid.to_numpy(dtype=np.int64),
            "trace_id": [f"c3na:{row}" for row in rows],
            "source_cell": cells // receivers,
            "receiver_cell": cells % receivers,
            "candidate_id": candidate_id,
        }
    )


def make_c3_outer_mask(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The ``make_outer_mask`` rule on a dense C3 crop: exact-count SHA256 rank.

    Cells are ranked by the SHA256 of the canonical JSON list ``[mask_study_id
    (default study_id), candidate_id, mask_id, mask_seed, cell_id]``, digest then
    cell ID ascending; the first ``test_trace_count`` become ``test``.
    """
    if manifest.cell_id.duplicated().any():
        raise ValueError("outer mask requires unique cells")
    count = config["test_trace_count"]
    if len(manifest) != config["eligible_trace_count"] or not 0 < count < len(manifest):
        raise ValueError("outer mask population/count mismatch")
    if not manifest.candidate_id.eq(config["candidate_id"]).all():
        raise ValueError("mask candidate mismatch")
    mask = manifest[MASK_COLUMNS].sort_values("cell_id").reset_index(drop=True).copy()
    mask["clean_target"] = True
    prefix = [
        config.get("mask_study_id", config["study_id"]),
        config["candidate_id"],
        config["mask_id"],
        config["mask_seed"],
    ]
    mask["stable_hash"] = stable_cell_hashes(prefix, mask.cell_id)
    order = mask.sort_values(["stable_hash", "cell_id"]).index
    rank = np.zeros(len(mask), dtype=np.int64)
    rank[order] = np.arange(1, len(mask) + 1)
    mask["hash_rank"] = rank
    mask["split"] = np.where(rank <= count, "test", "observed")
    mask["mask_id"], mask["mask_seed"] = config["mask_id"], config["mask_seed"]
    if int(mask.split.eq("observed").sum()) != config["observed_trace_count"]:
        raise ValueError("outer observed count mismatch")
    return mask
