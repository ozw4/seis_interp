"""Compare each method's training visibility with its test visibility on one prepared run.

Geometry and masks only: no waveform is read, no model is built and nothing is scored.
A whole-shot mask hides every trace of a shot, so a test query has no observed trace in
its own source gather. A method whose training view hides individual traces still gives
its training queries that in-gather context, and then it is fitted on a task its test
never presents. This script measures that gap in each method's own neighbourhood terms.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--method", action="append", choices=("regsi_real", "ccnet5d_grid"), required=True
    )
    parser.add_argument("--queries", type=int, default=1500, help="sampled ReGSI destinations")
    parser.add_argument("--patches", type=int, default=400, help="sampled CCNet5D patches")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name, value in (("--queries", args.queries), ("--patches", args.patches)):
        if value < 1:
            parser.error(f"{name} must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"

    import pandas as pd
    import yaml

    contract = yaml.safe_load((args.experiment / "study/config.yaml").read_text())
    if contract.get("mask_family") not in ("periodic_shot", "random_shot"):
        parser.error("this audit applies to a whole-shot mask family only")
    geometry = pd.read_parquet(args.experiment / "input/real_geometry.parquet")
    partition = pd.read_parquet(args.experiment / "input/partition.parquet")
    role = partition.set_index("cell_id").role.reindex(geometry.cell_id.to_numpy()).to_numpy()
    tables = []
    for method in dict.fromkeys(args.method):
        config = yaml.safe_load(
            (args.experiment / "method_runs" / method / "config.resolved.yaml").read_text()
        )
        measure = regsi_context if method == "regsi_real" else ccnet_context
        count = args.queries if method == "regsi_real" else args.patches
        tables.append(measure(geometry, role, contract, config, count).assign(method=method))
    report = pd.concat(tables, ignore_index=True)
    print(report.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(orient="records"), indent=1) + "\n")


def regsi_context(geometry, role, contract, config, queries):
    """In-gather incoming edges per relation, under the training episode and at test."""
    import numpy as np
    import pandas as pd

    from seis_interp.processing.trace_graph_geometry import (
        RELATION_NAMES,
        compute_trace_graph_geometry,
    )
    from seis_interp.processing.trace_graph_neighbors import (
        FixedTraceGraphNeighborIndex,
        TraceGraphSpatialIndex,
    )
    from seis_interp.processing.trace_graph_settings import TraceGraphSettings

    ids = geometry.cell_id.to_numpy()
    gather = (
        geometry.source_grid_line * contract["spatial_shape"][1] + geometry.source_grid_point
    ).to_numpy()
    observed, test = role == "train", role == "test"
    full = compute_trace_graph_geometry(
        geometry[["source_x_m", "source_y_m"]].to_numpy(),
        geometry[["receiver_x_m", "receiver_y_m"]].to_numpy(),
        azimuth_min_offset_m=config["geometry_features"]["azimuth_min_offset_m"],
    )
    relations = config["graph"]["relations"]
    settings = TraceGraphSettings(
        relation_scales_m=[
            [relations[name][first], relations[name][second]]
            for name, first, second in (
                ("source", "source_scale_m", "receiver_scale_m"),
                ("receiver", "source_scale_m", "receiver_scale_m"),
                ("cmp", "midpoint_scale_m", "offset_vector_scale_m"),
                ("offset_azimuth", "midpoint_scale_m", "offset_vector_scale_m"),
            )
        ],
        neighbors_per_relation=config["graph"]["neighbors_per_relation"],
        radius=config["graph"]["max_normalized_distance"],
        candidate_chunk_size=config["graph"]["candidate_chunk_size"],
        topology=config["graph"]["topology"],
        neighbor_search=config["graph"]["neighbor_search"],
    )
    # Candidates are the training domain: the observed traces, in their own row order.
    candidates = np.flatnonzero(observed)
    index = TraceGraphSpatialIndex(
        _rows(full, candidates), ids[candidates], **settings.subgraph_kwargs()
    )

    def relation_counts(destinations, eligible, view):
        edges = FixedTraceGraphNeighborIndex.from_spatial_index(index, eligible).select(
            _rows(full, destinations), ids[destinations]
        )
        sender_row = _positions(ids[candidates])
        destination_row = _positions(ids[destinations])
        senders = np.array([sender_row[int(i)] for i in edges.sender_ids], dtype=np.int64)
        targets = np.array([destination_row[int(i)] for i in edges.destination_ids], dtype=np.int64)
        same = gather[candidates][senders] == gather[destinations][targets]
        rows = []
        for relation, name in enumerate(RELATION_NAMES):
            chosen = edges.edge_type == relation
            rows.append(_row(view, name, chosen.sum(), same[chosen], targets[chosen], queries))
        return rows + [_row(view, "all relations", len(edges.edge_type), same, targets, queries)]

    # The episode generator hides a uniform fraction of the observed traces and queries them.
    fraction = config["training"]["inner_mask_fraction"]
    episode = np.random.default_rng(config["model_seed"])
    hidden = episode.choice(ids[candidates], size=round(len(candidates) * fraction), replace=False)
    hidden_rows = np.flatnonzero(observed & np.isin(ids, hidden))
    visible = observed & ~np.isin(ids, hidden)
    sample = np.random.default_rng(991)
    return pd.DataFrame(
        relation_counts(
            _sample(sample, hidden_rows, queries), visible[candidates], "training query"
        )
        + relation_counts(
            _sample(sample, np.flatnonzero(test), queries),
            np.ones(len(candidates), dtype=bool),
            "test query",
        )
    )


def ccnet_context(geometry, role, contract, config, patches):
    """Visible in-gather cells inside a patch, for a pseudo-target and for a test cell."""
    import numpy as np
    import pandas as pd

    from seis_interp.training.ccnet5d_observed_patches import (
        _draw_pseudo_target_mask,
        _eligible_spatial_starts,
    )

    shape = tuple(contract["spatial_shape"])
    patch_shape = tuple(config["patches"]["shape"][1:])
    fraction = config["patches"]["inner_mask_fraction"]
    axes = ["source_grid_line", "source_grid_point", "receiver_grid_line", "receiver_grid_point"]
    cells = geometry[axes].to_numpy()
    observed = np.zeros(shape, dtype=bool)
    test = np.zeros(shape, dtype=bool)
    observed[tuple(cells[role == "train"].T)] = True
    test[tuple(cells[role == "test"].T)] = True
    starts = _eligible_spatial_starts(
        observed, patch_shape=patch_shape, inner_mask_fraction=fraction
    )
    placement = np.random.default_rng(config["model_seed"])
    draw = np.random.default_rng(config["model_seed"] + 1)
    counts = {"training pseudo-target": [], "test cell": []}
    for _ in range(patches):
        slices = tuple(
            slice(start, start + extent)
            for start, extent in zip(
                starts[int(placement.integers(len(starts)))], patch_shape, strict=True
            )
        )
        available = observed[slices]
        hidden = _draw_pseudo_target_mask(available, fraction=fraction, generator=draw)
        # A gather is one source station; a patch holds this many of its receiver cells.
        in_gather = (available & ~hidden).sum(axis=(2, 3))
        for view, mask in (("training pseudo-target", hidden), ("test cell", test[slices])):
            lines, points = np.nonzero(mask)[:2]
            counts[view].extend(in_gather[lines, points].tolist())
    capacity = patch_shape[2] * patch_shape[3]
    return pd.DataFrame(
        [
            {
                "view": view,
                "relation": f"in-patch gather cells (of {capacity})",
                "per_query": float(np.mean(values)),
                "queries_with_any_pct": 100.0 * float(np.mean(np.asarray(values) > 0)),
                "sampled": len(values),
            }
            for view, values in counts.items()
        ]
    )


def _row(view, relation, edges, same, targets, queries):
    import numpy as np

    return {
        "view": view,
        "relation": relation,
        "per_query": float(same.sum()) / queries,
        "queries_with_any_pct": 100.0 * len(np.unique(targets[same])) / queries,
        "sampled": int(edges),
    }


def _rows(geometry, rows):
    from dataclasses import fields

    return type(geometry)(**{f.name: getattr(geometry, f.name)[rows] for f in fields(geometry)})


def _positions(trace_ids):
    return {int(trace_id): row for row, trace_id in enumerate(trace_ids)}


def _sample(generator, rows, count):
    import numpy as np

    if count >= len(rows):
        return rows
    return np.sort(generator.choice(rows, size=count, replace=False))


if __name__ == "__main__":
    main()
