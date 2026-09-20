"""Run geometry EDA on hash-bound FORGE header and waveform audit artifacts."""

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.forge_geometry_qc import (
    STATUS_LABELS,
    availability_matrix,
    fit_station_lattice,
    numeric_quantiles,
    prepare_geometry,
    reference_pairs,
    sparse_grid_occupancy,
    spatial_coordinates,
    station_id_occupancy,
    station_spacing,
)
from seis_interp.visualization.forge_geometry_qc import (
    plot_availability,
    plot_grid_comparison,
    plot_station_id_support,
    plot_station_layout,
    plot_station_spacing,
    plot_trace_geometry,
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_forge_geometry_audit(repo: Path, study: Path) -> Path:
    started = datetime.now(timezone.utc)
    config = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    wave_run = repo / inputs["waveform_run"]
    wave_meta = json.loads((wave_run / "metadata.json").read_text())
    wave_path = repo / wave_meta["interim_directory"] / "waveform_qc.parquet"
    if sha256_file(wave_path) != wave_meta["waveform_table_sha256"]:
        raise ValueError("waveform artifact hash mismatch")
    wave_inputs = yaml.safe_load((wave_run / "inputs.resolved.yaml").read_text())
    header_run = repo / wave_inputs["header_run"]
    header_meta = json.loads((header_run / "metadata.json").read_text())
    header_dir = repo / header_meta["interim_directory"]
    header_path = header_dir / "trace_headers.parquet"
    nav_path = header_dir / "source_navigation.csv"
    for path in [header_path, nav_path]:
        if sha256_file(path) != header_meta["artifacts"][path.relative_to(repo).as_posix()]:
            raise ValueError("header artifact hash mismatch")
    wave_lock = json.loads((wave_run / "input.lock.json").read_text())
    if sha256_file(header_path) != wave_lock["header_table_sha256"]:
        raise ValueError("waveform/header lineage mismatch")
    table = pd.read_parquet(wave_path)
    domain, sources, receivers, observed = prepare_geometry(table)
    time = observed[["sample_count", "sample_interval_us"]].drop_duplicates()
    if len(time) != 1:
        raise ValueError("inconsistent time axes")
    n_samples, dt_us = map(int, time.iloc[0])
    matrix = availability_matrix(domain, len(sources), len(receivers))
    reference = reference_pairs(sources, receivers)
    fits, residuals, spacing = {}, {}, {}
    for role, stations in [("source", sources), ("receiver", receivers)]:
        fits[role], residuals[role] = fit_station_lattice(stations, role)
        spacing[role] = station_spacing(stations, role)
    east = fits["receiver"]["coefficients_xy_m"][1]
    east = (np.asarray(east) / np.linalg.norm(east)).tolist()
    frames = {"utm": [1.0, 0.0], "receiver_aligned": east}
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + commit[:12]
    run = repo / "runs" / study.name / run_id
    interim = repo / "data/interim/forge_2017" / run_id
    (run / "figures").mkdir(parents=True, exist_ok=False)
    (interim / "grid_cells").mkdir(parents=True, exist_ok=False)
    for name in ["config.yaml", "inputs.yaml"]:
        shutil.copyfile(study / name, run / name.replace(".yaml", ".resolved.yaml"))
    source_repo = Path(__file__).resolve().parents[3]
    for relative in [
        "src/seis_interp/processing/forge_geometry_qc.py",
        "src/seis_interp/processing/geometry.py",
        "src/seis_interp/visualization/forge_geometry_qc.py",
        "src/seis_interp/pipelines/audit_forge_geometry.py",
        "src/seis_interp/data/forge_headers.py",
        "scripts/audit_forge_geometry.py",
    ]:
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_repo / relative, destination)
    for role, stations in [("source", sources), ("receiver", receivers)]:
        axis = 1 if role == "source" else 0
        stations["numerically_usable_traces"] = (matrix >= 3).sum(axis=axis)
        stations["review_traces"] = (matrix == 3).sum(axis=axis)
        stations.to_csv(run / f"{role}_stations.csv", index=False)
        residuals[role].to_csv(run / f"{role}_lattice_residuals.csv", index=False)
        for label, data in zip(["along_line", "across_line"], spacing[role], strict=True):
            data.to_csv(run / f"{role}_{label}_spacing.csv", index=False)
    np.savez_compressed(run / "station_pair_availability.npz", status=matrix)
    geometry_cols = [
        "source_file",
        "trace_index",
        "ffid",
        "source_id",
        "receiver_id",
        "source_line",
        "source_point",
        "receiver_line",
        "receiver_point",
        "source_x_m",
        "source_y_m",
        "receiver_x_m",
        "receiver_y_m",
        "midpoint_x_m",
        "midpoint_y_m",
        "offset_x_m",
        "offset_y_m",
        "offset_m",
        "azimuth_deg",
        "axial_azimuth_deg",
        "waveform_status",
    ]
    observed[geometry_cols].to_parquet(interim / "trace_geometry.parquet", index=False)
    domain.loc[
        ~domain.geometry_usable,
        [
            "source_file",
            "trace_index",
            "source_id",
            "receiver_id",
            "trace_identification_code",
            "waveform_status",
        ],
    ].to_csv(run / "excluded_recorded_pairs.csv", index=False)
    line_coverage = domain.groupby(["source_line", "receiver_line"]).agg(
        recorded_pairs=("ffid", "size"), usable_pairs=("geometry_usable", "sum")
    )
    line_coverage["usable_fraction"] = line_coverage.usable_pairs / line_coverage.recorded_pairs
    line_coverage.to_csv(run / "line_pair_coverage.csv")
    grid_results = []
    for frame in config["frames"]:
        for grid in config["grids"]:
            arguments = (grid["representation"], config["origin_xy_m"], frames[frame])
            ref_coords = spatial_coordinates(reference, *arguments)
            obs_coords = spatial_coordinates(observed, *arguments)
            for phase in config["bin_phases"]:
                metrics, cells = sparse_grid_occupancy(
                    ref_coords, obs_coords, grid["widths_m"], phase
                )
                metrics.update(
                    frame=frame,
                    grid_id=grid["id"],
                    representation=grid["representation"],
                    dense_float32_bytes=metrics["bbox_cells"] * n_samples * 4,
                )
                grid_results.append(metrics)
                cells.to_parquet(
                    interim / "grid_cells" / f"{frame}_{grid['id']}_phase{phase:g}.parquet",
                    index=False,
                )
        print(f"Geometry occupancy completed: {frame}", flush=True)
    grid_table = pd.DataFrame(grid_results)
    grid_table.to_csv(run / "grid_occupancy.csv", index=False)
    _json(run / "grid_occupancy.json", grid_results)
    navigation = pd.read_csv(nav_path)
    source_nav_coverage = navigation.merge(
        sources, left_on=["line", "point"], right_on=["source_line", "source_point"], how="left"
    )
    source_nav_coverage["has_local_record"] = source_nav_coverage.source_id.notna()
    source_nav_coverage.groupby("line").agg(
        navigation_stations=("point", "size"), local_stations=("has_local_record", "sum")
    ).to_csv(run / "source_navigation_coverage.csv")
    summary = {
        "complete": True,
        "trace_count": len(table),
        "recorded_seismic_pairs": len(domain),
        "source_count": len(sources),
        "receiver_count": len(receivers),
        "source_line_count": int(sources.source_line.nunique()),
        "receiver_line_count": int(receivers.receiver_line.nunique()),
        "numeric_usable_pairs": len(observed),
        "recorded_station_product": int(matrix.size),
        "station_product_fill_fraction": len(observed) / matrix.size,
        "availability_codes": STATUS_LABELS,
        "availability_counts": {
            STATUS_LABELS[int(k)]: int(v)
            for k, v in zip(*np.unique(matrix, return_counts=True), strict=True)
        },
        "known_coordinate_reference_pairs": len(reference),
        "unknown_coordinate_receivers": receivers.loc[
            receivers.receiver_x_m.isna(), ["receiver_line", "receiver_point"]
        ].to_dict("records"),
        "fully_usable_receiver_count": int((matrix >= 3).all(axis=0).sum()),
        "known_reference_missing_pairs": len(reference) - len(observed),
        "station_id_rectangle": station_id_occupancy(
            sources, receivers, observed, config["station_id_steps"]
        ),
        "source_navigation_station_count": len(navigation),
        "local_source_fraction_of_navigation": len(sources) / len(navigation),
        "time_axis": {
            "samples": n_samples,
            "dt_s": dt_us / 1e6,
            "duration_s": (n_samples - 1) * dt_us / 1e6,
        },
        "offset_m": numeric_quantiles(observed.offset_m),
        "azimuth_10deg_counts": np.histogram(
            observed.azimuth_deg.dropna(), bins=np.arange(0, 361, 10)
        )[0].tolist(),
        "coordinate_ranges_m": {
            c: [float(observed[c].min()), float(observed[c].max())]
            for c in geometry_cols
            if c.endswith("_m")
        },
        "lattice_fits": fits,
        "frame_east_axes": frames,
        "spacing": {
            role: {
                "adjacent_point_m": numeric_quantiles(
                    along.loc[along.point_id_step.eq(1), "distance_m"]
                ),
                "point_id_gap_counts": along.loc[along.point_id_step.gt(1), "point_id_step"]
                .value_counts()
                .to_dict(),
                "matched_point_line_m": numeric_quantiles(across.distance_m),
            }
            for role, (along, across) in spacing.items()
        },
    }
    _json(run / "summary.json", summary)
    plot_station_layout(sources, receivers, navigation, matrix, run / "figures/station_layout.png")
    plot_station_spacing(spacing, run / "figures/station_spacing.png")
    plot_availability(matrix, sources, receivers, run / "figures/station_availability.png")
    plot_station_id_support(
        sources, receivers, config["station_id_steps"], run / "figures/station_id_support.png"
    )
    plot_trace_geometry(observed, config["fold_bin_width_m"], run / "figures/offset_midpoint.png")
    plot_grid_comparison(grid_table, run / "figures/grid_occupancy.png")
    bound_paths = [
        wave_path,
        header_path,
        nav_path,
        wave_run / "metadata.json",
        wave_run / "input.lock.json",
        wave_run / "inputs.resolved.yaml",
        header_run / "metadata.json",
    ]
    _json(
        run / "input.lock.json",
        {p.relative_to(repo).as_posix(): sha256_file(p) for p in bound_paths},
    )
    _json(
        run / "metadata.json",
        {
            "git_commit": commit,
            "seed": None,
            "deterministic": True,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": {
                p: version(p) for p in ["numpy", "pandas", "matplotlib", "pyarrow", "PyYAML"]
            },
            "interim_directory": interim.relative_to(repo).as_posix(),
            "artifacts": {
                p.relative_to(repo).as_posix(): sha256_file(p)
                for p in sorted(interim.rglob("*.parquet"))
            },
        },
    )
    print(f"Geometry audit saved: {run.relative_to(repo)}", flush=True)
    return run
