"""Export an actual-coordinate map for one source of a sealed FORGE mask."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.visualization.forge_geometry_map import (
    create_source_mask_map,
    prepare_source_mask_map,
)
from seis_interp.visualization.forge_survey_map import create_survey_map, prepare_survey_map


def visualize_forge_geometry(repo, preparation, output, *, source_cell=112, dpi=600):
    """Read only sealed coordinates/mask metadata; write PNG, vector PDF and SVG."""
    start = utc_now()
    seal_path = preparation / "preparation_manifest.json"
    seal = json.loads(seal_path.read_text())
    if seal["status"] != "prepared":
        raise ValueError("preparation is not complete")
    names = [
        "input/input_contract.json",
        "input/m1_trace_manifest.parquet",
        "masks/m1_random80_mvp_v1.parquet",
    ]
    hashes = {name: seal["artifacts"][name] for name in names}
    verify_hashes(preparation, hashes)
    contract = json.loads((preparation / names[0]).read_text())
    manifest = pd.read_parquet(preparation / names[1])
    mask = pd.read_parquet(preparation / names[2])
    sources, receivers = prepare_source_mask_map(manifest, mask, source_cell=source_cell)
    figure = create_source_mask_map(
        sources,
        receivers,
        source_cell=source_cell,
        missing_fraction=contract["missing_fraction"],
        mask_seed=seal["mask_seed"],
    )
    output.mkdir(parents=True, exist_ok=False)
    sources.to_csv(output / "source_stations.csv", index=False)
    receivers.to_csv(output / "receiver_states.csv", index=False)
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("png", "pdf", "svg"):
            figure.savefig(
                output / f"geometry_mask_source{source_cell}.{extension}",
                dpi=dpi,
                facecolor="white",
            )
    plt.close(figure)
    for relative in (
        "scripts/forge_geometry_map.py",
        "src/seis_interp/visualization/forge_geometry_map.py",
        "src/seis_interp/pipelines/visualize_forge_geometry.py",
    ):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / relative).read_bytes())
    write_json(
        output / "manifest.json",
        {
            **execution_identity(repo),
            "status": "completed",
            "start_time": start,
            "end_time": utc_now(),
            "preparation": str(preparation.relative_to(repo)),
            "preparation_manifest_sha256": sha256_file(seal_path),
            "input_hashes": hashes,
            "mask_id": seal["mask_id"],
            "mask_seed": seal["mask_seed"],
            "mask_hash": seal["mask_hash"],
            "global_missing_fraction": contract["missing_fraction"],
            "source_cell": source_cell,
            "source_count": len(sources),
            "receiver_count": len(receivers),
            "receiver_state_counts": {
                split: int(count) for split, count in receivers.split.value_counts().items()
            },
            "coordinates": "actual UTM easting/northing, meters, equal axis scale",
            "mask_display": "receiver states for one fixed source; other sources are context",
            "width_mm": 178,
            "height_mm": 178,
            "dpi": dpi,
            "font_family": "DejaVu Sans",
            "waveforms_read": 0,
            "new_inference": False,
            "new_test_scoring": False,
            "artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return output


def visualize_forge_survey(repo, preparation, geometry_run, output, *, dpi=600):
    """Locate the sealed analysis stations in hash-verified full-survey geometry."""
    start = utc_now()
    seal_path = preparation / "preparation_manifest.json"
    seal = json.loads(seal_path.read_text())
    if seal["status"] != "prepared":
        raise ValueError("preparation is not complete")
    analysis_name = "input/m1_trace_manifest.parquet"
    verify_hashes(preparation, {analysis_name: seal["artifacts"][analysis_name]})
    metadata_path = geometry_run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    geometry_name = (Path(metadata["interim_directory"]) / "trace_geometry.parquet").as_posix()
    lock_path = geometry_run / "input.lock.json"
    lock = json.loads(lock_path.read_text())
    navigation_names = [name for name in lock if Path(name).name == "source_navigation.csv"]
    if len(navigation_names) != 1:
        raise ValueError("full geometry requires one source navigation table")
    navigation_name = navigation_names[0]
    hashes = {
        geometry_name: metadata["artifacts"][geometry_name],
        navigation_name: lock[navigation_name],
    }
    verify_hashes(repo, hashes)
    columns = [
        f"{role}_{field}"
        for role in ("source", "receiver")
        for field in ("line", "point", "x_m", "y_m")
    ]
    sources, receivers = prepare_survey_map(
        pd.read_parquet(repo / geometry_name, columns=columns),
        pd.read_csv(repo / navigation_name),
        pd.read_parquet(preparation / analysis_name, columns=columns),
    )
    summary_path = geometry_run / "summary.json"
    summary = json.loads(summary_path.read_text())
    unknown_receivers = len(summary["unknown_coordinate_receivers"])
    if (
        len(sources) != summary["source_navigation_station_count"]
        or int(sources.recorded.sum()) != summary["source_count"]
        or len(receivers) != summary["receiver_count"] - unknown_receivers
    ):
        raise ValueError("full-survey station counts differ from audit summary")
    figure = create_survey_map(sources, receivers, unknown_receiver_count=unknown_receivers)
    output.mkdir(parents=True, exist_ok=False)
    sources.to_csv(output / "survey_sources.csv", index=False)
    receivers.to_csv(output / "survey_receivers.csv", index=False)
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("png", "pdf", "svg"):
            figure.savefig(
                output / f"survey_analysis_extent.{extension}", dpi=dpi, facecolor="white"
            )
    plt.close(figure)
    for relative in (
        "scripts/forge_survey_map.py",
        "src/seis_interp/visualization/forge_survey_map.py",
        "src/seis_interp/pipelines/visualize_forge_geometry.py",
    ):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / relative).read_bytes())
    write_json(
        output / "manifest.json",
        {
            **execution_identity(repo),
            "status": "completed",
            "start_time": start,
            "end_time": utc_now(),
            "preparation": str(preparation.relative_to(repo)),
            "preparation_manifest_sha256": sha256_file(seal_path),
            "analysis_geometry_sha256": seal["artifacts"][analysis_name],
            "geometry_run": str(geometry_run.relative_to(repo)),
            "geometry_metadata_sha256": sha256_file(metadata_path),
            "geometry_input_lock_sha256": sha256_file(lock_path),
            "geometry_summary_sha256": sha256_file(summary_path),
            "input_hashes": hashes,
            "candidate_id": seal["candidate_id"],
            "survey_source_count": len(sources),
            "survey_recorded_source_count": int(sources.recorded.sum()),
            "survey_known_receiver_count": len(receivers),
            "unknown_receiver_count": unknown_receivers,
            "analysis_source_count": int(sources.in_analysis.sum()),
            "analysis_receiver_count": int(receivers.in_analysis.sum()),
            "coordinates": "actual UTM 12N easting/northing, meters, equal axis scale",
            "extent_box": "axis-aligned combined XY extent of selected source/receiver stations",
            "width_mm": 178,
            "height_mm": 160,
            "dpi": dpi,
            "font_family": "DejaVu Sans",
            "waveforms_read": 0,
            "new_inference": False,
            "new_test_scoring": False,
            "artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return output
