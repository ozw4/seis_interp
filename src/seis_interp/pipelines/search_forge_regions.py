"""Orchestrate hash-bound FORGE region search without resampling waveforms."""

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.forge_geometry_qc import availability_matrix, prepare_geometry
from seis_interp.processing.forge_region_search import (
    compare_region_pair,
    pair_cell_counts,
    pair_characteristics,
    rank_region_pairs,
    region_trace_mapping,
    search_station_regions,
    select_finalists,
)
from seis_interp.visualization.forge_region_search import (
    plot_region_geometry,
    plot_region_occupancy,
    plot_search_tradeoffs,
)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _load_inputs(repo, inputs):
    wave_run, qc_run, geo_run = (
        repo / inputs[k] for k in ["waveform_run", "qc_run", "geometry_run"]
    )
    wm, qm, gm = (
        json.loads((r / "metadata.json").read_text()) for r in [wave_run, qc_run, geo_run]
    )
    wave_path = repo / wm["interim_directory"] / "waveform_qc.parquet"
    mask_path = repo / qm["processed_directory"] / "fixed_qc_mask.parquet"
    geometry_path = repo / gm["interim_directory"] / "trace_geometry.parquet"
    expected = {
        wave_path: wm["waveform_table_sha256"],
        mask_path: qm["artifacts"][mask_path.relative_to(repo).as_posix()],
        geometry_path: gm["artifacts"][geometry_path.relative_to(repo).as_posix()],
    }
    lock = {}
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"input hash mismatch: {path}")
        lock[path.relative_to(repo).as_posix()] = actual
    for run in [wave_run, qc_run, geo_run]:
        path = run / "metadata.json"
        lock[path.relative_to(repo).as_posix()] = sha256_file(path)
    for run in [qc_run, geo_run]:
        path = run / "input.lock.json"
        lineage = json.loads(path.read_text())
        if lineage[wave_path.relative_to(repo).as_posix()] != expected[wave_path]:
            raise ValueError("waveform lineage mismatch")
        lock[path.relative_to(repo).as_posix()] = sha256_file(path)
    columns = [
        "source_file",
        "trace_index",
        "ffid",
        "source_line",
        "source_point",
        "receiver_line",
        "receiver_point",
        "source_x_m",
        "source_y_m",
        "receiver_x_m",
        "receiver_y_m",
        "trace_identification_code",
        "header_eligible",
        "numerically_usable",
        "waveform_status",
        "sample_count",
        "sample_interval_us",
    ]
    wave = pd.read_parquet(wave_path, columns=columns)
    mask = pd.read_parquet(mask_path)
    keys = ["source_file", "trace_index"]
    if not wave[keys].equals(mask[keys]) or wave.duplicated(keys).any():
        raise ValueError("QC mask trace keys/order mismatch")
    if not mask.header_eligible.equals(wave.header_eligible):
        raise ValueError("header eligibility mismatch")
    if not mask.eligible_after_fixed_qc.equals(wave.header_eligible & wave.numerically_usable):
        raise ValueError("fixed QC eligibility mismatch")
    if (
        mask.loc[mask.fixed_qc_excluded, "exclusion_reason"].value_counts().to_dict()
        != inputs["expected_fixed_exclusions"]
    ):
        raise ValueError("fixed exclusion counts differ from authorized input")
    for key in [
        "eligible_after_fixed_qc",
        "fixed_qc_excluded",
        "exclusion_reason",
        "review_pending",
        "dc_review",
        "amplitude_review",
        "near_constant_review",
    ]:
        wave[key] = mask[key]
    domain, sources, receivers, observed = prepare_geometry(wave)
    geometry = pd.read_parquet(geometry_path, columns=keys)
    if len(geometry) != len(observed) or not pd.MultiIndex.from_frame(geometry).equals(
        pd.MultiIndex.from_frame(observed[keys])
    ):
        raise ValueError("geometry eligible trace keys/order mismatch")
    axes = observed[["sample_count", "sample_interval_us"]].drop_duplicates()
    if len(axes) != 1:
        raise ValueError("inconsistent time axes")
    return domain, sources, receivers, int(axes.iloc[0, 0]), int(axes.iloc[0, 1]), lock


def run_forge_region_search(repo: Path, study: Path) -> Path:
    started = datetime.now(timezone.utc)
    config = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    domain, sources, receivers, samples, dt_us, lock = _load_inputs(repo, inputs)
    status = availability_matrix(domain, len(sources), len(receivers))
    # The current fixed mask, including pending diagnostics, is authoritative.
    si, ri = domain.source_id.to_numpy(), domain.receiver_id.to_numpy()
    pending = domain.eligible_after_fixed_qc & domain.review_pending
    status[si[pending], ri[pending]] = 3
    flags = {}
    for flag in ["dc_review", "amplitude_review", "near_constant_review"]:
        flags[flag] = np.zeros_like(status, dtype=bool)
        flags[flag][si, ri] = domain[flag]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + commit[:12]
    run = repo / "runs" / study.name / run_id
    interim = repo / "data/interim/forge_2017" / run_id
    (run / "figures").mkdir(parents=True, exist_ok=False)
    (interim / "candidates").mkdir(parents=True, exist_ok=False)
    for name in ["config", "inputs"]:
        shutil.copyfile(study / f"{name}.yaml", run / f"{name}.resolved.yaml")
    for relative in [
        "src/seis_interp/processing/forge_region_search.py",
        "src/seis_interp/processing/forge_geometry_qc.py",
        "src/seis_interp/processing/geometry.py",
        "src/seis_interp/visualization/forge_region_search.py",
        "src/seis_interp/pipelines/search_forge_regions.py",
        "src/seis_interp/data/forge_headers.py",
        "scripts/search_forge_regions.py",
    ]:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[3] / relative, target)
    _json(run / "input.lock.json", lock)
    all_pairs, all_finalists, station_results, grid_definitions = [], [], [], {}
    for size, shapes in config["sizes"].items():
        candidates = {}
        for role, stations in [("source", sources), ("receiver", receivers)]:
            all_regions, selected = search_station_regions(stations, role, shapes[role], config)
            candidates[role] = {r.metrics["candidate_id"]: r for r in selected}
            station_results.extend(
                {
                    "size": size,
                    **r.metrics,
                    "shortlisted": r.metrics["candidate_id"] in candidates[role],
                }
                for r in all_regions
            )
            print(
                f"{size} {role}: {len(all_regions)} regions; {len(selected)} shortlisted",
                flush=True,
            )
        rows = []
        for source in candidates["source"].values():
            for receiver in candidates["receiver"].values():
                row = compare_region_pair(source, receiver, status, config, samples)
                row.update(pair_characteristics(source, receiver, status, flags))
                row.update(
                    size=size,
                    candidate_id=f"{size}_{source.metrics['candidate_id']}_{receiver.metrics['candidate_id']}",
                )
                rows.append(row)
        if not rows:
            raise ValueError(f"no candidate combinations for size {size}")
        ranked = rank_region_pairs(pd.DataFrame(rows))
        finalists = select_finalists(
            ranked, candidates["source"], candidates["receiver"], status, config
        )
        finalists["size_rank"] = np.arange(1, len(finalists) + 1)
        all_pairs.append(ranked)
        all_finalists.append(finalists)
        for row in finalists.to_dict("records"):
            source = candidates["source"][row["source_candidate"]]
            receiver = candidates["receiver"][row["receiver_candidate"]]
            name = row["candidate_id"]
            out = interim / "candidates" / name
            out.mkdir()
            mapping = region_trace_mapping(domain, source, receiver)
            mapping.to_parquet(out / "trace_mapping.parquet", index=False)
            for role, region in [("source", source), ("receiver", receiver)]:
                region.stations.to_parquet(out / f"{role}_stations.parquet", index=False)
            _, _, counts, recorded, support = pair_cell_counts(source, receiver, status)
            cells = pd.DataFrame(
                {
                    "grid_cell": np.arange(len(counts)),
                    "retained_count": counts,
                    "recorded_count": recorded,
                    "station_pair_support": support,
                }
            )
            cells.to_parquet(out / "grid_cells.parquet", index=False)
            grid_definitions[name] = {
                "source": source.metrics,
                "receiver": receiver.metrics,
                "trace_mapping": (out / "trace_mapping.parquet").relative_to(repo).as_posix(),
            }
            plot_region_geometry(source, receiver, run / "figures" / f"{name}_geometry.png")
            plot_region_occupancy(
                source,
                receiver,
                counts,
                recorded,
                support,
                run / "figures" / f"{name}_occupancy.png",
            )
        print(
            f"{size}: {len(ranked)} pairs, "
            f"{int(ranked.meets_guidelines.sum())} meet guidelines; {len(finalists)} finalists",
            flush=True,
        )
    pairs, finalists = (
        pd.concat(all_pairs, ignore_index=True),
        pd.concat(all_finalists, ignore_index=True),
    )
    pairs.to_csv(run / "candidate_metrics.csv", index=False)
    finalists.to_csv(run / "finalists.csv", index=False)
    pd.DataFrame(station_results).to_csv(run / "station_candidates.csv", index=False)
    _json(run / "candidate_grids.json", grid_definitions)
    plot_search_tradeoffs(pairs, finalists, run / "figures/tradeoffs.png")
    _json(
        run / "summary.json",
        {
            "complete": True,
            "station_candidate_count": len(station_results),
            "pair_candidate_count": len(pairs),
            "finalist_count": len(finalists),
            "per_size": [
                {
                    "size": size,
                    "pair_count": len(group),
                    "meets_guidelines": int(group.meets_guidelines.sum()),
                }
                for size, group in pairs.groupby("size", sort=False)
            ],
            "fixed_exclusions": inputs["expected_fixed_exclusions"],
            "eligible_trace_count": int(domain.eligible_after_fixed_qc.sum()),
            "unknown_coordinate_receivers": receivers.loc[
                receivers.receiver_x_m.isna(), "receiver_id"
            ].tolist(),
            "time_samples": samples,
            "sample_interval_us": dt_us,
            "selection": "geometry first; QC diagnostics only; final benchmark adoption pending",
            "search": "coarse/fine deterministic shortlist; not exhaustive global optimization",
        },
    )
    artifacts = [*interim.rglob("*.parquet"), *run.glob("*.csv"), run / "candidate_grids.json"]
    _json(
        run / "metadata.json",
        {
            "git_commit": commit,
            "seed": None,
            "deterministic": True,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "interim_directory": interim.relative_to(repo).as_posix(),
            "artifacts": {p.relative_to(repo).as_posix(): sha256_file(p) for p in artifacts},
        },
    )
    print(f"Region search saved: {run.relative_to(repo)}", flush=True)
    return run
