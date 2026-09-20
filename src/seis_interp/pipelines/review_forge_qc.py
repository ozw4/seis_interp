"""Persist fixed exclusions and evidence for unresolved FORGE waveform flags."""

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_waveforms import read_forge_gather
from seis_interp.processing.forge_qc_review import (
    KEYS,
    build_qc_review,
    measure_review_waveform,
    review_distributions,
    select_review_controls,
    select_review_examples,
    summarize_qc_review,
)
from seis_interp.visualization.forge_qc_review import (
    plot_review_distributions,
    plot_review_example,
    plot_station_sequence,
)


def _json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_forge_qc_review(repo: Path, study: Path) -> Path:
    started = datetime.now(timezone.utc)
    config = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    wave_run = repo / inputs["waveform_run"]
    metadata = json.loads((wave_run / "metadata.json").read_text())
    wave_path = repo / metadata["interim_directory"] / "waveform_qc.parquet"
    if sha256_file(wave_path) != metadata["waveform_table_sha256"]:
        raise ValueError("waveform artifact hash mismatch")
    table = build_qc_review(pd.read_parquet(wave_path), config)
    summary = summarize_qc_review(table)
    if summary["fixed_exclusions"] != inputs["expected_fixed_exclusions"]:
        raise ValueError("fixed exclusion counts differ from authorized input")
    sensitivity, stations = review_distributions(table, config)
    examples = select_review_examples(table, config)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + commit[:12]
    run = repo / "runs" / study.name / run_id
    interim = repo / "data/interim/forge_2017" / run_id
    processed = repo / "data/processed/forge_2017" / run_id
    (run / "figures/examples").mkdir(parents=True, exist_ok=False)
    interim.mkdir(parents=True, exist_ok=False)
    processed.mkdir(parents=True, exist_ok=False)
    for name in ["config.yaml", "inputs.yaml"]:
        shutil.copyfile(study / name, run / name.replace(".yaml", ".resolved.yaml"))
    source_repo = Path(__file__).resolve().parents[3]
    for relative in [
        "src/seis_interp/processing/forge_qc_review.py",
        "src/seis_interp/visualization/forge_qc_review.py",
        "src/seis_interp/pipelines/review_forge_qc.py",
        "src/seis_interp/data/forge_headers.py",
        "src/seis_interp/data/forge_waveforms.py",
        "scripts/review_forge_qc.py",
    ]:
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_repo / relative, destination)
    mask_columns = KEYS + [
        "ffid",
        "header_eligible",
        "fixed_qc_excluded",
        "exclusion_reason",
        "eligible_after_fixed_qc",
        "review_pending",
        "near_constant_review",
        "dc_review",
        "amplitude_review",
    ]
    table[mask_columns].to_parquet(processed / "fixed_qc_mask.parquet", index=False)
    review_columns = mask_columns + [
        "receiver_line",
        "receiver_point",
        "source_line",
        "source_point",
        "std_to_rms",
        "dc_to_rms",
        "rms",
        "mean",
        "std",
        "rms_to_shot_offset_median",
        "ac_rms_to_shot_offset_median",
        "ac_reference_count",
        "amplitude_direction",
    ]
    table[review_columns].to_parquet(interim / "qc_review.parquet", index=False)
    table.loc[table.fixed_qc_excluded, review_columns].to_csv(
        run / "fixed_exclusions.csv", index=False
    )
    table.loc[table.near_constant_review, review_columns].to_csv(
        run / "near_constant_candidates.csv", index=False
    )
    sensitivity.to_csv(run / "threshold_sensitivity.csv", index=False)
    stations.to_csv(run / "receiver_rates.csv", index=False)
    retained = table[table.eligible_after_fixed_qc]
    retained.groupby(["source_line", "receiver_line"]).agg(
        retained_count=("trace_index", "size"),
        dc_count=("dc_review", "sum"),
        amplitude_count=("amplitude_review", "sum"),
        near_constant_count=("near_constant_review", "sum"),
    ).to_csv(run / "line_pair_counts.csv")
    examples.to_csv(run / "representative_traces.csv", index=False)
    wave_inputs = yaml.safe_load((wave_run / "inputs.resolved.yaml").read_text())
    source_root = repo / wave_inputs["dataset_root"]
    expected = {
        x["path"]: x["sha256"]
        for x in json.loads((wave_run / "input.lock.json").read_text())["segy_files"]
    }
    bound = [
        wave_path,
        wave_run / "metadata.json",
        wave_run / "input.lock.json",
        wave_run / "inputs.resolved.yaml",
    ]
    checks = {p.relative_to(repo).as_posix(): sha256_file(p) for p in bound}
    controls_rows, measurements = [], []
    for filename, selected in examples.groupby("source_file", sort=True):
        path = source_root / filename
        digest = sha256_file(path)
        if digest != expected[filename]:
            raise ValueError(f"SEG-Y input changed: {filename}")
        checks[path.relative_to(repo).as_posix()] = digest
        gather = (
            table[table.source_file.eq(filename)].sort_values("trace_index").reset_index(drop=True)
        )
        values, dt = read_forge_gather(path, gather)
        for _, target in selected.iterrows():
            measurements.append(
                {
                    "stratum": target.review_stratum,
                    "selection": target.selection,
                    "ffid": int(target.ffid),
                    "source_file": filename,
                    "trace_index": int(target.trace_index),
                    **measure_review_waveform(values[int(target.trace_index)], dt),
                }
            )
            controls = select_review_controls(gather, target, config["controls_per_example"])
            controls_rows.extend(
                {
                    "stratum": target.review_stratum,
                    "selection": target.selection,
                    "source_file": filename,
                    "target_trace_index": int(target.trace_index),
                    "control_trace_index": int(row.trace_index),
                    "point_distance": float(row.point_distance),
                }
                for row in controls.itertuples()
            )
            name = f"{target.review_stratum}_{target.selection}_ffid{int(target.ffid)}.png"
            plot_review_example(
                values, gather, target, controls, dt, run / "figures/examples" / name
            )
    pd.DataFrame(controls_rows).to_csv(run / "representative_controls.csv", index=False)
    pd.DataFrame(measurements).to_csv(run / "representative_measurements.csv", index=False)
    plot_review_distributions(table, stations, run / "figures")
    plot_station_sequence(table, run / "figures/receiver_101_579_sequence.png")
    summary.update(
        complete=True,
        example_count=len(examples),
        example_file_count=int(examples.source_file.nunique()),
    )
    _json(run / "summary.json", summary)
    _json(run / "input.lock.json", checks)
    artifacts = [*interim.glob("*.parquet"), *processed.glob("*.parquet")]
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
                p: version(p) for p in ["numpy", "pandas", "matplotlib", "segyio", "pyarrow"]
            },
            "interim_directory": interim.relative_to(repo).as_posix(),
            "processed_directory": processed.relative_to(repo).as_posix(),
            "artifacts": {p.relative_to(repo).as_posix(): sha256_file(p) for p in artifacts},
        },
    )
    print(f"QC review saved: {run.relative_to(repo)}", flush=True)
    return run
