"""Orchestrate the input-bound FORGE waveform audit and diagnostic figures."""

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
from seis_interp.data.forge_waveforms import read_forge_gather
from seis_interp.processing.forge_waveform_qc import (
    flag_relative_amplitude,
    select_representative_shots,
    summarize_waveform_qc,
    waveform_metrics,
)
from seis_interp.visualization.forge_waveform_qc import (
    plot_gather,
    plot_qc_overview,
    plot_review_traces,
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_forge_waveform_audit(repo: Path, study: Path) -> Path:
    """Fail on changed input; never modify header eligibility or source amplitudes."""
    started = datetime.now(timezone.utc)
    config = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    header_run = repo / inputs["header_run"]
    metadata = json.loads((header_run / "metadata.json").read_text())
    header_path = repo / metadata["interim_directory"] / "trace_headers.parquet"
    if sha256_file(header_path) != metadata["artifacts"][header_path.relative_to(repo).as_posix()]:
        raise ValueError("header artifact hash mismatch")
    cols = [
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
        "sample_count",
        "sample_interval_us",
    ]
    headers = pd.read_parquet(header_path, columns=cols)
    headers["offset_m"] = np.hypot(
        headers.receiver_x_m - headers.source_x_m, headers.receiver_y_m - headers.source_y_m
    )
    representatives = select_representative_shots(headers)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + commit[:12]
    run = repo / "runs" / study.name / run_id
    interim = repo / "data/interim/forge_2017" / run_id
    (run / "figures").mkdir(parents=True, exist_ok=False)
    interim.mkdir(parents=True, exist_ok=False)
    for name in ["config.yaml", "inputs.yaml"]:
        shutil.copyfile(study / name, run / name.replace(".yaml", ".resolved.yaml"))
    source_repo = Path(__file__).resolve().parents[3]
    for relative in [
        "src/seis_interp/pipelines/audit_forge_waveforms.py",
        "src/seis_interp/data/forge_waveforms.py",
        "src/seis_interp/data/forge_headers.py",
        "src/seis_interp/processing/forge_waveform_qc.py",
        "src/seis_interp/visualization/forge_waveform_qc.py",
        "scripts/audit_forge_waveforms.py",
    ]:
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_repo / relative, destination)
    lock = json.loads((header_run / "input.lock.json").read_text())
    expected = {entry["path"]: entry for entry in lock}
    root = repo / inputs["dataset_root"]
    rows, spectra, checked, by_file_spectra = [], None, [], []
    for i, (filename, group) in enumerate(headers.groupby("source_file", sort=True)):
        group = group.sort_values("trace_index").reset_index(drop=True)
        path = root / filename
        if (
            path.stat().st_size != expected[filename]["size_bytes"]
            or sha256_file(path) != expected[filename]["sha256"]
        ):
            raise ValueError(f"source file changed since header audit: {filename}")
        checked.append(expected[filename])
        values, dt = read_forge_gather(path, group)
        stats, aggregate = waveform_metrics(values, dt, config, group.header_eligible.to_numpy())
        rows.append(pd.concat([group, stats], axis=1))
        if spectra is None:
            spectra = {**aggregate, "dt_s": dt}
        else:
            if dt != spectra["dt_s"] or not np.array_equal(
                aggregate["frequency_hz"], spectra["frequency_hz"]
            ):
                raise ValueError("incompatible time/frequency grids across files")
            for key in [
                "power_sum",
                "normalized_power_sum",
                "spectrum_count",
                "time_energy_sum",
                "finite_trace_count",
            ]:
                spectra[key] += aggregate[key]
        by_file_spectra.append(
            {
                "source_file": filename,
                "ffid": int(group.ffid.iloc[0]),
                "spectrum_count": aggregate["spectrum_count"],
            }
        )
        if int(group.ffid.iloc[0]) in representatives:
            plot_gather(
                values, group, dt, run / "figures" / f"gather_{int(group.ffid.iloc[0]):03d}.png"
            )
        if (i + 1) % 25 == 0:
            print(f"Waveform audit {i + 1}/{headers.source_file.nunique()} files", flush=True)
    table = flag_relative_amplitude(pd.concat(rows, ignore_index=True), config)
    table.to_parquet(interim / "waveform_qc.parquet", index=False)
    candidate = table[table.header_eligible]
    candidate.groupby(["source_file", "ffid", "waveform_status"]).size().rename(
        "count"
    ).reset_index().to_csv(run / "status_by_file.csv", index=False)
    candidate[candidate.waveform_status.ne("passed_numeric_checks")].to_csv(
        run / "flagged_candidates.csv", index=False
    )
    table.groupby(
        ["trace_identification_code", "all_zero", "constant", "numerically_usable"]
    ).size().rename("count").reset_index().to_csv(run / "status_by_code.csv", index=False)
    candidate.groupby(["receiver_line", "receiver_point"]).agg(
        trace_count=("ffid", "size"),
        all_zero_count=("all_zero", "sum"),
        amplitude_review_count=("amplitude_review", "sum"),
        median_rms=("rms", "median"),
    ).to_csv(run / "receiver_qc.csv")
    # Rank examples only after the full QC; these are diagnostic, not benchmark selection.
    review = candidate.nlargest(4, "rms_to_shot_offset_median")
    review = pd.concat([review, candidate[candidate.all_zero].head(2)]).drop_duplicates(
        ["source_file", "trace_index"]
    )
    review.to_csv(run / "review_examples.csv", index=False)
    for filename, selected in review.groupby("source_file"):
        group = (
            headers[headers.source_file.eq(filename)]
            .sort_values("trace_index")
            .reset_index(drop=True)
        )
        values, dt = read_forge_gather(root / filename, group)
        plot_review_traces(
            values,
            group,
            selected.trace_index.astype(int).tolist(),
            dt,
            run / "figures" / f"review_{int(group.ffid.iloc[0]):03d}.png",
        )
    np.savez_compressed(run / "aggregate_spectra.npz", **spectra)
    plot_qc_overview(table, spectra, run / "figures/overview.png")
    summary = summarize_waveform_qc(table)
    summary.update(
        file_count=len(checked),
        representative_ffids=representatives,
        spectrum_trace_count=int(spectra["spectrum_count"]),
        complete=True,
    )
    _json(run / "summary.json", summary)
    _json(
        run / "input.lock.json",
        {"segy_files": checked, "header_table_sha256": sha256_file(header_path)},
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
                p: version(p) for p in ["numpy", "pandas", "segyio", "matplotlib", "pyarrow"]
            },
            "interim_directory": interim.relative_to(repo).as_posix(),
            "waveform_table_sha256": sha256_file(interim / "waveform_qc.parquet"),
        },
    )
    print(f"Audit saved: {run.relative_to(repo)}", flush=True)
    return run
