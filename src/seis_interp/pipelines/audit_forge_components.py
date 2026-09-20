"""Save header-grounded component evidence beside a new component audit run."""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.forge_component_evidence import summarize_component_evidence


def run_forge_component_audit(repo: Path, waveform_run: Path) -> Path:
    """Check bound input hashes and save component evidence in a new run."""
    wave_run = repo / waveform_run
    metadata = json.loads((wave_run / "metadata.json").read_text())
    wave_path = repo / metadata["interim_directory"] / "waveform_qc.parquet"
    if sha256_file(wave_path) != metadata["waveform_table_sha256"]:
        raise ValueError("waveform artifact hash mismatch")
    # Inputs are bound to the completed waveform run's saved configuration.
    import yaml

    inputs = yaml.safe_load((wave_run / "inputs.resolved.yaml").read_text())
    header_run = repo / inputs["header_run"]
    header_meta = json.loads((header_run / "metadata.json").read_text())
    header_dir = repo / header_meta["interim_directory"]
    header_paths = [header_dir / name for name in ["trace_headers.parquet", "file_headers.parquet"]]
    for path in header_paths:
        if sha256_file(path) != header_meta["artifacts"][path.relative_to(repo).as_posix()]:
            raise ValueError("header artifact hash mismatch")
    summary, groups = summarize_component_evidence(
        pd.read_parquet(
            header_paths[0],
            columns=[
                "source_file",
                "trace_index",
                "header_eligible",
                "receiver_line",
                "receiver_point",
                "raw_header",
            ],
        ),
        pd.read_parquet(header_paths[1]),
        pd.read_parquet(wave_path),
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + commit[:12] + "_components"
    )
    out = wave_run.parent / run_id
    out.mkdir(exist_ok=False)
    (out / "component_evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
    groups.to_csv(out / "vendor_group_measurements.csv", index=False)
    lock = {p.relative_to(repo).as_posix(): sha256_file(p) for p in [wave_path, *header_paths]}
    (out / "input.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    for path in [Path(__file__), repo / "src/seis_interp/processing/forge_component_evidence.py"]:
        shutil.copyfile(path, out / path.name)
    print(out.relative_to(repo))
    return out
