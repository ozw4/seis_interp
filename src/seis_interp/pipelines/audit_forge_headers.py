"""Run a reproducible, header-only audit of the local FORGE subset."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml

from seis_interp.data.forge_archives import verify_forge_archive
from seis_interp.data.forge_headers import read_forge_headers, sha256_file
from seis_interp.data.forge_navigation import read_forge_navigation
from seis_interp.processing.forge_header_qc import (
    attach_navigation,
    check_forge_file_headers,
    classify_forge_headers,
    mark_duplicate_headers,
    summarize_forge_audit,
    summarize_forge_stations,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def run_forge_header_audit(repo: Path, study: Path) -> Path:
    """Create a fresh run and interim table; continue past file failures and report them."""
    started = datetime.now(timezone.utc)
    config = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    root = repo / inputs["dataset_root"]
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    seismic_paths = [p for p in paths if p.suffix.lower() in {".sgy", ".segy"}]
    if not seismic_paths:
        raise ValueError(f"no SEG-Y files in {root}")
    commit = _git(repo, "rev-parse", "HEAD")
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + commit[:12]
    run = repo / "runs" / study.name / run_id
    interim = repo / "data/interim/forge_2017" / run_id
    run.mkdir(parents=True, exist_ok=False)
    interim.mkdir(parents=True, exist_ok=False)
    for name in ("config.yaml", "inputs.yaml"):
        shutil.copyfile(study / name, run / name.replace(".yaml", ".resolved.yaml"))
    source_repo = Path(__file__).resolve().parents[3]
    source_paths = [
        Path(__file__),
        source_repo / "src/seis_interp/data/forge_headers.py",
        source_repo / "src/seis_interp/data/forge_archives.py",
        source_repo / "src/seis_interp/data/forge_navigation.py",
        source_repo / "src/seis_interp/processing/forge_header_qc.py",
        source_repo / "src/seis_interp/processing/geometry.py",
        source_repo / "scripts/audit_forge_headers.py",
    ]
    for path in source_paths:
        target = run / "source" / path.relative_to(source_repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    inventory = [
        {
            "path": p.relative_to(root).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in paths
    ]
    _write_json(run / "input.lock.json", inventory)
    archive_checks = []
    for archive in inputs.get("download_archives", []):
        print(f"Verifying ZIP and extracted files: {archive['path']}", flush=True)
        check = verify_forge_archive(root / archive["path"], root / archive["destination"])
        archive_checks.append({**archive, **check})
    _write_json(run / "archive_checks.json", archive_checks)
    navigation = {
        role: read_forge_navigation(root / inputs["navigation"][role])
        for role in ("source", "receiver")
    }
    for role, nav in navigation.items():
        nav.to_csv(interim / (role + "_navigation.csv"), index=False)
    tables, files, failures, file_headers = [], [], [], []
    (run / "text_headers").mkdir()
    for index, path in enumerate(seismic_paths):
        relative = path.relative_to(root).as_posix()
        try:
            metadata, raw = read_forge_headers(path)
            table = classify_forge_headers(raw, metadata)
            table.insert(0, "source_file", relative)
            issues = check_forge_file_headers(table, metadata, path)
            if issues:
                table["header_eligible"] = False
                table["header_issues"] = (
                    table.header_issues + ";file_header_inconsistency"
                ).str.strip(";")
            for role, nav in navigation.items():
                table = attach_navigation(table, nav, role, config["navigation_tolerance_m"])
            raw_file_header = metadata.pop("raw_file_header")
            text_header = metadata.pop("text_header")
            text_path = run / "text_headers" / (relative + ".txt")
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text_header)
            files.append(
                {
                    "source_file": relative,
                    **metadata,
                    "issues": issues,
                    "ffid": int(table.ffid.iloc[0]),
                    "source_line": (
                        float(table.source_line.iloc[0])
                        if pd.notna(table.source_line.iloc[0])
                        else None
                    ),
                    "source_point": (
                        float(table.source_point.iloc[0])
                        if pd.notna(table.source_point.iloc[0])
                        else None
                    ),
                }
            )
            tables.append(table)
            file_headers.append({"source_file": relative, "raw_file_header": raw_file_header})
        except (ValueError, OSError) as error:
            failures.append({"source_file": relative, "error": str(error)})
        if (index + 1) % 50 == 0:
            print(f"Audited {index + 1}/{len(seismic_paths)} files", flush=True)
    if not tables:
        _write_json(run / "failures.json", failures)
        raise RuntimeError(f"no readable SEG-Y files; see {run}")
    table = mark_duplicate_headers(pd.concat(tables, ignore_index=True))
    table.to_parquet(interim / "trace_headers.parquet", index=False)
    pd.DataFrame(file_headers).to_parquet(interim / "file_headers.parquet", index=False)
    file_counts = (
        table.groupby("source_file")
        .agg(
            header_eligible_count=("header_eligible", "sum"),
            receiver_xy_zero_count=("receiver_xy_zero", "sum"),
        )
        .reset_index()
    )
    pd.DataFrame(files).merge(file_counts, on="source_file", validate="one_to_one").to_csv(
        run / "files.csv", index=False
    )
    table.groupby(
        ["source_file", "ffid", "trace_identification_code", "trace_class", "header_eligible"]
    ).size().rename("trace_count").reset_index().to_csv(
        run / "classification_by_file.csv", index=False
    )
    summary = summarize_forge_audit(table, files, inputs["expected_ffid_range"])
    source_stations, receiver_stations, station_counts = summarize_forge_stations(table)
    source_stations.to_csv(run / "source_stations.csv", index=False)
    receiver_stations.to_csv(run / "receiver_stations.csv", index=False)
    summary["station_counts"] = station_counts
    summary.update(
        failed_files=failures,
        discovered_segy_count=len(seismic_paths),
        complete=not failures,
        verified_archive_count=len(archive_checks),
        navigation_record_counts={r: len(n) for r, n in navigation.items()},
    )
    _write_json(run / "summary.json", summary)
    _write_json(
        run / "metadata.json",
        {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_status": _git(repo, "status", "--short"),
            "seed": None,
            "deterministic": True,
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ["numpy", "pandas", "pyarrow", "PyYAML"]},
            "interim_directory": interim.relative_to(repo).as_posix(),
            "artifacts": {
                p.relative_to(repo).as_posix(): sha256_file(p) for p in interim.iterdir()
            },
        },
    )
    print(f"Audit saved: {run.relative_to(repo)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} files failed; see {run / 'summary.json'}")
    return run
