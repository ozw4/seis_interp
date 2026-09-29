"""Re-render fixed sections from an evaluated matrix into a separate immutable run."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import read_selected_traces, verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces
from seis_interp.visualization import forge_mvp


def visualize_forge_mvp(repo: Path, source: Path) -> Path:
    """Read saved physical predictions; open targets only for an accepted matrix."""
    start_time = utc_now()
    final_path = source / "aggregate/final_mvp_manifest.json"
    final = json.loads(final_path.read_text())
    if final["status"] != "accepted" or final["completed_methods"] != list(METHODS):
        raise ValueError("section visualization requires an accepted six-method matrix")
    verify_hashes(source, final["artifacts"])
    matrix = json.loads((source / "matrix_manifest.json").read_text())
    preparation = repo / matrix["preparation"]
    preparation_hash = sha256_file(preparation / "preparation_manifest.json")
    if (
        matrix["methods"] != list(METHODS)
        or matrix["preparation_hash"] != preparation_hash
        or final["preparation_hash"] != preparation_hash
    ):
        raise ValueError("section visualization preparation mismatch")
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    manifest = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    if not np.array_equal(manifest.cell_id, mask.cell_id):
        raise ValueError("section mask and trace mapping do not align")
    selected = manifest.source_cell.eq(
        contract["visualization"]["source_cell"]
    ) | manifest.receiver_cell.eq(contract["visualization"]["receiver_cell"])
    manifest = manifest.loc[selected].reset_index(drop=True)
    mask = mask.loc[selected].reset_index(drop=True)
    test = mask.split.eq("test").to_numpy()
    predictions = {}
    for method in METHODS:
        path = source / "runs" / method
        index = pd.read_parquet(path / "prediction_index.parquet").set_index("cell_id")
        positions = index.loc[mask.loc[test, "cell_id"], "prediction_row"].to_numpy()
        predictions[method] = np.load(path / "prediction.npy", mmap_mode="r")[positions]
    raw = json.loads((preparation / "input/raw_input_hashes.json").read_text())
    verify_hashes(repo / raw["dataset_root"], raw["hashes"])
    native = native_time_contract(contract)
    # Excluded (validation-preparation parent test) traces are never read; they stay zero.
    readable = mask.split.ne("excluded").to_numpy()
    truth = np.zeros((len(manifest), contract["time_sample_count"]), dtype=np.float32)
    truth[readable] = resample_forge_traces(
        read_selected_traces(
            repo / raw["dataset_root"],
            manifest.loc[readable],
            native["time_sample_count"],
            native["sample_interval_us"],
        ),
        contract,
    )
    for method in METHODS:
        section = truth.copy()
        section[test] = predictions[method]
        predictions[method] = section
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo
        / "runs"
        / seal["study_id"]
        / f"{stamp}_{identity['git_commit'][:12]}_trace_rms_figures"
    )
    (output / "figures").mkdir(parents=True, exist_ok=False)
    forge_mvp.plot_sections(manifest, mask, truth, predictions, contract, output / "figures")
    write_json(output / "input_contract.json", contract)
    (output / "source").mkdir()
    for path in (Path(__file__), Path(forge_mvp.__file__)):
        (output / "source" / path.name).write_bytes(path.read_bytes())
    write_json(
        output / "manifest.json",
        {
            **identity,
            "status": "completed",
            "start_time": start_time,
            "end_time": utc_now(),
            "source_run": str(source.relative_to(repo)),
            "source_manifest_hash": sha256_file(final_path),
            "preparation": str(preparation.relative_to(repo)),
            "preparation_hash": preparation_hash,
            "normalization": "all panels share full-duration reference trace RMS after resampling",
            "zero_rms_divisor": 1.0,
            "artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return output
