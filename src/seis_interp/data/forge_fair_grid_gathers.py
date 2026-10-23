"""Verified final-evaluation predictions and selected FORGE display waveforms."""

import json

import numpy as np
import pandas as pd

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import read_selected_traces, verify_hashes
from seis_interp.forge_mvp_config import NEURAL_METHODS
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces


def load_evaluated_gather_inputs(repo, evaluation):
    """Require the completed six-method final test evaluation before opening targets."""
    record = json.loads((evaluation / "manifest.json").read_text())
    if record["status"] != "completed" or record["selection_domain"] != "final_test":
        raise ValueError("publication gathers require a completed final-test evaluation")
    if {run["method_id"] for run in record["final_runs"]} != set(METHODS):
        raise ValueError("final evaluation must contain all six methods")
    verify_hashes(evaluation, record["artifacts"])
    preparation = repo / record["preparation"]
    if sha256_file(preparation / "preparation_manifest.json") != record["preparation_hash"]:
        raise ValueError("final-evaluation preparation hash mismatch")
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    index = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    if (
        not np.array_equal(index.cell_id, mask.cell_id)
        or not mask.split.isin(["observed", "test"]).all()
    ):
        raise ValueError("gather mapping and final-test mask do not align")
    return record, preparation, contract, index.assign(split=mask.split.to_numpy())


def load_gather_predictions(repo, record, index, *, seed):
    """One fixed seed per neural method; never pick a run from its test score."""
    test = index.split.eq("test").to_numpy()
    predictions, sources = {}, {}
    for method in METHODS:
        selected_seed = seed if method in NEURAL_METHODS else None
        candidates = [
            run
            for run in record["final_runs"]
            if run["method_id"] == method and run["seed"] == selected_seed
        ]
        if len(candidates) != 1:
            raise ValueError(f"{method}: no unique final prediction for seed {selected_seed}")
        run = candidates[0]
        tuning = repo / run["tuning_run"]
        manifest = json.loads((tuning / "manifest.json").read_text())
        relative = f"{run['candidate']}/prediction.npy"
        if (
            manifest["status"] != "completed"
            or manifest["artifacts"][relative] != run["prediction_hash"]
        ):
            raise ValueError(f"{method}: final prediction differs from the evaluated run")
        verify_hashes(
            tuning,
            {
                relative: run["prediction_hash"],
                "prediction_index.parquet": manifest["artifacts"]["prediction_index.parquet"],
            },
        )
        saved_index = pd.read_parquet(tuning / "prediction_index.parquet").set_index("cell_id")
        positions = saved_index.reindex(index.loc[test, "cell_id"]).prediction_row
        if positions.isna().any():
            raise ValueError(f"{method}: prediction does not cover the selected test cells")
        values = np.load(tuning / relative, mmap_mode="r", allow_pickle=False)
        gathered = np.zeros((len(index), values.shape[1]), dtype=np.float32)
        gathered[test] = values[positions.to_numpy(dtype=int)]
        if not np.isfinite(gathered).all():
            raise ValueError(f"{method}: nonfinite gather prediction")
        predictions[method], sources[method] = gathered, run
    return predictions, sources


def read_gather_reference(repo, preparation, contract, index):
    """Read only fixed display traces after verifying their frozen predictions."""
    raw = json.loads((preparation / "input/raw_input_hashes.json").read_text())
    verify_hashes(repo / raw["dataset_root"], raw["hashes"])
    native = native_time_contract(contract)
    target = read_selected_traces(
        repo / raw["dataset_root"],
        index,
        native["time_sample_count"],
        native["sample_interval_us"],
    )
    return resample_forge_traces(target, contract)
