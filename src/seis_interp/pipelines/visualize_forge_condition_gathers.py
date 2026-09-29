"""Frozen-checkpoint inference and fixed-geometry gather visualization."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs, verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces, verify_saved_metrics
from seis_interp.training.forge_condition_prediction import predict_regsi_condition
from seis_interp.visualization.forge_condition_gathers import plot_condition_gather


def visualize_forge_condition_gathers(repo: Path, source: Path, *, device="cuda:1"):
    """Use predeclared source 112/receiver 144, with native-RMS and physical views."""
    record = json.loads((source / "manifest.json").read_text())
    if record["status"] != "completed" or record["normalization"] != "trace_oracle":
        raise ValueError("expected a completed trace-oracle condition")
    verify_hashes(source, record["artifacts"])
    cache = repo / record["cache"]
    cache_record = json.loads((cache / "manifest.json").read_text())
    factor = record["factor"]
    names = ["geometry.parquet", f"waveforms_q{factor}.npy", "waveforms_q1.npy"]
    verify_hashes(cache, {name: cache_record["artifacts"][name] for name in names})
    mask = pd.read_parquet(source / "mask.parquet")
    geometry = pd.read_parquet(cache / "geometry.parquet")
    if not np.array_equal(mask.cell_id, geometry.cell_id):
        raise ValueError("mask and geometry rows do not align")
    selected = mask.source_cell.eq(112) | mask.receiver_cell.eq(144)
    sections = mask.loc[selected].copy().reset_index(drop=True)
    observed = mask.split.eq("observed").to_numpy()
    query_cells = sections.loc[sections.split.eq("test"), "cell_id"].to_numpy()
    saved_index = pd.read_parquet(source / "evaluation_index.parquet")
    saved_cells = saved_index.cell_id.to_numpy()
    inference_cells = np.concatenate([saved_cells, query_cells[~np.isin(query_cells, saved_cells)]])
    values = np.load(cache / f"waveforms_q{factor}.npy", mmap_mode="r")
    observed_values = values[observed].astype(float)
    observed_rms = np.sqrt(np.mean(observed_values**2, axis=1))
    unit = (observed_values / observed_rms[:, None]).astype(np.float32)
    time_s = np.arange(values.shape[1]) * factor * cache_record["original_sample_interval_us"] / 1e6
    inputs = ForgeModelInputs(
        geometry=geometry,
        observed_cells=mask.loc[observed, "cell_id"].to_numpy(),
        test_cells=mask.loc[~observed, "cell_id"].to_numpy(),
        observed_values=unit,
        spatial_shape=tuple(cache_record["spatial_shape"]),
        time_s=time_s,
        global_rms=1.0,
        binding={"cache_hash": sha256_file(cache / "manifest.json")},
    )
    config = yaml.safe_load((source / "config.yaml").read_text())
    config["device"] = device
    state = torch.load(source / "state_ema.pt", map_location="cpu", weights_only=True)
    prediction = predict_regsi_condition(inputs, config, inference_cells, state)
    query_rows = pd.Index(inference_cells).get_indexer(query_cells)
    gather_prediction = prediction.prediction[query_rows]
    # Evaluation/display boundary: held-out waveforms are first opened here.
    positions = pd.Index(mask.cell_id).get_indexer(sections.cell_id)
    target = values[positions].astype(float)
    rms = np.sqrt(np.mean(target**2, axis=1))
    test_rows = np.flatnonzero(sections.split.eq("test"))
    reconstructed = target.copy()
    reconstructed[test_rows] = gather_prediction.astype(float) * rms[test_rows, None]
    reconstructed = reconstructed.astype(np.float32)
    target = target.astype(np.float32)
    native = np.load(cache / "waveforms_q1.npy", mmap_mode="r")[positions].astype(float)
    native_rms = np.sqrt(np.mean(native**2, axis=1))
    shared, new_rows, old_rows = np.intersect1d(inference_cells, saved_cells, return_indices=True)
    old_unit = np.load(source / "prediction_ema.npy")[old_rows].astype(float)
    old_unit /= np.load(source / "evaluation_rms.npy")[old_rows, None]
    if len(shared):
        np.testing.assert_allclose(prediction.prediction[new_rows], old_unit, rtol=1e-4, atol=5e-6)
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        repo
        / "runs/study_054_forge_regsi_conditions"
        / (
            f"{stamp}_{identity['git_commit'][:12]}_q{factor}_"
            f"missing{record['missing_fraction']:g}_gathers"
        )
    )
    output.mkdir(parents=True)
    sections.to_parquet(output / "gather_index.parquet", index=False)
    np.savez_compressed(
        output / "gathers.npz",
        target=target.astype(np.float32),
        reconstructed=reconstructed.astype(np.float32),
        native_rms=native_rms,
        time_s=time_s,
        query_cells=query_cells,
        prediction_unit=gather_prediction,
    )
    metrics, per_trace = evaluate_traces(
        reconstructed[test_rows], target[test_rows], sections.iloc[test_rows].reset_index(drop=True)
    )
    verify_saved_metrics(metrics, per_trace)
    write_json(output / "metrics.json", metrics)
    per_trace.to_parquet(output / "metrics_by_trace.parquet", index=False)
    for role, cell, count in (("source", 112, 256), ("receiver", 144, 192)):
        other = "receiver" if role == "source" else "source"
        rows = np.flatnonzero(sections[f"{role}_cell"].eq(cell))
        cells = sections.iloc[rows][f"{other}_cell"].to_numpy()
        np.testing.assert_array_equal(cells, np.arange(count))
        is_observed = sections.iloc[rows].split.eq("observed").to_numpy()
        title = (
            f"ReGSI-real EMA | {record['missing_fraction']:.0%} missing | {factor} ms"
            f" | common {role} {cell} | {int((~is_observed).sum())}/{count} traces predicted"
        )
        for mode, scale, clip, label in (
            ("rms", native_rms[rows], 3.0, "Amplitude / native target RMS"),
            (
                "physical",
                np.ones(len(rows)),
                float(np.percentile(np.abs(native[rows]), 99)),
                "Stored amplitude units",
            ),
        ):
            plot_condition_gather(
                target[rows],
                reconstructed[rows],
                is_observed,
                time_s,
                cells,
                scale,
                output / f"common_{role}_{mode}.png",
                title=title,
                scale_label=label,
                clip=clip,
            )
        keep = time_s <= 1.5
        plot_condition_gather(
            target[rows][:, keep],
            reconstructed[rows][:, keep],
            is_observed,
            time_s[keep],
            cells,
            native_rms[rows],
            output / f"common_{role}_rms_zoom.png",
            title=title + " | 0-1.5 s detail",
            scale_label="Amplitude / native target RMS",
            clip=3.0,
        )
    for relative in (
        "src/seis_interp/pipelines/visualize_forge_condition_gathers.py",
        "src/seis_interp/training/forge_condition_prediction.py",
        "src/seis_interp/visualization/forge_condition_gathers.py",
        "scripts/forge_condition_gathers.py",
    ):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / relative).read_bytes())
    write_json(
        output / "manifest.json",
        {
            **identity,
            "status": "completed",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "source_run": str(source.relative_to(repo)),
            "source_manifest_hash": sha256_file(source / "manifest.json"),
            "checkpoint_hash": sha256_file(source / "state_ema.pt"),
            "cache_manifest_hash": sha256_file(cache / "manifest.json"),
            "factor": factor,
            "missing_fraction": record["missing_fraction"],
            "source_cell": 112,
            "receiver_cell": 144,
            "query_count": len(query_cells),
            "inference_query_count": len(inference_cells),
            "no_context_queries": int((~prediction.has_observed_context[query_rows]).sum()),
            "probe_batch_preserved": True,
            "overlapping_saved_probe_count": len(shared),
            "overlapping_predictions_verified": bool(len(shared)),
            "test_waveforms_read_only_after_prediction": True,
            "oracle_rms_restoration": True,
            "normalised_display": "all panels divided by same per-trace native target RMS",
            "physical_clip": "per gather, native target absolute amplitude 99th percentile",
            "artifacts": {
                str(p.relative_to(output)): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(output, flush=True)
    return output
