"""Small recorded PoC runs with opaque artifacts for artifact-only comparison tests."""

import json
from copy import deepcopy
from pathlib import Path

from seis_interp.data.file_checksums import file_sha256


def poc_input_lock() -> dict[str, object]:
    input_files = {
        group: {name: {"sha256": "a" * 64} for name in names}
        for group, names in (
            ("interim", ("traces.parquet", "amplitudes.npy", "time_s.npy", "dataset.json")),
            ("processed", ("trace_split.parquet", "normalization.json", "preparation.json")),
            ("mask", ("observation_mask.parquet", "interpolation_mask.json")),
        )
    }
    return {
        "benchmark_id": "c3_sl25_40_random80_observed_only_v1",
        "dataset_id": "seg_c3_na",
        "case_id": "random80_case",
        "volume_id": "fixed_volume",
        "selection": {
            "time": [0, 384],
            "source_line": [25, 41],
            "shot_in_line": [28, 60],
            "relative_receiver_x": [0, 8],
            "relative_receiver_y": [18, 50],
        },
        "shape": [384, 16, 32, 8, 32],
        "observed_trace_count": 26214,
        "target_trace_count": 104858,
        "mask": {
            "kind": "random_trace",
            "missing_fraction": 0.8,
            "random_seed": 42,
            "unit": "complete_trace",
            "files": deepcopy(input_files["mask"]),
        },
        "benchmark_case": {
            "case_id": "random80_case",
            "file": "benchmark_case.json",
            "sha256": "b" * 64,
            "input_files": input_files,
        },
        "benchmark_volume": {
            "volume_id": "fixed_volume",
            "files": {
                "volume_index.parquet": {"sha256": "c" * 64},
                "volume.json": {"sha256": "d" * 64},
            },
        },
    }


def write_poc_run_artifacts(
    directory: Path, method: str, *, inputs_lock: dict[str, object] | None = None
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = poc_input_lock() if inputs_lock is None else deepcopy(inputs_lock)
    neural = method not in {"pocs", "drr"}
    artifacts = {"checkpoint": None}
    for name, file_name in (
        [("prediction", "prediction.npy"), ("checkpoint", "final.pt")]
        if neural
        else [("prediction", "prediction.npy")]
    ):
        path = directory / file_name
        path.write_bytes(f"opaque {method} {name} artifact".encode())
        artifacts[name] = {"path": file_name, "sha256": file_sha256(path)}
    operation = (
        "reconstruction_seconds"
        if not neural
        else "fit_seconds"
        if method == "nersi"
        else "training_seconds"
    )
    timing = {operation: 2.0, "evaluation_seconds": 0.5, "end_to_end_seconds": 4.0}
    if neural:
        timing["prediction_seconds"] = 1.0
    target_count = lock["target_trace_count"]
    metadata = {
        "method": method,
        "status": "success",
        **{key: lock[key] for key in ("benchmark_id", "case_id", "volume_id")},
        "coverage": {
            "target_trace_count": target_count,
            "covered_target_trace_count": target_count,
            "target_coverage_fraction": 1.0,
            "uncovered_trace_count": 0,
            "uncovered_sample_count": 0,
            "complete": True,
            "boundary_targets_included": True,
        },
        "compute": {
            "parameter_count": 123 if neural else None,
            "optimizer_updates": 3 if neural else None,
            "supervised_trace_presentations": 10 if neural else None,
        },
        "artifacts": artifacts,
        "timing": timing,
        "resource_usage": {"process_max_rss_kib": 1024},
    }
    metrics = {
        "evaluation_domain": "evaluation_target",
        "amplitude_domain": "physical",
        "evaluation_target": {
            "target_trace_count": target_count,
            "covered_target_trace_count": target_count,
            "snr_db": 10.0,
            "snr_status": "finite",
            "rmse": 0.1,
            "relative_l2": 0.31622776601683794,
            "mean_trace_relative_mse": 0.1,
        },
    }
    for file_name, record in (
        ("inputs.lock.json", lock),
        ("metadata.json", metadata),
        ("metrics.json", metrics),
    ):
        (directory / file_name).write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
