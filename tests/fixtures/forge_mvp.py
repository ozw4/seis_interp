"""Small synthetic SEG-Y candidate with an irregular source geometry and one hole."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import segyio
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.forge_mvp_contract import (
    METHODS,
    continuous_indices,
    grid_coordinates,
)


def tiny_forge_study(root):
    repository = Path(__file__).resolve().parents[2]
    canonical = repository / "studies/study_052_forge_m1_mvp"
    study = root / "studies/tiny_forge_mvp"
    study.mkdir(parents=True)
    config = yaml.safe_load((canonical / "config.yaml").read_text())
    config.update(
        study_id=study.name,
        candidate_id="tiny",
        spatial_shape=[2, 3, 2, 3],
        eligible_trace_count=35,
        test_trace_count=28,
        observed_trace_count=7,
        time_sample_count=9,
    )
    config["visualization"].update(source_cell=2, receiver_cell=2, time_range_s=[0, 0.008])
    grids, stations = {}, {}
    for role in ("source", "receiver"):
        definition = {
            "shape": [2, 3],
            "first_u_m": 0.0,
            "first_v_m": 0.0,
            "along_spacing_m": 50.0,
            "across_spacing_m": 200.0,
            "origin_xy_m": [0.0, 0.0],
            "direction": [1.0, 0.0],
            "normal": [0.0, 1.0],
        }
        indices = np.column_stack(np.unravel_index(np.arange(6), (2, 3)))
        projected = grid_coordinates(indices, definition)
        actual = projected.copy()
        actual[:, 0] += (np.arange(6) + 1) ** 2 * (0.7 if role == "source" else 0.03)
        grids[role] = definition
        stations[role] = pd.DataFrame(
            {
                "station_id": np.arange(6),
                "grid_line": indices[:, 0],
                "grid_point": indices[:, 1],
                "cell": np.arange(6),
                "x_m": actual[:, 0],
                "y_m": actual[:, 1],
                "grid_x_m": projected[:, 0],
                "grid_y_m": projected[:, 1],
            }
        )
    source, receiver = np.indices((6, 6))
    s, r = source.ravel(), receiver.ravel()
    mapping = pd.DataFrame(
        {
            "source_file": [f"{i}.sgy" for i in s],
            "trace_index": r,
            "source_id": s,
            "receiver_id": r,
            "grid_cell": np.arange(36),
            "source_cell": s,
            "receiver_cell": r,
            "sample_count": 9,
            "sample_interval_us": 1000,
            "geometry_usable": True,
            "eligible_after_fixed_qc": True,
            "header_eligible": True,
            "fixed_qc_excluded": False,
            "dc_review": False,
            "amplitude_review": False,
            "near_constant_review": False,
        }
    )
    for role, ids in (("source", s), ("receiver", r)):
        selected = stations[role].iloc[ids]
        for name in ("x_m", "y_m", "grid_x_m", "grid_y_m", "grid_line", "grid_point"):
            mapping[f"{role}_{name}"] = selected[name].to_numpy()
        xy = selected[["x_m", "y_m"]].to_numpy()
        projected = selected[["grid_x_m", "grid_y_m"]].to_numpy()
        mapping[f"{role}_distance_m"] = np.linalg.norm(xy - projected, axis=1)
        mapping[f"{role}_normalized_distance"] = np.linalg.norm(
            continuous_indices(xy, grids[role]) - selected[["grid_line", "grid_point"]].to_numpy(),
            axis=1,
        )
    mapping.loc[35, ["eligible_after_fixed_qc", "header_eligible"]] = False
    mapping.loc[2, "dc_review"] = True
    mapping.loc[3, "amplitude_review"] = True
    mapping.loc[4, "near_constant_review"] = True
    cells = pd.DataFrame(
        {"grid_cell": np.arange(36), "retained_count": mapping.eligible_after_fixed_qc.astype(int)}
    )
    directory = root / "candidate"
    directory.mkdir()
    paths = {}
    for name, table in {
        "trace_mapping": mapping,
        "grid_cells": cells,
        **{f"{k}_stations": v for k, v in stations.items()},
    }.items():
        path = directory / f"{name}.parquet"
        table.to_parquet(path, index=False)
        paths[str(path.relative_to(root))] = sha256_file(path)
    grids["trace_mapping"] = "candidate/trace_mapping.parquet"
    (directory / "candidate_grids.json").write_text(json.dumps({"tiny": grids}))
    paths["candidate/candidate_grids.json"] = sha256_file(directory / "candidate_grids.json")
    (directory / "metadata.json").write_text(json.dumps({"artifacts": paths}))
    raw = root / "raw"
    raw.mkdir()
    values = (np.sin(np.arange(9)[None, :] + np.arange(36)[:, None] * 0.1) + 0.2).astype(np.float32)
    lock = []
    for i in range(6):
        spec = segyio.spec()
        spec.samples, spec.format, spec.tracecount, spec.endian = np.arange(9), 5, 6, "little"
        path = raw / f"{i}.sgy"
        with segyio.create(str(path), spec) as handle:
            handle.bin[segyio.BinField.Interval] = 1000
            for j in range(6):
                handle.trace[j] = values[i * 6 + j]
        lock.append({"path": path.name, "sha256": sha256_file(path)})
    (root / "raw.lock.json").write_text(json.dumps(lock))
    inputs = {
        "dataset_root": "raw",
        "region_metadata": "candidate/metadata.json",
        "region_metadata_sha256": sha256_file(directory / "metadata.json"),
        "raw_input_lock": "raw.lock.json",
        "raw_input_lock_sha256": sha256_file(root / "raw.lock.json"),
    }
    for name, value in (("config", config), ("inputs", inputs)):
        (study / f"{name}.yaml").write_text(yaml.safe_dump(value))
    configs = {}
    for method in METHODS:
        cfg = yaml.safe_load((canonical / f"{method}.yaml").read_text())
        cfg["device"] = "cpu"
        if "training" in cfg:
            cfg["training"].update(max_steps=1, report_interval=1)
        if method == "pocs_grid":
            cfg["algorithm"].update(n_iterations=2, window_shape=[9, 2, 3, 2, 3], overlap=[0] * 5)
        elif method == "drr_grid":
            cfg["algorithm"].update(
                n_iterations=1, rank=1, spatial_window_shape=[2, 3, 2, 3], spatial_overlap=[0] * 4
            )
        elif method == "ccnet5d_grid":
            cfg["model"].update(hidden_channels=2, intermediate_channels=2, kernel_size=1)
            cfg["patches"]["shape"] = [9, 2, 3, 2, 3]
            cfg["prediction"]["core_shape"] = [9, 2, 3, 2, 3]
        elif method == "nersi_real":
            cfg["model"].update(
                time_sample_count=9,
                fourier_components=2,
                encoder_width=4,
                latent_channels=2,
                decoder_channels=[2, 2, 2],
                kernel_size=1,
            )
        else:
            cfg["model"].update(
                width=8, attention_width=4, relation_embedding_dim=2, node_fourier_components=2
            )
            cfg["graph"]["neighbors_per_relation"] = 1
            cfg["graph"]["relations"] = {
                name: {key: 1000.0 for key in scales}
                for name, scales in cfg["graph"]["relations"].items()
            }
            cfg["training"]["query_batch_size"] = 2
            cfg["prediction"]["query_batch_size"] = 8
        configs[method] = deepcopy(cfg)
        (study / f"{method}.yaml").write_text(yaml.safe_dump(cfg))
    return study, config, inputs, configs, mapping, stations, cells, grids, values
