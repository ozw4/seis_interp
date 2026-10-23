"""Tiny synthetic C3 crop and five-method fair-grid study in the C3 input layout.

The crop has the C3 v3 axis semantics on a [2, 8, 2, 8] grid: 160 m source lines,
80 m shots staggered by 40 m on odd lines, and receivers 40 m apart relative to
their source. Its 256 cells map to a permuted subset of 300 interim rows of 32
samples at 8 ms, of which the crop uses the first 24 (``volume.json`` time 0:24).
The matched budget is two loss traces per update over four updates.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.c3_volume_index_store import write_c3_volume_index
from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.c3_volume_index import INDEX_CONTRACT, VOLUME_AXIS_ORDER
from seis_interp.processing.forge_mvp_contract import AXES

STUDY_ID = "tiny_c3_fair_grid"
SPATIAL_SHAPE = [2, 8, 2, 8]
TIME_COUNT = 24
INTERIM_ROWS = 300
INTERIM_SAMPLES = 32
STEPS = 4
TRACES = 2
DECAYS = [0.5, 0.8, 0.95]
BUDGET = {
    "max_steps": STEPS,
    "learning_rate": 0.001,
    "loss_traces_per_update": TRACES,
    "parameter_count_target": 1300,
    "parameter_count_tolerance": 0.95,
}
VOLUME = "data/processed/tiny_c3/volume"
INTERIM = "data/interim/tiny_c3"
TRAINING = {
    "learning_rate": 0.001,
    "max_steps": STEPS,
    "weight_decay": 0.0,
    "report_interval": 1,
    "learning_rate_schedule": "constant",
    "mixed_precision": "off",
}


def tiny_c3_volume_index() -> pd.DataFrame:
    """C-ordered volume index; ``array_row`` is a fixed permutation of interim rows."""
    lines, shots, rx, ry = (axis.ravel() for axis in np.indices(SPATIAL_SHAPE))
    rows = np.random.default_rng(3).permutation(INTERIM_ROWS)[: lines.size]
    return pd.DataFrame(
        {
            "array_row": rows.astype(np.int64),
            "ffid": (2401 + lines * SPATIAL_SHAPE[1] + shots).astype(np.int64),
            "source_line_index": lines.astype(np.int64),
            "shot_in_line_index": shots.astype(np.int64),
            "relative_receiver_x_index": rx.astype(np.int64),
            "relative_receiver_y_index": ry.astype(np.int64),
            "source_x_m": 7860.0 + 160.0 * lines,
            "source_y_m": 2500.0 + 80.0 * shots - 40.0 * (lines % 2),
            "relative_receiver_x_m": -20.0 + 40.0 * rx,
            "relative_receiver_y_m": -1000.0 + 40.0 * ry,
        }
    )


def tiny_c3_amplitudes(index: pd.DataFrame) -> np.ndarray:
    """Interim traces: smooth moveout events on crop rows, noise elsewhere; no zero trace."""
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 1.0, (INTERIM_ROWS, INTERIM_SAMPLES))
    samples = np.arange(INTERIM_SAMPLES)[None, :]
    offset = np.hypot(index.relative_receiver_x_m, index.relative_receiver_y_m).to_numpy()
    phase = 0.004 * offset + 0.3 * index.source_line_index.to_numpy()
    values[index.array_row.to_numpy()] = 3.0 * np.sin(0.5 * samples - phase[:, None]) + 0.5
    return values.astype(np.float32)


def write_tiny_c3_inputs(root: Path) -> dict:
    """Volume artifact and interim arrays under ``root``; returns the study ``inputs.yaml``."""
    index = tiny_c3_volume_index()
    size = len(index)
    write_c3_volume_index(
        root / VOLUME,
        index,
        {
            "volume_id": "tiny_c3_volume",
            "dataset_id": "seg_c3_na",
            "partition": "test",
            "config_source": None,
            "axis_order": list(VOLUME_AXIS_ORDER),
            "selection": {
                "time": [0, TIME_COUNT],
                **{
                    axis: [0, n]
                    for axis, n in zip(VOLUME_AXIS_ORDER[1:], SPATIAL_SHAPE, strict=True)
                },
            },
            "shape": [TIME_COUNT, *SPATIAL_SHAPE],
            "trace_count": size,
            "role_counts": {"observed": 52, "evaluation_target": size - 52},
            "index_contract": dict(INDEX_CONTRACT),
            "benchmark_case": {
                "case_id": "tiny_c3_case",
                "file": "benchmark_case.json",
                "sha256": "0" * 64,
            },
        },
    )
    interim = root / INTERIM
    interim.mkdir(parents=True)
    np.save(interim / "amplitudes.npy", tiny_c3_amplitudes(index), allow_pickle=False)
    np.save(interim / "time_s.npy", np.arange(INTERIM_SAMPLES) * 0.008, allow_pickle=False)
    paths = {
        "volume_index": f"{VOLUME}/volume_index.parquet",
        "volume_metadata": f"{VOLUME}/volume.json",
        "amplitudes": f"{INTERIM}/amplitudes.npy",
        "time_s": f"{INTERIM}/time_s.npy",
    }
    inputs = {}
    for name, path in paths.items():
        inputs[name] = path
        inputs[f"{name}_sha256"] = sha256_file(root / path)
    return inputs


def tiny_c3_contract() -> dict:
    """80 % test (205 of 256), 51 observed; the holdout keeps 41 observed and 10 V."""
    size = int(np.prod(SPATIAL_SHAPE))
    return {
        "study_id": STUDY_ID,
        "dataset_id": "seg_c3_na",
        "candidate_id": "tiny_c3",
        "axis_order": list(AXES),
        "spatial_shape": list(SPATIAL_SHAPE),
        "eligible_trace_count": size,
        "natural_missing_cell_count": 0,
        "time_sample_count": TIME_COUNT,
        "sample_interval_us": 8000,
        "mask_id": "tiny_c3_random80_v1",
        "mask_family": "random_trace",
        "missing_fraction": 0.8,
        "mask_seed": 11,
        "test_trace_count": round(0.8 * size),
        "observed_trace_count": size - round(0.8 * size),
        "model_seed": 7301,
        "mask_study_id": STUDY_ID,
        "preprocessing": {
            "dtype": "float32",
            "amplitude_scaling": "observed_global_rms",
            "agc": False,
            "trace_rms": False,
            "bandpass": None,
            "mute": None,
            "spatial_interpolation": False,
            "alignment": None,
        },
        "matched_training": dict(BUDGET),
        "validation_holdout": {
            "fraction": 0.2,
            "mask_id": "tiny_c3_validation_v1",
            "mask_seed": 12,
        },
    }


def tiny_c3_configs() -> dict:
    """The five method configs on the tiny crop, inside the matched budget."""
    common = {"loss": "masked_trace_relative_mse", "device": "cpu"}
    classical = {**common, "model_seed": None, "checkpoint_rule": "none"}
    neural = {**common, "model_seed": 7301, "checkpoint_rule": "fixed_final_ema"}
    configs = {
        "pocs_grid": {
            "method_id": "pocs_grid",
            "geometry_mode": "grid",
            **classical,
            "algorithm": {
                "n_iterations": 2,
                "threshold_start": 1.0,
                "threshold_end": 0.2,
                "window_shape": [TIME_COUNT, 1, 8, 1, 8],
                "overlap": [0, 0, 0, 0, 0],
            },
            "execution": {"backend": "batched"},
        },
        "drr_grid": {
            "method_id": "drr_grid",
            "geometry_mode": "grid",
            **classical,
            "algorithm": {
                "rank": 1,
                "damping_power": 3,
                "n_iterations": 1,
                "frequency_min_hz": 0.0,
                "frequency_max_hz": None,
                "spatial_window_shape": list(SPATIAL_SHAPE),
                "spatial_overlap": [0, 0, 0, 0],
            },
            "execution": {"backend": "batched", "workers": 1},
        },
        "ccnet5d_grid": {
            "method_id": "ccnet5d_grid",
            "geometry_mode": "grid",
            **neural,
            "model": {
                "hidden_channels": 2,
                "intermediate_channels": 2,
                "kernel_size": 3,
                "output_activation": "linear",
                "time_kernel_size": 3,
                "time_dilation": 1,
            },
            "training": {
                "optimizer": "adam",
                **TRAINING,
                "ema_decay": 0.8,
                "supervised_traces_per_update": TRACES,
            },
            "patches": {"shape": [TIME_COUNT, *SPATIAL_SHAPE], "inner_mask_fraction": 0.5},
            "prediction": {"core_shape": [TIME_COUNT, *SPATIAL_SHAPE]},
        },
        "regsi_real": {
            "method_id": "regsi_real",
            "geometry_mode": "real",
            **neural,
            "model": {
                "width": 8,
                "message_passing_rounds": 2,
                "time_downsample_factor": 2,
                "stem_kernel_size": 7,
                "temporal_kernel_size": 5,
                "temporal_dilations": [1, 2],
                "attention_width": 4,
                "relation_embedding_dim": 2,
                "relation_fusion": "learned_gate",
                "method_variant": "relational",
                "explicit_azimuth_features": True,
                "amplitude_mode": "train_global_rms",
                "max_edge_time_shift_samples": 0,
                "node_fourier_components": 2,
            },
            "training": {
                "optimizer": "adamw",
                **TRAINING,
                "query_batch_size": TRACES,
                # round(41 x 0.5) = 20 and round(51 x 0.5) = 26 hidden traces per episode.
                "inner_mask_fraction": 0.5,
                "gradient_clip_norm": 1.0,
                "cudnn_benchmark": False,
                "ema_decay": 0.8,
            },
            "graph": {
                "neighbors_per_relation": 1,
                "max_normalized_distance": 1.0,
                "candidate_chunk_size": 4096,
                "topology": "multi_relation",
                "neighbor_search": "exact_index",
                "relations": {
                    "source": {"source_scale_m": 160.0, "receiver_scale_m": 640.0},
                    "receiver": {"source_scale_m": 640.0, "receiver_scale_m": 160.0},
                    "cmp": {"midpoint_scale_m": 160.0, "offset_vector_scale_m": 640.0},
                    "offset_azimuth": {"midpoint_scale_m": 640.0, "offset_vector_scale_m": 160.0},
                },
            },
            "geometry_features": {
                "position_scale_m": 1000.0,
                "offset_scale_m": 1000.0,
                "azimuth_min_offset_m": 1.0,
            },
            "prediction": {"query_batch_size": 16},
        },
        "nersi_profile": {
            "method_id": "nersi_profile",
            "geometry_mode": "real",
            **neural,
            "model": {
                "input_features": 3,
                "fourier_components": 2,
                "frequency_base": [1.1, 1.1, 1.1],
                "encoder_width": 4,
                "latent_channels": 1,
                "decoder_channels": [2, 2, 2],
                "profile_shape": [TIME_COUNT, SPATIAL_SHAPE[3]],
                "upsample_scales": [2, 2, 2],
                "kernel_size": 1,
                "activation": "gelu",
                "output_activation": "linear",
            },
            "training": {
                "optimizer": "adam",
                **TRAINING,
                "supervised_traces_per_update": TRACES,
                "ema_decay": 0.8,
            },
            "prediction": {"profile_batch_size": 2},
        },
    }
    return deepcopy(configs)


def tiny_c3_study(root: Path) -> tuple[Path, dict, dict]:
    """Inputs, ``studies/<study>`` and the preparation script copy under ``root``."""
    repository = Path(__file__).resolve().parents[2]
    script = root / "scripts/c3_fair_grid_prepare.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes((repository / "scripts/c3_fair_grid_prepare.py").read_bytes())
    inputs = write_tiny_c3_inputs(root)
    study = root / "studies" / STUDY_ID
    study.mkdir(parents=True)
    contract, configs = tiny_c3_contract(), tiny_c3_configs()
    for name, value in (("config", contract), ("inputs", inputs), *configs.items()):
        (study / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return study, contract, configs
