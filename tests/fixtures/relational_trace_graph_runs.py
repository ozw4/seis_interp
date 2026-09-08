"""Executable toy configs and disjoint native/dense trace graph benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from tests.fixtures.ccnet5d_artifacts import prepare_ccnet5d_artifacts


@dataclass(frozen=True)
class TraceGraphRunArtifacts:
    interim: Path
    processed: Path
    masks: dict[str, Path]
    cases: dict[str, Path]
    volumes: dict[str, Path]


def prepare_trace_graph_run_artifacts(
    directory: Path, *, dense: bool = False
) -> TraceGraphRunArtifacts:
    source = prepare_ccnet5d_artifacts(directory, shuffle_tables=True)
    masks, cases, volumes = {}, {}, {}
    for partition, lines, seed in (
        ("train", [0, 2], 17),
        ("validation", [2, 3], 19),
        ("test", [3, 4], 23),
    ):
        masks[partition] = source.processed / "masks" / partition
        cases[partition] = source.processed / "cases" / partition
        prepare_interpolation_mask(
            source.interim,
            source.processed,
            masks[partition],
            partition=partition,
            kind="random_trace",
            missing_fraction=0.5,
            random_seed=seed,
            config_source="studies/synthetic/config.yaml",
        )
        prepare_benchmark_case(
            source.interim,
            source.processed,
            masks[partition],
            cases[partition],
            case_id=partition,
        )
        if dense:
            volumes[partition] = source.processed / "volumes" / partition
            prepare_c3_volume_index(
                source.interim,
                source.processed,
                masks[partition],
                cases[partition],
                volumes[partition],
                volume_id=partition,
                time_range=(1, 4),
                source_line_range=tuple(lines),
                shot_in_line_range=(0, 2),
                relative_receiver_x_range=(0, 2),
                relative_receiver_y_range=(0, 2),
            )
    return TraceGraphRunArtifacts(source.interim, source.processed, masks, cases, volumes)


def trace_graph_training_config() -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "synthetic_c3_ccnet5d"},
        "model": {
            "name": "relational_trace_graph",
            "width": 8,
            "message_passing_rounds": 2,
            "time_downsample_factor": 2,
            "stem_kernel_size": 3,
            "temporal_kernel_size": 3,
            "temporal_dilations": [1, 2],
            "attention_width": 4,
            "relation_embedding_dim": 3,
            "relation_fusion": "learned_gate",
        },
        "graph": {
            "neighbors_per_relation": 2,
            "max_normalized_distance": 1.0,
            "candidate_chunk_size": 16,
            "relations": {
                "source": {"source_scale_m": 2000.0, "receiver_scale_m": 5000.0},
                "receiver": {"source_scale_m": 5000.0, "receiver_scale_m": 2000.0},
                "cmp": {"midpoint_scale_m": 2000.0, "offset_vector_scale_m": 5000.0},
                "offset_azimuth": {"midpoint_scale_m": 5000.0, "offset_vector_scale_m": 2000.0},
            },
        },
        "geometry_features": {
            "position_scale_m": 1000.0,
            "offset_scale_m": 1000.0,
            "azimuth_min_offset_m": 0.1,
        },
        "training_data": {"pool": "all_train_traces", "time_samples": [1, 4]},
        "training_mask": {
            "kinds": ["random_trace", "random_whole_ffid"],
            "kind_probabilities": [0.5, 0.5],
            "missing_fractions": [0.5],
        },
        "training": {
            "device": "cpu",
            "random_seed": 7,
            "loss": "masked_mse",
            "max_steps": 2,
            "query_batch_size": 4,
            "validation_interval": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
            "query_batch_size": 4,
        },
    }


def trace_graph_prediction_config() -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "synthetic_c3_ccnet5d"},
        "benchmark_case": {"id": "test"},
        "prediction": {"device": "cpu", "query_batch_size": 3},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }


def write_trace_graph_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
