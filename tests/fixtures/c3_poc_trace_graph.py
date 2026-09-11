"""Tiny verified random-80 volume inputs and fixed GNN PoC configuration."""

from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from tests.fixtures.c3_volume_run_artifacts import prepare_c3_volume_run_artifacts


def prepare_poc_trace_graph_inputs(path, *, target_offset=0.0):
    artifacts = prepare_c3_volume_run_artifacts(
        path,
        dataset_id="seg_c3_na",
        missing_fraction=0.8,
        target_offset=target_offset,
        time_sample_count=8,
    )
    selection = artifacts.volume_metadata["selection"]
    start, stop = selection["source_line"]
    dimensions = C3BenchmarkDimensions(
        time_range=tuple(selection["time"]),
        sail_line_numbers=(start, stop - 1),
        shape=tuple(artifacts.volume_metadata["shape"]),
    )
    paths = {
        "interim_dir": artifacts.interim,
        "processed_dir": artifacts.processed,
        "mask_dir": artifacts.mask,
        "case_dir": artifacts.case,
        "volume_dir": artifacts.volume,
        "dimensions": dimensions,
    }
    config = poc_trace_graph_config(selection)
    return load_c3_random80_poc_inputs(config=config, **paths), config, paths


def poc_trace_graph_config(selection):
    return {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "seg_c3_na"},
        "benchmark_case": {},
        "benchmark_volume": {"selection": selection},
        "interpolation_mask": {"kind": "random_trace", "missing_fraction": 0.8},
        "model": {
            "name": "relational_trace_graph",
            "width": 8,
            "message_passing_rounds": 1,
            "time_downsample_factor": 2,
            "stem_kernel_size": 3,
            "temporal_kernel_size": 3,
            "temporal_dilations": [1],
            "attention_width": 4,
            "relation_embedding_dim": 2,
            "relation_fusion": "learned_gate",
        },
        "graph": {
            "neighbors_per_relation": 2,
            "max_normalized_distance": 1.0,
            "candidate_chunk_size": 16,
            "neighbor_search": "exact_index",
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
        "training": {
            "device": "cpu",
            "random_seed": 7,
            "loss": "masked_trace_relative_mse",
            "amplitude_scaling": "observed_volume_global_rms",
            "optimizer": "adamw",
            "max_steps": 3,
            "query_batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "inner_mask_fraction": 0.5,
            "report_interval": 1,
        },
        "prediction": {"query_batch_size": 5},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }
