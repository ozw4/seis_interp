from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from seis_interp.cli import main
from seis_interp.training.trace_graph_checkpoints import load_trace_graph_checkpoint
from tests.fixtures.whole_shot_survey import prepare_whole_shot_survey


def _build_trace_graph_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    interim, processed, trace_filter = prepare_whole_shot_survey(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": 5},
                "sampling": {
                    "random_ffid_holdout_fraction": 0.5,
                    "validation_fraction_of_holdout": 0.5,
                    "split_scope": "whole_ffid",
                    "trace_amplitude_filter": trace_filter.to_dict(),
                    "duplicate_physical_coordinate_policy": "keep_lowest_array_row",
                },
                "normalization": {
                    "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
                    "amplitude": "train_global_rms",
                },
                "model": {
                    "name": "trace_graph_interpolator",
                    "hidden_width": 8,
                    "graph_mode": "trace_lattice",
                    "attention_time_resolution": "pooled",
                    "use_gradient_checkpointing": False,
                    "refinement_passes": 1,
                    "message_passing_rounds": 1,
                    "time_downsample_factor": 5,
                    "stem_kernel_size": 3,
                    "temporal_kernel_size": 3,
                    "temporal_dilations": [1],
                    "spatial_kernel_size": 3,
                    "attention_width": 8,
                    "distance_epsilon": 1.0e-6,
                    "target_coordinates": ["source_x_m", "source_y_m"],
                    "target_coordinate_scaling": "train_minmax",
                    "neighborhood": {
                        "type": "nearest_train_source_gathers",
                        "distance": "euclidean_source_xy_m",
                        "source_gather_count": 2,
                    },
                },
                "training": {
                    "amplitude_scaling": "per_trace_rms",
                    "loss": "masked_l2_spectrum_slope_amplitude",
                    "optimizer": "adamw",
                    "learning_rate": 1.0e-3,
                    "weight_decay": 1.0e-5,
                    "learning_rate_schedule": "cosine",
                    "minimum_learning_rate": 3.0e-5,
                    "total_steps": 1,
                    "batch_size": 1,
                    "target_sampling": "epoch_without_replacement",
                    "exclude_target_ffid_neighbors": True,
                    "neighbor_dropout": 0.0,
                    "spectrum_weight": 0.0,
                    "slope_weight": 0.0,
                    "amplitude_weight": 0.0,
                    "gradient_clip_norm": 1.0,
                    "evaluation_interval_steps": 1,
                    "validation_batch_size": 1,
                    "training_audit_count": 4,
                    "mixed_precision": "bfloat16",
                    "device": "cuda:0",
                    "ffid_range": [10, 17],
                },
                "evaluation": {
                    "primary_metric": "oracle_per_trace_unit_rms_global_snr_db",
                    "success_threshold_db": 25.0,
                    "comparison": "strictly_greater_than",
                    "required_eligible_ffid_count": 4780,
                    "required_sample_count": 625,
                    "required_effective_split_counts": {
                        "train": 578685,
                        "validation": 437087,
                        "test": 1287693,
                    },
                    "required_ffid_split_counts": {
                        "train": 1195,
                        "validation": 896,
                        "test": 2689,
                    },
                    "required_fully_excluded_ffids": [],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, interim, processed


def test_cli_runs_leakage_safe_trace_graph_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = _build_trace_graph_fixture(tmp_path)
    output = tmp_path / "run"

    exit_code = main(
        [
            "train",
            "trace-graph",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--json",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    metrics = json.loads(captured.out)
    assert metrics == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "trace_graph_interpolator 0/1" in captured.err
    assert "trace_graph_interpolator 1/1" in captured.err

    assert math.isfinite(metrics["oracle_per_trace_unit_rms_global_snr_db"])
    assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert metrics["training_audit_trace_count"] == 4
    assert metrics["duplicate_physical_coordinates"]["remaining_duplicate_physical_cell_count"] == 0

    scope_checks = metrics["formal_success_scope"]["checks"]
    assert scope_checks["target_ffid_neighbor_entries_zero"]
    assert scope_checks["neighbor_sources_train_only"]
    assert scope_checks["train_source_coordinate_collisions_zero"]
    assert scope_checks["train_validation_source_coordinate_overlap_zero"]
    assert scope_checks["checkpoint_raw_metric_reproduced"]
    assert scope_checks["validation_metric_domain_matches"]

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["preparation"]["split_scope"] == "whole_ffid"
    assert inputs_lock["selection"]["ffid_split_overlap_count"] == 0
    assert inputs_lock["selection"]["maximum_splits_per_ffid"] == 1
    assert inputs_lock["receiver_grid"]["shape"] == [8, 68]
    assert (
        inputs_lock["duplicate_physical_coordinates"] == (metrics["duplicate_physical_coordinates"])
    )
    assert inputs_lock["amplitude_access"]["value_rows_materialized_by_split"] == {
        "excluded": False,
        "test": False,
        "train": True,
        "validation": True,
    }
    assert inputs_lock["model"]["graph_mode"] == "trace_lattice"
    assert inputs_lock["training"]["target_sampling"] == "epoch_without_replacement"
    assert inputs_lock["training"]["target_sampling_seed"] == 5 + 3
    assert inputs_lock["training"]["neighbor_dropout_seed"] == 5 + 1
    assert inputs_lock["training"]["target_sampling_rng_independent_of_neighbor_dropout"]

    resolved = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["training"]["device"] == "cpu"

    checkpoint = load_trace_graph_checkpoint(output / "artifacts/best.pt")
    assert checkpoint.best_step == metrics["best_step"]
    assert checkpoint.graph_mode == "trace_lattice"
    assert checkpoint.graph_mode == inputs_lock["checkpoint"]["graph_mode"]
    assert (
        checkpoint.best_validation_global_snr_db
        == metrics["oracle_per_trace_unit_rms_global_snr_db"]
    )
