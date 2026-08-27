from __future__ import annotations

import numpy as np
import pytest

from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.training.point_sampler import build_trace_points


def _training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    spatial = np.arange(20, dtype=np.float64).reshape(4, 5) / 20.0
    amplitudes = np.asarray(
        [[row + sample / 10.0 for sample in range(5)] for row in range(1, 5)],
        dtype=np.float32,
    )
    rows = np.arange(4, dtype=np.int64)
    return time, spatial, amplitudes, rows


def _model_config() -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "model": {
            "input_features": 6,
            "hidden_width": 8,
            "hidden_layers": 1,
            "omega_0": 10.0,
            "hidden_omega": 1.0,
        },
        "training": {"learning_rate": 1.0e-3},
    }


def test_training_condition_prints_start_and_one_line_per_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    time, spatial, amplitudes, rows = _training_arrays()
    training_coordinates, training_targets = build_trace_points(time, spatial, amplitudes, rows)

    pipeline.run_training_fit_condition(
        config=_model_config(),
        label="progress_probe",
        batch_mode="random_replacement",
        full_batch=False,
        replacement=True,
        total_updates=2,
        report_interval=1,
        batch_size=4,
        normalized_time=time,
        normalized_spatial_by_array_row=spatial,
        normalized_amplitudes=amplitudes,
        selected_array_rows=rows,
        all_coordinate_tensor=None,
        all_target_tensor=None,
        training_coordinates=training_coordinates,
        training_targets=training_targets,
        sample_count=5,
        prediction_batch_size=7,
        device="cpu",
        random_seed=42,
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "[progress_probe] start: total_updates=2 batch_mode=random_replacement "
        "batch_size=4 traces=4 device=cpu"
    )
    progress_lines = lines[1:]
    assert len(progress_lines) == 2
    for step, line in enumerate(progress_lines, start=1):
        assert line.startswith(f"[progress_probe] step {step}/2 loss=")
        for field in ("median_snr_db=", "median_corr=", "rms_ratio="):
            assert field in line
