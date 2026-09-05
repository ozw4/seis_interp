from __future__ import annotations

from pathlib import Path

from seis_interp.data.c3_volume_index_inputs import load_bound_benchmark_case
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from tests.fixtures.c3_volume_artifacts import prepare_c3_volume_artifacts


def test_prepares_dense_case_bound_volume_without_amplitude_copy(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_artifacts(tmp_path)
    output = artifacts.processed_dir / "volumes" / "synthetic-volume"

    metadata = prepare_c3_volume_index(
        artifacts.interim_dir,
        artifacts.processed_dir,
        artifacts.mask_dir,
        artifacts.case_dir,
        output,
        volume_id="synthetic_volume",
        time_range=(0, 4),
        source_line_range=artifacts.source_line_range,
        shot_in_line_range=artifacts.shot_in_line_range,
        relative_receiver_x_range=(0, 8),
        relative_receiver_y_range=(0, 68),
        config_source="studies/synthetic/config.yaml",
    )
    index, loaded = load_c3_volume_index(output)
    case = load_bound_benchmark_case(loaded, case_dir=artifacts.case_dir)

    assert metadata == loaded
    assert loaded["shape"] == [4, 1, 1, 8, 68]
    assert loaded["trace_count"] == len(index) == 544
    assert sum(loaded["role_counts"].values()) == 544
    assert case["case_id"] == "synthetic_case"
    assert set(path.name for path in output.iterdir()) == {
        "volume_index.parquet",
        "volume.json",
    }
