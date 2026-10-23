import json
from pathlib import Path

import pandas as pd
import pytest

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.pipelines.visualize_forge_geometry import (
    visualize_forge_geometry,
    visualize_forge_survey,
)


@pytest.fixture
def prepared_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "seis_interp.pipelines.visualize_forge_geometry.execution_identity",
        lambda _: {"git_commit": "test"},
    )
    repository = Path(__file__).resolve().parents[2]
    for relative in (
        "scripts/forge_geometry_map.py",
        "scripts/forge_survey_map.py",
        "src/seis_interp/visualization/forge_geometry_map.py",
        "src/seis_interp/visualization/forge_survey_map.py",
        "src/seis_interp/pipelines/visualize_forge_geometry.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / relative).read_bytes())
    root = tmp_path / "preparation"
    (root / "input").mkdir(parents=True)
    (root / "masks").mkdir()
    manifest = pd.DataFrame(
        {
            "cell_id": [0, 1, 2, 3],
            "source_cell": [0, 0, 1, 1],
            "receiver_cell": [0, 1, 0, 1],
            "source_line": [1, 1, 1, 1],
            "source_point": [1, 1, 2, 2],
            "receiver_line": [5, 5, 5, 5],
            "receiver_point": [1, 2, 1, 2],
            "source_x_m": [335100, 335100, 335200, 335200],
            "source_y_m": [4261300, 4261300, 4261400, 4261400],
            "receiver_x_m": [335150, 335250, 335150, 335250],
            "receiver_y_m": [4261200, 4261200, 4261200, 4261200],
        }
    )
    manifest.to_parquet(root / "input/m1_trace_manifest.parquet", index=False)
    pd.DataFrame(
        {"cell_id": [3, 1, 2, 0], "split": ["test", "observed", "observed", "test"]}
    ).to_parquet(root / "masks/m1_random80_mvp_v1.parquet", index=False)
    write_json(root / "input/input_contract.json", {"missing_fraction": 0.5})
    seal = {
        "status": "prepared",
        "candidate_id": "tiny_analysis",
        "mask_id": "random50_test",
        "mask_seed": 1,
        "mask_hash": sha256_file(root / "masks/m1_random80_mvp_v1.parquet"),
        "artifacts": {
            str(path.relative_to(root)): sha256_file(path)
            for path in root.rglob("*")
            if path.is_file()
        },
    }
    write_json(root / "preparation_manifest.json", seal)
    return root


@pytest.fixture
def full_geometry(tmp_path, prepared_geometry):
    geometry_run = tmp_path / "full_geometry"
    geometry_run.mkdir()
    interim = tmp_path / "full_interim"
    interim.mkdir()
    geometry = pd.read_parquet(prepared_geometry / "input/m1_trace_manifest.parquet")
    extra = geometry.iloc[[-1]].copy()
    extra.loc[:, "source_point"] = 3
    extra.loc[:, "receiver_point"] = 3
    extra.loc[:, "source_x_m"] = 335300
    extra.loc[:, "source_y_m"] = 4261500
    extra.loc[:, "receiver_x_m"] = 335350
    extra.loc[:, "receiver_y_m"] = 4261200
    geometry_path = interim / "trace_geometry.parquet"
    pd.concat([geometry, extra], ignore_index=True).to_parquet(geometry_path, index=False)
    navigation_path = interim / "source_navigation.csv"
    pd.DataFrame(
        {
            "line": [1, 1, 1, 1],
            "point": [1, 2, 3, 4],
            "x_m": [335100, 335200, 335300, 335400],
            "y_m": [4261300, 4261400, 4261500, 4261600],
        }
    ).to_csv(navigation_path, index=False)
    write_json(
        geometry_run / "metadata.json",
        {
            "interim_directory": str(interim.relative_to(tmp_path)),
            "artifacts": {str(geometry_path.relative_to(tmp_path)): sha256_file(geometry_path)},
        },
    )
    write_json(
        geometry_run / "input.lock.json",
        {str(navigation_path.relative_to(tmp_path)): sha256_file(navigation_path)},
    )
    write_json(
        geometry_run / "summary.json",
        {
            "source_count": 3,
            "source_navigation_station_count": 4,
            "receiver_count": 4,
            "unknown_coordinate_receivers": [{"receiver_line": 5, "receiver_point": 4}],
        },
    )
    return geometry_run


def test_geometry_export_uses_only_metadata_and_records_sealed_mask(tmp_path, prepared_geometry):
    output = tmp_path / "output"
    visualize_forge_geometry(tmp_path, prepared_geometry, output, source_cell=0, dpi=30)
    record = json.loads((output / "manifest.json").read_text())
    assert record["waveforms_read"] == 0
    assert record["receiver_state_counts"] == {"test": 1, "observed": 1}
    assert not record["new_inference"] and not record["new_test_scoring"]
    assert pd.read_csv(output / "receiver_states.csv").split.tolist() == ["test", "observed"]
    for extension in ("png", "pdf", "svg"):
        assert (output / f"geometry_mask_source0.{extension}").is_file()
    for name, expected in record["artifacts"].items():
        assert sha256_file(output / name) == expected


def test_geometry_export_rejects_tampered_mask_before_output(tmp_path, prepared_geometry):
    with (prepared_geometry / "masks/m1_random80_mvp_v1.parquet").open("ab") as stream:
        stream.write(b"tampered")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="input hash mismatch"):
        visualize_forge_geometry(tmp_path, prepared_geometry, output, source_cell=0, dpi=30)
    assert not output.exists()


def test_survey_export_records_full_and_analysis_station_counts(
    tmp_path, prepared_geometry, full_geometry
):
    output = tmp_path / "survey_output"
    visualize_forge_survey(tmp_path, prepared_geometry, full_geometry, output, dpi=30)
    record = json.loads((output / "manifest.json").read_text())
    assert record["waveforms_read"] == 0
    assert record["survey_source_count"] == 4
    assert record["survey_recorded_source_count"] == 3
    assert record["survey_known_receiver_count"] == 3
    assert record["unknown_receiver_count"] == 1
    assert record["analysis_source_count"] == record["analysis_receiver_count"] == 2
    assert pd.read_csv(output / "survey_sources.csv").in_analysis.tolist() == [
        True,
        True,
        False,
        False,
    ]
    for extension in ("png", "pdf", "svg"):
        assert (output / f"survey_analysis_extent.{extension}").is_file()
    for name, expected in record["artifacts"].items():
        assert sha256_file(output / name) == expected


def test_survey_export_rejects_modified_navigation_before_output(
    tmp_path, prepared_geometry, full_geometry
):
    with (tmp_path / "full_interim/source_navigation.csv").open("a") as stream:
        stream.write("1,99,0,0\n")
    output = tmp_path / "survey_output"
    with pytest.raises(ValueError, match="input hash mismatch"):
        visualize_forge_survey(tmp_path, prepared_geometry, full_geometry, output, dpi=30)
    assert not output.exists()
