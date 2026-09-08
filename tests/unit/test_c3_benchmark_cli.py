from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.cli import main
from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from tests.fixtures.c3_benchmark import synthetic_benchmark_cases

STUDY = REPOSITORY_ROOT / "studies/study_027_c3_na_benchmark"


def test_default_dry_run_is_strict_json_and_writes_nothing(tmp_path):
    source = tmp_path / "interim"
    source.mkdir()
    for name in ("traces.parquet", "time_s.npy", "amplitudes.npy", "dataset.json"):
        (source / name).write_bytes(b"dry-run must not read these contents")
    output = tmp_path / "output"
    inputs = {
        "source": {"interim": "interim"},
        "outputs": {"root": "output"},
        "cases": synthetic_benchmark_cases(),
    }
    path = tmp_path / "inputs.yaml"
    path.write_text(yaml.safe_dump(inputs))
    before = set(tmp_path.rglob("*"))
    from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark

    result = prepare_c3_benchmark(STUDY / "config.yaml", path)
    assert result["ready"] and result["status"] == "dry_run"
    assert result["fixed_shape"] == [384, 16, 32, 8, 32]
    assert set(tmp_path.rglob("*")) == before
    assert not output.exists()
    json.dumps(result, allow_nan=False)


def test_dry_run_missing_input_and_collision_exit_codes(tmp_path, capsys):
    inputs = {
        "source": {"interim": "missing"},
        "outputs": {"root": "output"},
        "cases": synthetic_benchmark_cases(),
    }
    path = tmp_path / "inputs.yaml"
    path.write_text(yaml.safe_dump(inputs))
    arguments = [
        "data",
        "prepare-c3-benchmark",
        "--config",
        str(STUDY / "config.yaml"),
        "--inputs",
        str(path),
        "--json",
    ]
    assert main(arguments) == 1
    result = json.loads(capsys.readouterr().out)
    assert len(result["missing_inputs"]) == 4
    assert not (tmp_path / "output").exists()
    assert main([*arguments, "--execute"]) == 1
    assert "missing" in capsys.readouterr().err
    (tmp_path / "output").mkdir()
    assert main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["output_exists"]


@pytest.mark.parametrize("command", ["qc-c3-geometry", "qc-c3-crop"])
def test_qc_cli_rejects_fixed_time_violation_before_io(tmp_path, capsys, command):
    config = load_resolved_config(STUDY / "config.yaml")
    config["benchmark_volume"]["selection"]["time"] = [1, 385]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert (
        main(
            [
                "data",
                command,
                "--config",
                str(path),
                "--input",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "output"),
                "--json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "selection.time" in captured.err
    assert not (tmp_path / "output").exists()


def test_geometry_cli_success_without_amplitude_file_and_overwrite_refusal(tmp_path, capsys):
    line = np.repeat(np.arange(42), 544)
    x = np.tile(np.repeat(np.arange(-140, 141, 40), 68), 42)
    y = np.tile(np.tile(np.arange(-2680, 1, 40), 8), 42)
    table = pd.DataFrame(
        {
            "array_row": np.arange(len(line)),
            "ffid": line + 1,
            "source_x_m": line * 160.0,
            "source_y_m": (line % 2) * 40.0,
            "receiver_x_m": line * 160.0 + x,
            "receiver_y_m": (line % 2) * 40.0 + y,
        }
    )
    interim = tmp_path / "interim"
    interim.mkdir()
    table.to_parquet(interim / "traces.parquet", index=False)
    np.save(interim / "time_s.npy", 0.5 + np.arange(384) * 0.008)
    args = [
        "data",
        "qc-c3-geometry",
        "--config",
        str(STUDY / "config.yaml"),
        "--input",
        str(interim),
        "--output",
        str(tmp_path / "qc"),
        "--json",
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["time"]["time_start_s"] == 0.5
    assert result["sail_lines"]["index_range"] == [25, 41]
    assert main(args) == 1
    assert "already exists" in capsys.readouterr().err
