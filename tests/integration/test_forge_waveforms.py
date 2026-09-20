import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import segyio
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_waveforms import read_forge_gather
from seis_interp.pipelines.audit_forge_waveforms import run_forge_waveform_audit


def test_little_endian_waveforms_preserve_order_and_nonfinite_for_qc(tmp_path):
    path = tmp_path / "wave.sgy"
    spec = segyio.spec()
    spec.endian = "little"
    spec.format = 5
    spec.samples = np.arange(8)
    spec.tracecount = 3
    expected = np.arange(24, dtype=np.float32).reshape(3, 8)
    expected[2, 1] = np.nan
    with segyio.create(str(path), spec) as handle:
        handle.bin[segyio.BinField.Interval] = 1000
        handle.bin[segyio.BinField.Samples] = 8
        for i in range(3):
            handle.header[i] = {115: 8, 117: 1000}
            handle.trace[i] = expected[i]
    headers = pd.DataFrame(
        {"trace_index": [0, 1, 2], "sample_count": 8, "sample_interval_us": 1000}
    )
    values, dt = read_forge_gather(path, headers)
    np.testing.assert_array_equal(values, expected)
    assert dt == 0.001
    with pytest.raises(ValueError, match="file order"):
        read_forge_gather(path, headers.iloc[::-1])
    headers["sample_interval_us"] = 2000
    with pytest.raises(ValueError, match="time metadata"):
        read_forge_gather(path, headers)


def test_pipeline_covers_all_traces_and_binds_source_hashes(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    path = root / "tiny.sgy"
    spec = segyio.spec()
    spec.endian = "little"
    spec.format = 5
    spec.samples = np.arange(4001)
    spec.tracecount = 3
    t = np.arange(4001) * 0.001
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = 1000
        f.bin[segyio.BinField.Samples] = 4001
        for i, x in enumerate(
            [np.sin(2 * np.pi * 30 * t), np.zeros(4001), np.sin(2 * np.pi * 60 * t)]
        ):
            f.header[i] = {115: 4001, 117: 1000}
            f.trace[i] = x.astype(np.float32)
    headers = pd.DataFrame(
        {
            "source_file": ["tiny.sgy"] * 3,
            "trace_index": [0, 1, 2],
            "ffid": 1,
            "source_line": 507,
            "source_point": 101,
            "receiver_line": [101, 105, 109],
            "receiver_point": 517,
            "source_x_m": 0.0,
            "source_y_m": 0.0,
            "receiver_x_m": [100.0, 200.0, 300.0],
            "receiver_y_m": 0.0,
            "trace_identification_code": [1, 1, 6],
            "header_eligible": [True, True, False],
            "sample_count": 4001,
            "sample_interval_us": 1000,
        }
    )
    header_dir = tmp_path / "headers"
    header_dir.mkdir()
    header_path = header_dir / "trace_headers.parquet"
    headers.to_parquet(header_path, index=False)
    header_run = tmp_path / "header_run"
    header_run.mkdir()
    (header_run / "metadata.json").write_text(
        json.dumps(
            {
                "interim_directory": "headers",
                "artifacts": {"headers/trace_headers.parquet": sha256_file(header_path)},
            }
        )
    )
    (header_run / "input.lock.json").write_text(
        json.dumps(
            [
                {
                    "path": "tiny.sgy",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                },
            ]
        )
    )
    study = tmp_path / "studies/test_wave"
    study.mkdir(parents=True)
    config_path = (
        Path(__file__).resolve().parents[2] / "studies/study_048_forge_waveform_qc/config.yaml"
    )
    (study / "config.yaml").write_text(config_path.read_text())
    (study / "inputs.yaml").write_text(
        yaml.safe_dump({"dataset_root": "input", "header_run": "header_run"})
    )
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "0" * 40)
    run = run_forge_waveform_audit(tmp_path, study)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"]
    assert summary["trace_count"] == 3
    assert summary["candidate_count"] == 2
    assert summary["candidate_status_counts"] == {"passed_numeric_checks": 1, "all_zero": 1}
    assert summary["spectrum_trace_count"] == 1
    assert (run / "figures/overview.png").is_file()
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source file changed"):
        run_forge_waveform_audit(tmp_path, study)
