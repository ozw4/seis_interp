import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import segyio
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.pipelines.review_forge_qc import run_forge_qc_review
from seis_interp.processing.forge_waveform_qc import flag_relative_amplitude, waveform_metrics


def test_fixed_mask_and_review_are_bound_to_audit_and_raw_waveforms(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "input"
    root.mkdir()
    path = root / "tiny.sgy"
    times = np.arange(4001) * 0.001
    x = np.tile(np.sin(2 * np.pi * 20 * times), (8, 1)).astype(np.float32)
    x[0] = 0
    x[1] = 3
    x[2] = 100 + 0.0001 * x[2]
    spec = segyio.spec()
    spec.endian = "little"
    spec.format = 5
    spec.samples = np.arange(4001)
    spec.tracecount = len(x)
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = 1000
        f.bin[segyio.BinField.Samples] = 4001
        for i, row in enumerate(x):
            f.header[i] = {115: 4001, 117: 1000}
            f.trace[i] = row
    headers = pd.DataFrame(
        {
            "source_file": "tiny.sgy",
            "trace_index": range(8),
            "ffid": 1,
            "source_line": 501,
            "source_point": 101,
            "receiver_line": 101,
            "receiver_point": np.arange(577, 585),
            "source_x_m": 0.0,
            "source_y_m": 0.0,
            "receiver_x_m": np.arange(8) * 50.0,
            "receiver_y_m": 10.0,
            "offset_m": 200.0,
            "header_eligible": [True] * 7 + [False],
            "trace_identification_code": [1] * 7 + [6],
            "sample_count": 4001,
            "sample_interval_us": 1000,
        }
    )
    cfg = yaml.safe_load((repo / "studies/study_048_forge_waveform_qc/config.yaml").read_text())
    cfg["minimum_offset_bin_traces"] = 2
    metrics, _ = waveform_metrics(x, 0.001, cfg)
    table = flag_relative_amplitude(pd.concat([headers, metrics], axis=1), cfg)
    wave = tmp_path / "wave"
    wave.mkdir()
    wave_path = wave / "waveform_qc.parquet"
    table.to_parquet(wave_path, index=False)
    (wave / "metadata.json").write_text(
        json.dumps({"interim_directory": "wave", "waveform_table_sha256": sha256_file(wave_path)})
    )
    (wave / "inputs.resolved.yaml").write_text("dataset_root: input\n")
    (wave / "input.lock.json").write_text(
        json.dumps({"segy_files": [{"path": "tiny.sgy", "sha256": sha256_file(path)}]})
    )
    study = tmp_path / "studies/review"
    study.mkdir(parents=True)
    review_cfg = yaml.safe_load(
        (repo / "studies/study_050_forge_qc_review/config.yaml").read_text()
    )
    review_cfg["minimum_reference_traces"] = 2
    (study / "config.yaml").write_text(yaml.safe_dump(review_cfg))
    (study / "inputs.yaml").write_text(
        yaml.safe_dump(
            {
                "waveform_run": "wave",
                "expected_fixed_exclusions": {"all_zero": 1, "nonzero_constant": 1},
            }
        )
    )
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "0" * 40)
    run = run_forge_qc_review(tmp_path, study)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["retained_count"] == 5
    assert summary["near_constant_review_count"] == 1
    meta = json.loads((run / "metadata.json").read_text())
    mask = pd.read_parquet(tmp_path / meta["processed_directory"] / "fixed_qc_mask.parquet")
    assert mask.fixed_qc_excluded.tolist() == [True, True, False, False, False, False, False, False]
    assert mask.loc[2, "review_pending"] and mask.loc[2, "eligible_after_fixed_qc"]
    assert len(list((run / "figures/examples").glob("*.png"))) == summary["example_count"]
    assert all(sha256_file(tmp_path / p) == h for p, h in meta["artifacts"].items())
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="SEG-Y input changed"):
        run_forge_qc_review(tmp_path, study)
    with wave_path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="waveform artifact hash mismatch"):
        run_forge_qc_review(tmp_path, study)
