import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.pipelines.audit_forge_geometry import run_forge_geometry_audit


def test_geometry_pipeline_binds_lineage_and_preserves_qc_domains(tmp_path, monkeypatch):
    rows = []
    for si, (sl, sp) in enumerate([(507, 101), (507, 102), (508, 101), (508, 102)]):
        for ri, (rl, rp) in enumerate([(101, 501), (101, 502), (105, 501), (105, 502), (109, 503)]):
            dead = ri == 4 or (si == 0 and ri == 0)
            zero = si == 1 and ri == 1
            review = si == 2 and ri == 2
            rows.append(
                {
                    "source_file": f"{si}.sgy",
                    "trace_index": ri,
                    "ffid": si + 1,
                    "source_line": sl,
                    "source_point": sp,
                    "receiver_line": rl,
                    "receiver_point": rp,
                    "source_x_m": (sp - 101) * 50.0,
                    "source_y_m": (sl - 507) * 300.0,
                    "receiver_x_m": 0.0 if dead else (rl - 101) * 50.0,
                    "receiver_y_m": 0.0 if dead else (rp - 501) * 50.0,
                    "trace_identification_code": 2 if dead else 1,
                    "header_eligible": not dead,
                    "numerically_usable": not dead and not zero,
                    "waveform_status": "header_excluded"
                    if dead
                    else "all_zero"
                    if zero
                    else "review"
                    if review
                    else "passed_numeric_checks",
                    "sample_count": 4001,
                    "sample_interval_us": 1000,
                }
            )
    wave = tmp_path / "waves"
    header = tmp_path / "headers"
    wave.mkdir()
    header.mkdir()
    table = pd.DataFrame(rows)
    table.to_parquet(wave / "waveform_qc.parquet", index=False)
    table.to_parquet(header / "trace_headers.parquet", index=False)
    pd.DataFrame(
        {
            "line": [507, 507, 508, 508],
            "point": [101, 102, 101, 102],
            "x_m": [0, 50, 0, 50],
            "y_m": [0, 0, 300, 300],
        }
    ).to_csv(header / "source_navigation.csv", index=False)
    (header / "metadata.json").write_text(
        json.dumps(
            {
                "interim_directory": "headers",
                "artifacts": {
                    p.relative_to(tmp_path).as_posix(): sha256_file(p) for p in header.iterdir()
                },
            }
        )
    )
    (wave / "metadata.json").write_text(
        json.dumps(
            {
                "interim_directory": "waves",
                "waveform_table_sha256": sha256_file(wave / "waveform_qc.parquet"),
            }
        )
    )
    (wave / "inputs.resolved.yaml").write_text("header_run: headers\n")
    (wave / "input.lock.json").write_text(
        json.dumps({"header_table_sha256": sha256_file(header / "trace_headers.parquet")})
    )
    study = tmp_path / "studies/geometry"
    study.mkdir(parents=True)
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / "studies/study_049_forge_geometry_eda/config.yaml"
        ).read_text()
    )
    config["grids"] = [config["grids"][1], config["grids"][3]]
    config["origin_xy_m"] = [0, 0]
    (study / "config.yaml").write_text(yaml.safe_dump(config))
    (study / "inputs.yaml").write_text("waveform_run: waves\n")
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "0" * 40)
    run = run_forge_geometry_audit(tmp_path, study)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["numeric_usable_pairs"] == 14
    assert summary["recorded_station_product"] == 20
    assert summary["known_coordinate_reference_pairs"] == 16
    assert summary["known_reference_missing_pairs"] == 2
    assert summary["availability_counts"] == {
        "header_excluded": 5,
        "waveform_unusable": 1,
        "numeric_usable_review": 1,
        "passed_numeric_checks": 13,
    }
    assert summary["unknown_coordinate_receivers"] == [
        {"receiver_line": 109, "receiver_point": 503}
    ]
    grids = pd.read_csv(run / "grid_occupancy.csv")
    assert len(grids) == 8
    assert grids.observed_traces.eq(14).all()
    assert len(list((run / "figures").glob("*.png"))) == 6
    meta = json.loads((run / "metadata.json").read_text())
    assert all(sha256_file(tmp_path / p) == h for p, h in meta["artifacts"].items())
    (wave / "input.lock.json").write_text('{"header_table_sha256": "bad"}')
    with pytest.raises(ValueError, match="lineage mismatch"):
        run_forge_geometry_audit(tmp_path, study)
    with (wave / "waveform_qc.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="waveform artifact hash mismatch"):
        run_forge_geometry_audit(tmp_path, study)
