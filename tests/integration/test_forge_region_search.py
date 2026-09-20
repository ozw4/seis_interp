import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.pipelines.search_forge_regions import run_forge_region_search


def test_region_search_preserves_fixed_mask_and_hash_bound_mapping(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    study = tmp_path / "studies/search"
    study.mkdir(parents=True)
    config = yaml.safe_load(
        (repo / "studies/study_051_forge_region_search/config.yaml").read_text()
    )
    config.update(
        origin_xy_m=[0.0, 0.0],
        sizes={"tiny": {"source": [2, 4], "receiver": [2, 4]}},
        refinement_seeds=2,
        station_shortlist=3,
        finalists_per_size=2,
    )
    (study / "config.yaml").write_text(yaml.safe_dump(config))
    inputs = {
        "waveform_run": "wave",
        "qc_run": "qc",
        "geometry_run": "geo",
        "expected_fixed_exclusions": {"all_zero": 1, "nonzero_constant": 1},
    }
    (study / "inputs.yaml").write_text(yaml.safe_dump(inputs))
    sid, rid = np.indices((16, 16))
    s, r = sid.ravel(), rid.ravel()
    wave = pd.DataFrame(
        {
            "source_file": [f"{x:02}.sgy" for x in s],
            "trace_index": r,
            "ffid": s,
            "source_line": 501 + s // 8,
            "source_point": 101 + s % 8,
            "receiver_line": 101 + (r // 8) * 4,
            "receiver_point": 501 + r % 8,
            "source_x_m": (s % 8) * 50.0,
            "source_y_m": (s // 8) * 200.0,
            "receiver_x_m": (r % 8) * 50.0,
            "receiver_y_m": (r // 8) * 200.0,
            "trace_identification_code": 1,
            "header_eligible": True,
            "numerically_usable": True,
            "waveform_status": "passed_numeric_checks",
            "sample_count": 4001,
            "sample_interval_us": 1000,
        }
    )
    wave.loc[0, ["numerically_usable", "waveform_status"]] = [False, "all_zero"]
    wave.loc[1, ["numerically_usable", "waveform_status"]] = [False, "constant"]
    wave.loc[2, "waveform_status"] = "review"
    mask = wave[["source_file", "trace_index", "header_eligible"]].copy()
    mask["fixed_qc_excluded"] = ~wave.numerically_usable
    mask["eligible_after_fixed_qc"] = wave.numerically_usable
    mask["exclusion_reason"] = ""
    mask.loc[:1, "exclusion_reason"] = ["all_zero", "nonzero_constant"]
    for key in ["review_pending", "dc_review", "amplitude_review", "near_constant_review"]:
        mask[key] = mask.index == 2
    for name in ["wave", "qc", "geo"]:
        (tmp_path / name).mkdir()
    wp, mp, gp = (
        tmp_path / x
        for x in [
            "wave/waveform_qc.parquet",
            "qc/fixed_qc_mask.parquet",
            "geo/trace_geometry.parquet",
        ]
    )
    wave.to_parquet(wp, index=False)
    mask.to_parquet(mp, index=False)
    wave.loc[wave.numerically_usable, ["source_file", "trace_index"]].to_parquet(gp, index=False)
    (tmp_path / "wave/metadata.json").write_text(
        json.dumps({"interim_directory": "wave", "waveform_table_sha256": sha256_file(wp)})
    )
    for directory, field, artifact in [
        ("qc", "processed_directory", mp),
        ("geo", "interim_directory", gp),
    ]:
        (tmp_path / directory / "metadata.json").write_text(
            json.dumps(
                {
                    field: directory,
                    "artifacts": {artifact.relative_to(tmp_path).as_posix(): sha256_file(artifact)},
                }
            )
        )
        (tmp_path / directory / "input.lock.json").write_text(
            json.dumps({"wave/waveform_qc.parquet": sha256_file(wp)})
        )
    monkeypatch.setattr(
        "seis_interp.pipelines.search_forge_regions.subprocess.check_output",
        lambda *args, **kwargs: "0" * 40,
    )
    run = run_forge_region_search(tmp_path, study)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"] and summary["eligible_trace_count"] == 254
    assert summary["fixed_exclusions"] == inputs["expected_fixed_exclusions"]
    final = pd.read_csv(run / "finalists.csv")
    assert len(final) == 2 and final.meets_guidelines.all()
    assert (final.fill_fraction == 1).all() and (final.collision_fraction == 0).all()
    definitions = json.loads((run / "candidate_grids.json").read_text())
    for row in final.itertuples():
        mapping = pd.read_parquet(tmp_path / definitions[row.candidate_id]["trace_mapping"])
        assert len(mapping) == 64
        expected = mask.merge(
            mapping[["source_file", "trace_index"]],
            on=["source_file", "trace_index"],
            validate="one_to_one",
        ).sort_values(["source_file", "trace_index"])
        assert mapping.eligible_after_fixed_qc.tolist() == expected.eligible_after_fixed_qc.tolist()
        assert mapping.review_pending.tolist() == expected.review_pending.tolist()
        assert mapping.grid_cell.nunique() == 64
        np.testing.assert_allclose(mapping.source_grid_x_m, mapping.source_x_m, atol=1e-9)
        np.testing.assert_allclose(mapping.receiver_grid_y_m, mapping.receiver_y_m, atol=1e-9)
    assert len(list((run / "figures").glob("*.png"))) == 5
    metadata = json.loads((run / "metadata.json").read_text())
    for name, sha in metadata["artifacts"].items():
        assert sha256_file(tmp_path / name) == sha
    mp.write_bytes(mp.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="input hash mismatch"):
        run_forge_region_search(tmp_path, study)
