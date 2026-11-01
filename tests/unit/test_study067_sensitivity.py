"""Sensitivity contracts and the physical GPU0 admission namespace."""

import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from seis_interp.training import gpu_memory
from seis_interp.training.gpu_admission import choose_admission_device

STUDY = Path(__file__).resolve().parents[2] / "studies/study_067_forge_c3_common_grid"


@pytest.mark.parametrize(
    "arm,dataset,old,new,count",
    [
        ("sa_forge_mask010", "forge", b"inner_mask_fraction: 0.25", b"inner_mask_fraction: 0.1", 1),
        ("sa_c3_mask010_seeds", "c3", b"inner_mask_fraction: 0.25", b"inner_mask_fraction: 0.1", 1),
        ("sb_forge_fp32", "forge", b"mixed_precision: bf16", b"mixed_precision: 'off'", 2),
    ],
)
def test_sensitivity_methods_only_change_the_registered_factor(arm, dataset, old, new, count):
    production = (STUDY / dataset / "ccnet5d_grid.yaml").read_bytes()
    pilot = STUDY / "pilots" / arm
    assert production.count(old) == count
    assert (pilot / "ccnet5d_grid.yaml").read_bytes() == production.replace(old, new)
    assert (pilot / "inputs.yaml").read_bytes() == (STUDY / dataset / "inputs.yaml").read_bytes()
    grid = yaml.safe_load((pilot / "grid.yaml").read_text())
    assert grid == {
        "methods": {
            "ccnet5d_grid": yaml.safe_load((STUDY / "grid.yaml").read_text())["methods"][
                "ccnet5d_grid"
            ]
        }
    }
    contract = yaml.safe_load((pilot / "config.yaml").read_text())
    original = yaml.safe_load((STUDY / dataset / "config.yaml").read_text())
    assert contract["trace_split"] == original["trace_split"]
    assert contract["validation_holdout"] == original["validation_holdout"]
    assert contract["cuda_execution"]["global_admission"] is True
    assert contract["cuda_execution"]["devices"] == ["cuda:0"]
    assert "common_grid" not in contract and contract["methods"] == ["ccnet5d_grid"]
    if arm == "sb_forge_fp32":
        assert (
            contract["cuda_execution"]["minimum_free_memory_gib"]["ccnet5d_grid"]["a03_b02"] == 32
        )


def test_gpu0_pinning_keeps_the_unpinned_production_reservation_namespace(monkeypatch):
    gib = 2**30
    monkeypatch.setattr(
        gpu_memory,
        "_physical_gpu_memory",
        lambda: [("GPU-zero", 93 * gib, 93 * gib), ("GPU-one", 40 * gib, 93 * gib)],
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    production = gpu_memory.gpu_memory_info(["cuda:0", "cuda:1"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    pinned = gpu_memory.gpu_memory_info(["cuda:0"])
    assert pinned == {"cuda:0": production["cuda:0"]}
    slots = [
        {"device": "cuda:0", "required_bytes": 34 * gib, "current_reserved_bytes": 0},
        {"device": "cuda:1", "required_bytes": 47 * gib, "current_reserved_bytes": 40 * gib},
    ]
    assert choose_admission_device(["cuda:0"], {"cuda:0": 93 * gib}, slots, 60 * gib) is None
    assert choose_admission_device(["cuda:0"], {"cuda:0": 93 * gib}, slots, 34 * gib) == "cuda:0"


@pytest.mark.parametrize("arm", ["sa_forge_mask010", "sb_forge_fp32", "sa_c3_mask010_seeds"])
def test_sensitivity_launcher_dry_run_uses_only_frozen_source_and_gpu0(
    tmp_path, monkeypatch, capsys, arm
):
    scripts = STUDY.parents[1] / "scripts"
    experiment = tmp_path / "experiment"
    (experiment / "source/src/seis_interp").mkdir(parents=True)
    c3 = arm == "sa_c3_mask010_seeds"
    entry = experiment / (
        "study/run_seeds.py" if c3 else "source/scripts/run_forge_grid_experiment.py"
    )
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("# synthetic frozen entry")
    monkeypatch.syspath_prepend(str(scripts))
    arguments = ["launch", arm, "--experiment", str(experiment), "--dry-run"]
    if c3:
        arguments += ["--production", str(tmp_path / "production")]
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(scripts / "launch_study067_sensitivity.py"), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["command"][:3] == ["nice", "-n", "19"]
    assert result["command"][4] == str(entry)
    assert result["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert result["environment"]["PYTHONPATH"] == str(experiment / "source/src")
    assert all(
        result["environment"][name] == "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )
    assert not (experiment / "sensitivity.log").exists()


@pytest.mark.parametrize("uuid", ["GPU-01234567-89ab-cdef-0123-456789abcdef", "0"])
def test_launcher_resolves_physical_gpu0_to_uuid_before_start(tmp_path, monkeypatch, uuid):
    scripts = STUDY.parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import subprocess

    import launch_study067_pilots as launcher

    experiment = tmp_path / "experiment"
    (experiment / "source/src/seis_interp").mkdir(parents=True)
    entry = experiment / "source/scripts/run_forge_grid_experiment.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# synthetic frozen entry")
    calls = []

    def resolve(command, **kwargs):
        assert command == ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"]
        return uuid + "\n"

    monkeypatch.setattr(subprocess, "check_output", resolve)
    monkeypatch.setattr(launcher, "start", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["launch", "sb_forge_fp32", "--experiment", str(experiment)])
    if uuid == "0":
        with pytest.raises(ValueError, match="complete GPU UUID"):
            runpy.run_path(str(scripts / "launch_study067_sensitivity.py"), run_name="__main__")
        assert not calls
    else:
        runpy.run_path(str(scripts / "launch_study067_sensitivity.py"), run_name="__main__")
        assert calls == [{"gpu": uuid, "dry_run": False}]
