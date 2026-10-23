import importlib.util
import json
from pathlib import Path

import pytest

from seis_interp.pipelines.probe_forge_ccnet_memory import forge_ccnet_probe_plan

ROOT = Path(__file__).resolve().parents[2]


def probe_script():
    spec = importlib.util.spec_from_file_location(
        "probe_script", ROOT / "scripts/probe_forge_ccnet_memory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_arguments_and_plan_are_cpu_only(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("CUDA initialized"))
    script = probe_script()
    args = script.parse_args(["--device", "cuda:1", "--plan-only"])
    assert args.plan_only and args.device == "cuda:1" and args.minimum_free_memory_gib == 17
    assert args.reserve_gib == 8
    assert script.parse_args(["--device", "cuda:1", "--reserve-gib", "12"]).reserve_gib == 12
    plan = forge_ccnet_probe_plan(ROOT / args.study)
    assert [item["id"] for item in plan] == [f"a{axis:02d}_b05" for axis in range(1, 6)]
    assert [item["config"]["patches"]["shape"][1:] for item in plan] == [
        [4, 8, 4, 8],
        [4, 12, 4, 12],
        [6, 12, 6, 12],
        [8, 12, 8, 12],
        [8, 16, 8, 16],
    ]
    for item in plan:
        config = item["config"]
        assert config["model"]["time_dilation"] == 6
        assert config["prediction"]["core_shape"] == [2172, 4, 8, 4, 8]
        assert not config["execution"]["reuse_patch"]
        assert not config["execution"]["cache_patch_gradients"]
        assert config["training"]["supervised_traces_per_update"] == 64
        assert config["training"]["mixed_precision"] == "bf16"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--device", "cpu"],
        ["--device", "auto"],
        ["--device", "cuda:1", "--minimum-free-memory-gib", "0"],
        ["--device", "cuda:1", "--minimum-free-memory-gib", "nan"],
        ["--device", "cuda:1", "--reserve-gib", "0"],
        ["--device", "cuda:1", "--reserve-gib", "-1"],
        ["--device", "cuda:1", "--reserve-gib", "nan"],
        ["--device", "cuda:1", "--reserve-gib", "inf"],
    ],
)
def test_probe_rejects_ambiguous_device_or_invalid_memory(argv):
    with pytest.raises(SystemExit):
        probe_script().parse_args(argv)


def test_probe_caps_each_patch_skips_after_reserve_and_continues_after_capped_oom(
    tmp_path, monkeypatch
):
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    from seis_interp.data import forge_mvp_run_records, forge_trace_experiment
    from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
    from seis_interp.pipelines import probe_forge_ccnet_memory as probe
    from seis_interp.processing import forge_grid_partition
    from seis_interp.training import ccnet5d_memory_probe

    plan = forge_ccnet_probe_plan(ROOT / "studies/study_067_forge_c3_common_grid/forge")
    study = tmp_path / "study"
    reference = tmp_path / "reference"
    study.mkdir()
    (reference / "study").mkdir(parents=True)
    (reference / "input").mkdir()
    contract = yaml.safe_load(
        (ROOT / "studies/study_067_forge_c3_common_grid/forge/config.yaml").read_text()
    )
    contract["reference_experiment"] = "reference"
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    (reference / "study/config.yaml").write_text(yaml.safe_dump(contract))
    (reference / "experiment.json").write_text(json.dumps({"alignment": {}}))
    partition = pd.DataFrame({"cell_id": [0, 1, 2], "role": ["train", "validation", "test"]})
    partition.to_parquet(reference / "input/manifest.parquet")
    inputs = ForgeModelInputs(
        geometry=partition[["cell_id"]],
        observed_cells=np.array([0]),
        test_cells=np.array([1, 2]),
        observed_values=np.ones((1, 1), dtype=np.float32),
        spatial_shape=(3,),
        time_s=np.array([0.0]),
        global_rms=1.0,
        binding={},
        test_rms=np.ones(2),
    )
    monkeypatch.setattr(probe, "forge_ccnet_probe_plan", lambda _: plan)

    def observed_only(reference, method, *, verify_target_waveforms):
        assert not verify_target_waveforms
        return inputs, {}, partition

    monkeypatch.setattr(forge_trace_experiment, "load_forge_trace_inputs", observed_only)
    monkeypatch.setattr(
        forge_mvp_run_records, "execution_identity", lambda _: {"git_commit": "testcommit"}
    )
    monkeypatch.setattr(forge_grid_partition, "partition_forge_traces", lambda *_: partition)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    events = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    free_gib = iter([7, 24, 25, 50, 40])

    def memory(device):
        assert device == "cuda:1"
        assert events[-1] == "empty_cache"
        events.append("memory")
        return next(free_gib) * 2**30, 100 * 2**30

    def cap(fraction, device):
        assert device == "cuda:1"
        assert events[-1] == "memory"
        events.append(("cap", fraction))

    measured = []

    def measure(inputs, config, device):
        assert events[-1][0] == "cap"
        measured.append(config["patches"]["shape"][1:])
        if len(measured) == 1:
            raise torch.cuda.OutOfMemoryError("allocator cap")
        return {"status": "passed", "measured_peak_reserved_gib_ceil": 20}

    monkeypatch.setattr(torch.cuda, "mem_get_info", memory)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", cap)
    monkeypatch.setattr(ccnet5d_memory_probe, "measure_ccnet5d_memory", measure)
    output = probe.run_forge_ccnet_memory_probe(tmp_path, study, "cuda:1", 17)
    report = json.loads((output / "memory_probe.json").read_text())
    assert report["reserve_gib"] == 8
    rows = report["patches"]
    assert [row["status"] for row in rows] == [
        "skipped",
        "skipped",
        "exceeds_cap",
        "passed",
        "passed",
    ]
    assert all(row["reason"] == "insufficient_free_memory" for row in rows[:2])
    assert rows[2]["uncapped_peak_exceeds_cap"]
    for row, cap_gib in zip(rows, [0, 16, 17, 42, 32], strict=True):
        assert row["reserve_gib"] == 8
        assert row["cap_bytes"] == cap_gib * 2**30
        assert row["per_process_memory_fraction"] == cap_gib / 100
    assert len(measured) == 3
    assert report["status"] == "incomplete"
    assert report["validation_scoring_calls"] == report["test_scoring_calls"] == 0


@pytest.mark.parametrize("reserve", [0, -1, float("nan"), float("inf")])
def test_probe_runner_validates_reserve_before_loading_inputs(tmp_path, reserve):
    from seis_interp.pipelines.probe_forge_ccnet_memory import run_forge_ccnet_memory_probe

    with pytest.raises(ValueError, match="reserve_gib"):
        run_forge_ccnet_memory_probe(tmp_path, tmp_path, "cuda:1", 17, reserve)


def test_timing_e2e_and_benchmark_plans_keep_fresh_updates():
    study = ROOT / "studies/study_067_forge_c3_common_grid/forge"
    plan = forge_ccnet_probe_plan(study, mode="end-to-end")
    assert [item["id"] for item in plan] == ["a01_b02", "a05_b02"]
    paired = forge_ccnet_probe_plan(
        study,
        mode="timing",
        candidate_ids=["a05_b05"],
        cudnn_benchmark="both",
        time_kernel_size=9,
    )
    assert [item["id"] for item in paired] == [
        "a05_b05_benchmark_off_order1",
        "a05_b05_benchmark_on_order1",
        "a05_b05_benchmark_on_order2",
        "a05_b05_benchmark_off_order2",
    ]
    assert [item["config"]["execution"]["cudnn_benchmark"] for item in paired] == [
        False,
        True,
        True,
        False,
    ]
    assert [item["benchmark_order"] for item in paired] == ["off,on", "off,on", "on,off", "on,off"]
    for item in paired:
        config = item["config"]
        assert config["model"]["time_kernel_size"] == 9
        assert (config["model"]["hidden_channels"], config["model"]["intermediate_channels"]) == (
            36,
            37,
        )
        assert config["patches"]["inner_mask_fraction"] == 0.25
        assert not config["execution"]["cache_patch_gradients"]
    args = probe_script().parse_args(
        [
            "--device",
            "cuda:1",
            "--mode",
            "timing",
            "--updates",
            "100",
            "--cudnn-benchmark",
            "both",
        ]
    )
    assert args.updates == 100 and args.cudnn_benchmark == "both"


def test_end_to_end_finalizes_prediction_then_scores_only_validation(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    from seis_interp.data import forge_trace_experiment
    from seis_interp.pipelines.probe_forge_ccnet_memory import _end_to_end_probe
    from seis_interp.training import forge_c3_methods
    from seis_interp.training.forge_mvp_methods import ForgeMethodResult

    values = np.ones((2, 3), dtype=np.float32)
    cells = np.array([11, 12])
    result = ForgeMethodResult(
        values,
        cells,
        [],
        {},
        {"g0": {"status": "insufficient_updates"}},
        {"0.999": values, "0.9995": values * 2},
    )
    finalized = []

    def run(inputs, config, *, checkpoint_path):
        checkpoint_path.write_bytes(b"test checkpoint")
        finalized.append(True)
        return result

    def read(reference, domain):
        assert finalized and domain == "validation"
        assert (tmp_path / "prediction.npy").exists()
        return pd.DataFrame({"cell_id": cells, "clean_target": [True, True]}), values

    monkeypatch.setattr(forge_c3_methods, "run_forge_c3_method", run)
    monkeypatch.setattr(forge_trace_experiment, "read_forge_trace_targets", read)
    record = _end_to_end_probe(None, {"device": "cpu"}, tmp_path, tmp_path)
    assert record["validation_scoring_calls"] == 1 and record["test_scoring_calls"] == 0
    assert set(record["ema_shadow_metrics"]) == {"0.999", "0.9995"}
    assert sorted(path.name for path in tmp_path.glob("*.npy")) == [
        "cell_ids.npy",
        "prediction.npy",
    ]
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "passed"


def test_probe_times_updates_after_separate_warmup_and_reports_median(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from seis_interp.data import forge_aligned_inputs
    from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
    from seis_interp.training import ccnet5d_memory_probe as probe

    inputs = ForgeModelInputs(
        geometry=pd.DataFrame({"cell_id": np.arange(8)}),
        observed_cells=np.arange(6),
        test_cells=np.array([6, 7]),
        observed_values=np.ones((6, 7), dtype=np.float32),
        spatial_shape=(1, 2, 1, 4),
        time_s=np.arange(7) * 0.004,
        global_rms=1.0,
        binding={},
    )
    config = {
        "method_id": "ccnet5d_grid",
        "model_seed": 7301,
        "model": {"hidden_channels": 1, "intermediate_channels": 1, "kernel_size": 1},
        "alignment": {},
        "patches": {"shape": [7, 1, 2, 1, 4], "inner_mask_fraction": 0.5},
        "prediction": {"core_shape": [7, 1, 2, 1, 4]},
        "training": {
            "learning_rate": 0.001,
            "report_interval": 100,
            "ema_decay": 0.9,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "supervised_traces_per_update": 2,
            "mixed_precision": "off",
        },
        "loss": "masked_trace_mse",
    }
    monkeypatch.setattr(
        forge_aligned_inputs, "align_model_inputs", lambda inputs, _: (inputs, None)
    )
    # Setup takes 100 s and total training takes 3900 s on this fake clock.
    # Only three post-warm-up intervals (2, 4, 3 s) enter the speed statistic.
    ticks = iter([0, 100, 1000, 1002, 2000, 2004, 3000, 3003, 4000, 5000])
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    record = probe.measure_ccnet5d_memory(
        inputs, config, "cpu", optimizer_updates=3, warmup_updates=2
    )
    assert record["per_update_seconds"] == [2, 4, 3]
    assert record["seconds_per_update"] == 3
    assert record["training_seconds"] == 3900
    assert not record["timing_includes_startup"]
    assert record["timing_statistic"] == "per_update_median"
    assert record["warmup_optimizer_updates"] == 2
    assert record["timed_optimizer_updates"] == 3
    assert record["training_probe_optimizer_updates"] == 5
    assert record["diagnostics"]["supervised_traces_per_update"] == 2
    assert record["prediction_calls"] == 1


def test_benchmark_comparisons_preserve_both_orders_and_ignore_startup_and_outliers():
    from seis_interp.training.ccnet5d_memory_probe import benchmark_comparisons

    records = [
        {
            "status": "passed",
            "candidate_id": "a05_b05",
            "benchmark_order": order,
            "cudnn_benchmark": enabled,
            "per_update_seconds": updates,
            "warmup_optimizer_updates": 10,
            "training_seconds": startup,
        }
        for order, enabled, updates, startup in [
            ("off,on", False, [2, 3, 100], 1000),
            ("off,on", True, [1, 1, 100], 2000),
            ("on,off", True, [1, 1, 1], 3000),
            ("on,off", False, [2, 3, 4], 4000),
        ]
    ]
    comparison = benchmark_comparisons(records)["a05_b05"]
    assert comparison["off"] == 3 and comparison["on"] == 1
    assert comparison["speedup"] == 3
    assert comparison["orders"] == [
        {"order": ["off", "on"], "off": 3, "on": 1, "speedup": 3},
        {"order": ["on", "off"], "off": 3, "on": 1, "speedup": 3},
    ]
    assert benchmark_comparisons(records[:2]) == {}
    records[-1]["status"] = "exceeds_cap"
    assert benchmark_comparisons(records) == {}
