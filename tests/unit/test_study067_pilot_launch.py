import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c1_launch(tmp_path, monkeypatch):
    import yaml

    launch = script("launch_study067_pilots")
    monkeypatch.setattr(launch, "REPO", tmp_path)
    names = {name: name for name in launch.PREPARED if name.startswith("c1_")}
    monkeypatch.setattr(launch, "PREPARED", names)
    for name in names:
        c3 = "_c3_" in name
        resources = {
            "minimum_free_memory_gib": {
                "ccnet5d_grid": {"a05_b02" if c3 else "a03_b02": 32 if c3 else 17}
            },
            "admission_margin_gib": 2,
            "excluded_devices_while_pids_alive": {
                "cuda:0": [{"pid": 3958956, "process_start_time": 1111834778}]
            },
        }
        folder = tmp_path / name / "study"
        folder.mkdir(parents=True)
        (folder / "config.yaml").write_text(yaml.safe_dump({"cuda_execution": resources}))
    return launch


def test_pilot_commands_use_frozen_code_validation_only_and_expected_candidates():
    launch = script("launch_study067_pilots")
    for name in launch.PREPARED:
        if name.startswith("d2_"):
            continue
        command = launch.pilot_command(name)
        assert "pilot" in command
        assert any("source/scripts/run_" in value for value in command)
        candidate = command[command.index("--candidate") + 1]
        if name.startswith("c1_"):
            assert candidate == ("a05_b02" if "_c3_" in name else "a03_b02")
            assert command[command.index("--method") + 1] == "ccnet5d_grid"
            runner = (
                "run_c3_region_experiment.py" if "_c3_" in name else "run_forge_grid_experiment.py"
            )
            assert command[1].endswith(runner)
        else:
            assert candidate == ("a02_b04" if name.startswith("n1_") else "a02_b02")
        if name.startswith("d1_"):
            assert command[:3] == ["nice", "-n", "10"]


def test_d2_commands_cover_both_heavy_cells_on_both_datasets():
    launch = script("launch_study067_pilots")
    commands = list(launch.d2_commands(24))
    assert len(commands) == 4
    cells = set()
    for experiment, candidate, command in commands:
        assert command[:3] == ["nice", "-n", "10"]
        assert str(experiment / "study/measure_drr.py") in command
        assert command[command.index("--address-space-cap-gib") + 1] == "24"
        cells.add((command[command.index("--dataset") + 1], candidate))
    assert cells == {
        (dataset, cell) for dataset in ("forge", "c3") for cell in ("a05_b04", "a05_b05")
    }


def test_dry_run_starts_nothing_and_prints_only_declared_environment(monkeypatch, capsys, tmp_path):
    launch = script("launch_study067_pilots")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-printed")

    def forbidden(*args, **kwargs):
        pytest.fail("dry run attempted to start a process")

    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    launch.start(["python", "pilot"], tmp_path, "n1.log", gpu="GPU-test", dry_run=True)
    output = capsys.readouterr().out
    assert "must-not-be-printed" not in output
    record = json.loads(output)
    assert record["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert record["environment"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert not list(tmp_path.iterdir())


def test_launch_detaches_and_directs_logs_and_gpu_visibility(monkeypatch, tmp_path):
    launch = script("launch_study067_pilots")
    captured = {}

    def start(command, **kwargs):
        captured.update(kwargs)
        kwargs["stdout"].write("pilot output\n")
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(launch.subprocess, "Popen", start)
    process = launch.start(["pilot"], tmp_path, "n1.log", gpu="GPU-test")
    assert process.pid == 1234
    assert captured["start_new_session"] is True
    assert captured["stdin"] == launch.subprocess.DEVNULL
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert captured["env"]["PYTHONPATH"] == str(tmp_path / "source/src")
    assert (tmp_path / "n1.log").read_text() == "pilot output\n"
    with pytest.raises(FileExistsError):
        launch.start(["pilot"], tmp_path, "n1.log", gpu="GPU-test")


def test_c1_dry_run_routes_all_four_arms_without_processes_or_gpu_queries(
    monkeypatch, capsys, c1_launch
):
    launch = c1_launch

    def forbidden(*args, **kwargs):
        pytest.fail("C-1 dry run attempted a launch or GPU query")

    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    monkeypatch.setattr(launch.subprocess, "check_output", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch", "c1", "--gpu-uuid-c3", "GPU-aaaa", "--gpu-uuid-forge", "GPU-bbbb", "--dry-run"],
    )
    launch.main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 4
    assert len({record["experiment"] for record in records}) == 4
    for name, record in zip(
        (name for name in launch.PREPARED if name.startswith("c1_")), records, strict=True
    ):
        assert record["command"] == launch.pilot_command(name)
        assert record["log"] == record["experiment"] + "/c1.log"
        assert record["environment"]["CUDA_VISIBLE_DEVICES"] == (
            "GPU-aaaa" if "_c3_" in name else "GPU-bbbb"
        )
        assert record["environment"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert record["dry_run"] is True


@pytest.mark.parametrize(
    "arguments", [[], ["--gpu-uuid-c3", "GPU-aaaa"], ["--gpu-uuid-forge", "GPU-bbbb"]]
)
def test_c1_requires_both_dataset_gpu_uuids(monkeypatch, arguments):
    launch = script("launch_study067_pilots")
    monkeypatch.setattr(sys, "argv", ["launch", "c1", *arguments, "--dry-run"])
    with pytest.raises(SystemExit) as error:
        launch.main()
    assert error.value.code == 2


@pytest.mark.parametrize("short_gpu", [None, "GPU-aaaa", "GPU-bbbb"])
def test_c1_preflight_checks_both_concurrent_arms_and_margin(monkeypatch, short_gpu, c1_launch):
    launch = c1_launch
    monkeypatch.setattr(launch, "pinned_process_alive", lambda item: False)
    queried = []

    def output(command, **kwargs):
        gpu = next(value.removeprefix("--id=") for value in command if value.startswith("--id="))
        queried.append(gpu)
        gib = 68 if gpu == "GPU-aaaa" else 38
        return f"{gpu}, {gib * 1024 - (1 if gpu == short_gpu else 0)}\n"

    monkeypatch.setattr(launch.subprocess, "check_output", output)
    if short_gpu:
        with pytest.raises(ValueError, match="C-1 batch requires"):
            launch.check_c1_gpus("GPU-aaaa", "GPU-bbbb")
    else:
        launch.check_c1_gpus("GPU-aaaa", "GPU-bbbb")
        assert queried == ["GPU-aaaa", "GPU-bbbb"]


def test_c1_preflight_sums_both_datasets_on_the_same_uuid(monkeypatch, c1_launch):
    launch = c1_launch
    monkeypatch.setattr(launch, "pinned_process_alive", lambda item: False)
    monkeypatch.setattr(
        launch.subprocess, "check_output", lambda *a, **k: f"GPU-aaaa, {105 * 1024}"
    )
    with pytest.raises(ValueError, match="106 GiB"):
        launch.check_c1_gpus("GPU-aaaa", "GPU-aaaa")


def test_c1_preflight_rejects_live_pinned_worker_on_selected_physical_gpu(monkeypatch, c1_launch):
    launch = c1_launch
    monkeypatch.setattr(launch, "pinned_process_alive", lambda item: True)

    def output(command, **kwargs):
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return "GPU-aaaa, 3958956\n"
        return f"GPU-aaaa, {100 * 1024}\n"

    monkeypatch.setattr(launch.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="live pinned legacy worker"):
        launch.check_c1_gpus("GPU-aaaa", "GPU-bbbb")


def test_c1_checks_all_log_destinations_before_starting(monkeypatch, tmp_path):
    launch = script("launch_study067_pilots")
    monkeypatch.setattr(launch, "REPO", tmp_path)
    monkeypatch.setattr(launch, "PREPARED", {f"c1_{i}": str(i) for i in range(4)})
    (tmp_path / "3").mkdir()
    (tmp_path / "3/c1.log").touch()

    def forbidden(*args, **kwargs):
        pytest.fail("C-1 started before all destinations were checked")

    monkeypatch.setattr(launch, "start", forbidden)
    monkeypatch.setattr(launch, "check_c1_gpus", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["launch", "c1", "--gpu-uuid-c3", "GPU-aaaa", "--gpu-uuid-forge", "GPU-bbbb"]
    )
    with pytest.raises(FileExistsError, match="already launched"):
        launch.main()


@pytest.fixture
def d1_launch(tmp_path, monkeypatch):
    launch = script("launch_study067_pilots")
    monkeypatch.setattr(launch, "REPO", tmp_path)
    names = {name: name for name in launch.PREPARED if name.startswith("d1_")}
    monkeypatch.setattr(launch, "PREPARED", names)
    for name in names:
        experiment = tmp_path / name
        experiment.mkdir()
        if name.endswith("_rerun"):
            continue
        manifest = experiment / "method_runs/drr_grid/S1/a02_b02/manifest.json"
        manifest.parent.mkdir(parents=True)
        status = "running" if name == "d1_forge_d7" else "completed"
        manifest.write_text(json.dumps({"status": status, "test_scoring_calls": 0}))
        (experiment / "d1.log").write_text("previous launch\n")
    return launch


def test_d1_dry_run_selects_only_the_fresh_rerun_without_processes_or_gpu_queries(
    monkeypatch, capsys, d1_launch
):
    launch = d1_launch
    before = {path: path.read_bytes() for path in launch.REPO.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("D-1 dry run attempted a launch or GPU query")

    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    monkeypatch.setattr(launch.subprocess, "check_output", forbidden)
    monkeypatch.setattr(sys, "argv", ["launch", "d1", "--dry-run"])
    launch.main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 1
    record = records[0]
    experiment = launch.REPO / launch.PREPARED["d1_forge_d7_rerun"]
    assert record["experiment"] == str(experiment)
    assert record["command"] == launch.pilot_command("d1_forge_d7_rerun")
    assert record["command"][-4:] == ["--method", "drr_grid", "--candidate", "a02_b02"]
    assert record["log"] == str(experiment / "d1.log")
    assert record["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert record["environment"]["PYTHONPATH"] == str(experiment / "source/src")
    assert record["dry_run"] is True
    assert {path: path.read_bytes() for path in launch.REPO.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_d1_skips_all_completed_pilots_even_with_existing_logs(monkeypatch, d1_launch, dry_run):
    launch = d1_launch
    experiment = launch.REPO / launch.PREPARED["d1_forge_d7_rerun"]
    manifest = experiment / "method_runs/drr_grid/S1/a02_b02/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "completed"}))
    (experiment / "d1.log").touch()

    def forbidden(*args, **kwargs):
        pytest.fail("D-1 attempted to launch a completed or superseded pilot")

    monkeypatch.setattr(launch, "start", forbidden)
    monkeypatch.setattr(sys, "argv", ["launch", "d1", *(["--dry-run"] if dry_run else [])])
    launch.main()


def test_d1_checks_all_pending_log_destinations_before_starting(monkeypatch, d1_launch):
    launch = d1_launch
    experiment = launch.REPO / launch.PREPARED["d1_c3_d7"]
    (experiment / "method_runs/drr_grid/S1/a02_b02/manifest.json").write_text(
        json.dumps({"status": "running"})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("D-1 started before all destinations were checked")

    monkeypatch.setattr(launch, "start", forbidden)
    monkeypatch.setattr(sys, "argv", ["launch", "d1"])
    with pytest.raises(FileExistsError, match="already launched"):
        launch.main()


def test_check_d1_uses_the_rerun_and_both_other_references(monkeypatch, capsys, d1_launch):
    launch = d1_launch
    checked = []

    def check(experiment):
        checked.append(experiment)
        return {"experiment": str(experiment), "passed": True}

    monkeypatch.setitem(sys.modules, "check_study067_d1", SimpleNamespace(check=check))
    monkeypatch.setattr(sys, "argv", ["launch", "check-d1"])
    with pytest.raises(SystemExit) as result:
        launch.main()
    assert result.value.code == 0
    assert checked == [
        launch.REPO / launch.PREPARED[name]
        for name in ("d1_forge_d7_rerun", "d1_c3_d7", "d1_forge_d5b")
    ]
    assert len(json.loads(capsys.readouterr().out)) == 3


@pytest.mark.parametrize(
    "dataset,candidate,precision,budget",
    [("c3", "a05_b02", "off", 32), ("forge", "a03_b02", "bf16", 17)],
)
@pytest.mark.parametrize("kernel,parameters", [(3, 363292), (9, 363227)])
def test_c1_study_arms_preserve_s1_identity_and_final_settings(
    dataset, candidate, precision, budget, kernel, parameters
):
    from copy import deepcopy

    import yaml

    from seis_interp.data.experiment_imports import method_config_hash
    from seis_interp.processing.c3_region_search import c3_search_candidates
    from seis_interp.training.forge_mvp_methods import model_parameter_count

    study = REPO / "studies/study_067_forge_c3_common_grid"
    pilot = study / "pilots" / f"c1_{dataset}_tk{kernel}"
    contract = yaml.safe_load((pilot / "config.yaml").read_text())
    final = yaml.safe_load((study / dataset / "config.yaml").read_text())
    assert contract["methods"] == ["ccnet5d_grid"]
    assert contract["import_from"] == []
    assert "common_grid" not in contract
    for key in ("trace_split", "validation_holdout", "mask_id", "mask_seed", "preprocessing"):
        assert contract[key] == final[key]
    assert (pilot / "inputs.yaml").read_bytes() == (study / dataset / "inputs.yaml").read_bytes()
    resources = contract["cuda_execution"]
    assert resources["minimum_free_memory_gib"]["ccnet5d_grid"][candidate] == budget
    assert resources["admission_margin_gib"] == 2
    assert resources["global_admission"] is False
    base = yaml.safe_load((study / dataset / "ccnet5d_grid.yaml").read_text())
    expected = deepcopy(base)
    if kernel == 9:
        expected["model"].update(time_kernel_size=9, hidden_channels=36, intermediate_channels=37)
    arm = yaml.safe_load((pilot / "ccnet5d_grid.yaml").read_text())
    grid = yaml.safe_load((pilot / "grid.yaml").read_text())["methods"]["ccnet5d_grid"]
    assert grid == yaml.safe_load((study / "grid.yaml").read_text())["methods"]["ccnet5d_grid"]
    actual = next(
        item["config"] for item in c3_search_candidates(arm, grid) if item["id"] == candidate
    )
    expected = next(
        item["config"] for item in c3_search_candidates(expected, grid) if item["id"] == candidate
    )
    assert method_config_hash(actual) == method_config_hash(expected)
    assert model_parameter_count(arm, arm["patches"]["shape"][0]) == parameters
    assert arm["training"]["mixed_precision"] == precision
    assert arm["training"]["ema_eval_decays"] == [0.999, 0.9995, 0.9998]
    assert arm["training"]["ema_decay"] == 0.999
    assert arm["training"]["max_steps"] == 50000
    assert arm["patches"]["inner_mask_fraction"] == 0.25
    assert arm["execution"]["cudnn_benchmark"] is True
    assert arm["execution"]["reuse_patch"] is False
    assert arm["execution"]["cache_patch_gradients"] is False
    assert base["execution"]["cudnn_benchmark"] is True
    assert (study / dataset / "pocs_grid.yaml").read_bytes() == (
        study / dataset / "pocs_k1.yaml"
    ).read_bytes()


@pytest.mark.parametrize("difference,passed", [(0.009, True), (0.011, False)])
@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_d1_acceptance_uses_the_dataset_metric_and_tolerance(tmp_path, difference, passed, dataset):
    checker = script("check_study067_d1")
    (tmp_path / "study").mkdir()
    (tmp_path / "study/d1_reference.json").write_text(
        json.dumps({"dataset": dataset, "snr": 1.0, "label": "scratch_d7"})
    )
    run = tmp_path / "method_runs/drr_grid/S1/a02_b02"
    run.mkdir(parents=True)
    metrics = {"mean_trace_snr_db": 1.0 + difference}
    if dataset == "forge":
        metrics = {"clean_target": metrics}
    manifest = {"status": "completed", "test_scoring_calls": 0, "metrics": {"validation": metrics}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert checker.check(tmp_path)["passed"] is passed
    manifest["test_scoring_calls"] = 1
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed validation-only run"):
        checker.check(tmp_path)
