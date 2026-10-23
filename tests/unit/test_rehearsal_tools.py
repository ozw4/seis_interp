"""CPU-only generation, checking, detached launch and sampling evidence."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.launch_grid_experiment import launch
from seis_interp.evaluation.study067_rehearsal_checks import (
    capture_drill_snapshot,
    check_rehearsal_plan,
    check_study067_rehearsal,
)
from seis_interp.training.study067_rehearsal_generation import make_study067_rehearsal
from tests.fixtures.c3_region_experiment import tiny_c3_trace_study

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny_source(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    study, _, _ = tiny_c3_trace_study(fixture)
    ccnet_path = study / "ccnet5d_grid.yaml"
    ccnet = yaml.safe_load(ccnet_path.read_text())
    ccnet.setdefault("execution", {})["cudnn_benchmark"] = True
    ccnet_path.write_text(yaml.safe_dump(ccnet))
    source = tmp_path / "source"
    for dataset in ("c3", "forge"):
        shutil.copytree(study, source / dataset)
    return source


@pytest.mark.parametrize("variant", ["faults", "twin", "imports", "determinism"])
def test_generator_only_changes_allowed_values_and_checker_accepts_plans(
    tmp_path, tiny_source, variant
):
    out = tmp_path / "generated"
    audit = make_study067_rehearsal(
        tiny_source,
        out,
        no_faults=variant == "twin",
        imports_only=variant == "imports",
        determinism=variant == "determinism",
    )
    assert audit["variant"] == variant
    assert len(check_rehearsal_plan(out)) == 2
    for dataset in ("c3", "forge"):
        for name in ("grid.yaml", "inputs.yaml", "pocs_grid.yaml"):
            assert (out / dataset / name).read_bytes() == (
                tiny_source / dataset / name
            ).read_bytes()
        before = yaml.safe_load((tiny_source / dataset / "config.yaml").read_text())
        after = yaml.safe_load((out / dataset / "config.yaml").read_text())
        assert after["trace_split"]["evaluation_seeds"] == [990671, 990672, 990673]
        assert after.get("cuda_execution") == before.get("cuda_execution")
        assert after.get("import_from") == before.get("import_from")
        plan = yaml.safe_load((out / dataset / "rehearsal.yaml").read_text())
        assert bool(plan["faults"]) == (variant == "faults")
        holds = [fault for fault in plan["faults"] if fault["action"] == "barrier_hold"]
        assert len(holds) == (1 if variant == "faults" else 0)
        if holds:
            assert (holds[0]["method"], holds[0]["run"], holds[0]["point"]) == (
                "drr_grid",
                "a01_b01",
                "pre_method",
            )
            assert holds[0]["seconds"] == 21600
        config = yaml.safe_load((out / dataset / "ccnet5d_grid.yaml").read_text())
        assert config["execution"]["cudnn_benchmark"] is (variant != "determinism")
        benchmark_edits = [
            edit for edit in audit["changes"] if edit["key"] == "execution.cudnn_benchmark"
        ]
        assert len(benchmark_edits) == (2 if variant == "determinism" else 0)
        if variant == "imports":
            assert after["matched_training"] == before["matched_training"]
        else:
            assert after["matched_training"]["max_steps"] == 300
    with pytest.raises(FileExistsError, match="refusing"):
        make_study067_rehearsal(tiny_source, out)


@pytest.mark.parametrize("variant", [[], ["--determinism"]])
def test_generator_cli_then_checker_cpu_dry_run(tmp_path, tiny_source, variant):
    out = tmp_path / "cli_generated"
    command = [
        sys.executable,
        str(REPO / "scripts/make_study067_rehearsal.py"),
        "--source",
        str(tiny_source),
        "--out",
        str(out),
    ] + variant
    made = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert made.returncode == 0, made.stderr
    assert json.loads(made.stdout)["burned_evaluation_seeds"] == [990671, 990672, 990673]
    checked = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/check_study067_rehearsal.py"),
            "--plan-only",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stderr
    assert "PASS generated plan (runtime acceptance not evaluated)" in checked.stdout


def test_checker_fails_tampered_grid_and_missing_runtime_evidence(tmp_path, tiny_source):
    out = tmp_path / "generated"
    make_study067_rehearsal(tiny_source, out)
    with (out / "c3/grid.yaml").open("a") as stream:
        stream.write("\n# changed bytes\n")
    with pytest.raises(ValueError, match="immutable YAML"):
        check_rehearsal_plan(out)
    results = check_study067_rehearsal(
        tmp_path / "missing_c3",
        tmp_path / "missing_forge",
        tmp_path / "missing_gpu_log",
        launch_commit="fixture",
    )
    assert len(results) == 9
    assert not any(row["passed"] for row in results)
    assert all(row["evidence"] for row in results)
    assert {row["area"] for row in results} == {
        "Freeze and launch",
        "Admission and sharding",
        "Warm-up",
        "Resume",
        "Specific faults",
        "Divergence",
        "Barrier",
        "Summaries",
        "R0 (imports)",
    }


def test_launcher_uses_snapshot_for_supervisor_and_detaches_without_signals(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"
    script = experiment / "source/scripts/run_c3_region_experiment.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture\n")
    (experiment / "source/src/seis_interp").mkdir(parents=True)
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr("scripts.launch_grid_experiment.subprocess.Popen", popen)
    monkeypatch.setenv("PYTHONPATH", "/unfrozen/src")
    info = launch(tmp_path, experiment, "c3", only_dead_workers=True, dry_run=True)
    assert not calls and not (experiment / "supervisor.out").exists()
    assert str(script) in info["command"]
    assert info["command"][-1] == "--only-dead-workers"
    launch(tmp_path, experiment, "c3", only_dead_workers=True)
    command, kwargs = calls[0]
    assert command[0] == "nohup"
    assert kwargs["env"]["PYTHONPATH"] == str(experiment / "source/src")
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_sampler_collects_slot_gpu_and_frozen_supervisor_environment(tmp_path, monkeypatch):
    import os

    from seis_interp.training import rehearsal_observations as observations
    from seis_interp.training.worker_processes import process_start_time

    experiment = tmp_path / "experiment"
    workers = experiment / "workers"
    workers.mkdir(parents=True)
    (workers / "supervisor.json").write_text(json.dumps({"pid": os.getpid(), "status": "running"}))
    (workers / "s1_barrier.json").write_text('{"passed":true}')
    slots = tmp_path / "runs/.gpu_admission"
    slots.mkdir(parents=True)
    (slots / "slot_fixture.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "process_start_time": process_start_time(os.getpid()),
                "required_bytes": 100,
                "device": "cuda:0",
            }
        )
    )

    def query(command, **kwargs):
        return (
            f"GPU-fixture, {os.getpid()}, 1\n"
            if "--query-compute-apps" in command[1]
            else "0, GPU-fixture, 100, 2, 98, 50\n"
        )

    monkeypatch.setattr(observations.subprocess, "check_output", query)
    monkeypatch.setattr(observations.time, "sleep", lambda _: None)
    sample = observations.sample_rehearsal(tmp_path, [experiment])
    assert sample["gpus"][0]["total_bytes"] == 100 * 2**20
    assert sample["slots_before"][0]["required_bytes"] == 100
    assert sample["slots_before"] == sample["slots_after"]
    assert sample["slots_before"][0]["slot_file"] == "slot_fixture.json"
    assert sample["experiments"][str(experiment)]["processes"][0]["pid"] == os.getpid()
    assert sample["experiments"][str(experiment)]["barrier_sha256"]


def test_drill_snapshots_are_exclusive_and_cover_worker_and_summary_bytes(tmp_path):
    (tmp_path / "workers").mkdir()
    (tmp_path / "workers/m.json").write_text('{"status":"completed"}')
    (tmp_path / "summary.csv").write_text("seed_count\n3\n")
    before = capture_drill_snapshot(tmp_path, "before_relaunch")
    after = capture_drill_snapshot(tmp_path, "after_relaunch")
    assert json.loads(before.read_text()) == json.loads(after.read_text())
    with pytest.raises(FileExistsError):
        capture_drill_snapshot(tmp_path, "before_relaunch")


def test_sampler_reads_slots_before_gpu_query_and_after_one_and_a_half_monitor_polls(
    tmp_path, monkeypatch
):
    from seis_interp.training import rehearsal_observations as observations

    directory = tmp_path / "runs/.gpu_admission"
    directory.mkdir(parents=True)
    path = directory / "slot_ramping.json"
    slot = {"pid": 1, "device": "cuda:0", "current_reserved_now_bytes": 0}
    path.write_text(json.dumps(slot))
    events = []
    original = observations._slot_samples

    def slots(repo):
        events.append("slots")
        return original(repo)

    def gpu_query():
        events.append("gpu")
        return [], None

    def monitor_polls(seconds):
        assert seconds >= 1.5  # Production admission monitor poll is 1 second.
        events.append("polls")
        path.write_text(json.dumps(slot | {"current_reserved_now_bytes": 14 * 2**30}))

    monkeypatch.setattr(observations, "process_is_alive", lambda *_: True)
    monkeypatch.setattr(observations, "_slot_samples", slots)
    monkeypatch.setattr(observations, "_gpu_samples", gpu_query)
    monkeypatch.setattr(observations.time, "sleep", monitor_polls)
    sample = observations.sample_rehearsal(tmp_path, [])
    assert events == ["slots", "gpu", "polls", "slots"]
    assert sample["slots_before"][0]["current_reserved_now_bytes"] == 0
    assert sample["slots_after"][0]["current_reserved_now_bytes"] == 14 * 2**30


@pytest.mark.parametrize(
    "failure",
    [
        OSError("missing"),
        subprocess.CalledProcessError(1, "nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 10),
        "0, GPU-a, [N/A], 2, 3, 4",
        "",
    ],
)
def test_sampler_continues_through_failed_gpu_query_to_completed_supervisors(
    tmp_path, monkeypatch, failure
):
    from seis_interp.training import rehearsal_observations as observations

    experiment = tmp_path / "experiment"
    workers = experiment / "workers"
    workers.mkdir(parents=True)
    (experiment / "experiment.json").write_text('{"rehearsal": true}')
    (workers / "supervisor.json").write_text('{"status": "running"}')
    queries = []

    def query(*args, **kwargs):
        queries.append(1)
        if len(queries) == 1:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return "0, GPU-a, 100, 2, 98, 50\n"

    def sleep(seconds):
        if seconds == 10:
            (workers / "supervisor.json").write_text('{"status": "completed"}')

    monkeypatch.setattr(observations.subprocess, "check_output", query)
    monkeypatch.setattr(observations.time, "sleep", sleep)
    output = tmp_path / "samples.jsonl"
    observations.sample_until_complete(tmp_path, [experiment], output)
    samples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(samples) == 2
    assert samples[0]["gpu_error"] and samples[0]["gpus"] == []
    assert samples[0]["experiments"][str(experiment)]["states"]["supervisor"]["status"] == "running"
    assert samples[1]["gpu_error"] is None and samples[1]["gpus"]
    assert (
        samples[1]["experiments"][str(experiment)]["states"]["supervisor"]["status"] == "completed"
    )


def test_refusal_snapshot_allows_worker_progress_but_detects_identity_and_retry_changes(tmp_path):
    workers = tmp_path / "workers"
    workers.mkdir()
    state = {
        "pid": 123,
        "process_start_time": 100,
        "method_id": "nersi_profile",
        "status": "running",
        "phase": "waiting_for_all_S1",
        "completed_evaluation": 0,
    }
    path = workers / "nersi_profile.json"
    path.write_text(json.dumps(state))
    before = json.loads(capture_drill_snapshot(tmp_path, "before").read_text())
    path.write_text(
        json.dumps(
            state
            | {
                "phase": "evaluation",
                "completed_evaluation": 1,
                "status": "retry_pending",
                "child_processes": [{"pid": 99}],
            }
        )
    )
    progressed = json.loads(capture_drill_snapshot(tmp_path, "progress").read_text())
    assert before["worker_sha256"] == progressed["worker_sha256"]
    path.write_text(json.dumps(state | {"pid": 124}))
    replaced = json.loads(capture_drill_snapshot(tmp_path, "replaced").read_text())
    assert before["worker_sha256"] != replaced["worker_sha256"]
    path.write_text(json.dumps(state))
    (workers / "nersi_profile_retries.json").write_text('{"retries": 1}')
    retried = json.loads(capture_drill_snapshot(tmp_path, "retried").read_text())
    assert before["worker_sha256"] != retried["worker_sha256"]


def test_runbook_launch_gate_stops_launches_on_failed_precondition(tmp_path):
    runbook = (REPO / "studies/study_067_forge_c3_common_grid/rehearsal.md").read_text()
    # Exercise the actual operator shell gate, with no workers or launches.
    start = runbook.index("if \"$PY\" - <<'PY'")
    gate = runbook[start : runbook.index("```", start)]
    for outcome in ("false", "true"):
        script = (
            f"PY=/bin/{outcome}\nFG=fixture_forge\nC3=fixture_c3\n"
            'W() { :; }\nL() { echo LAUNCH "$@"; }\n' + gate
        )
        checked = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
        if outcome == "false":
            assert checked.returncode != 0 and "LAUNCH" not in checked.stdout
        else:
            assert checked.returncode == 0
            assert checked.stdout.splitlines() == [
                "LAUNCH forge fixture_forge --only-dead-workers",
                "LAUNCH c3 fixture_c3",
            ]


def test_rehearsal_watchdog_floor_allows_startup_and_still_detects_injected_stall(
    tmp_path, tiny_source
):
    from seis_interp.training.run_recovery import ProgressWatchdog

    out = tmp_path / "generated"
    make_study067_rehearsal(tiny_source, out)
    plan = yaml.safe_load((out / "forge/rehearsal.yaml").read_text())
    watchdog = plan["watchdog"]
    dog = ProgressWatchdog(
        tmp_path, watchdog["expected_report_seconds"], floor_seconds=watchdog["floor_seconds"]
    )
    assert dog.threshold == 900
    stall = next(
        fault
        for fault in plan["faults"]
        if fault["method"] == "regsi_real" and fault["action"] == "sleep"
    )
    assert dog.threshold < stall["seconds"]
