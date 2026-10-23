"""Work-driven heartbeats preserve stall detection and training cadence."""

import io
import json

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces
from seis_interp.processing.linear_moveout import align_moveout, unalign_moveout
from seis_interp.training import phase_progress, run_recovery
from seis_interp.training.exact_resume import atomic_torch_save
from seis_interp.training.phase_progress import PhaseProgress, ProgressHeartbeat, ProgressWriter


@pytest.mark.parametrize("heartbeats", [False, True])
def test_long_prediction_survives_only_while_work_emits_heartbeats(
    tmp_path, monkeypatch, heartbeats
):
    now = 0.0
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: now)
    exits, reports = [], []
    dog = run_recovery.ProgressWatchdog(
        tmp_path, 0.01, floor_seconds=0.09, exit_process=exits.append
    )

    def report(message):
        reports.append((now, message))
        dog.report(message)

    progress = PhaseProgress(report if heartbeats else None, seconds=0.03)
    output = []

    def advance_prediction(_):
        nonlocal now
        now += 0.01
        output.append(len(output) ** 2)
        progress(f"predicting {len(output)}/100 cells (primary)")
        return len(output) == 100

    monkeypatch.setattr(dog.stopped, "wait", advance_prediction)
    dog._watch()
    if heartbeats:
        assert len(output) == 100 and exits == []
        assert all(b[0] - a[0] >= 0.03 for a, b in zip(reports, reports[1:], strict=False))
        assert not dog.intervals
        # Once prediction stops making progress, the same watchdog must expire.
        now += 0.1
        monkeypatch.setattr(dog.stopped, "wait", lambda _: False)
        dog._watch()
    assert exits == [75]
    assert json.loads((tmp_path / "watchdog.json").read_text())["reason"] == "progress stalled"


def test_heartbeats_cannot_replace_slow_training_intervals_or_sample_a_partial_interval(
    tmp_path, monkeypatch
):
    now = 0.0
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: now)
    dog = run_recovery.ProgressWatchdog(tmp_path, 1, floor_seconds=1)
    for _ in range(10):
        now += 400
        dog.report("training step")
    before = list(dog.intervals)
    for _ in range(600):
        now += 30
        dog.report(ProgressHeartbeat("prediction warm-up"))
        assert dog.last == now and dog.threshold == 1200
    now += 10
    dog.report("first training report after warm-up")
    assert list(dog.intervals) == before
    now += 400
    dog.report("next training report")
    assert list(dog.intervals) == [*before, 400] and dog.threshold == 1200


def test_reports_are_throttled_by_monotonic_time(monkeypatch):
    now = 0.0
    monkeypatch.setattr(phase_progress.time, "monotonic", lambda: now)
    reports = []
    progress = PhaseProgress(reports.append)
    for _ in range(1000):
        progress("predicting")
    assert not reports
    now = 30
    progress("predicting 1200/4800 cells (ema 0.9998)")
    progress("predicting again")
    assert reports == ["predicting 1200/4800 cells (ema 0.9998)"]
    assert isinstance(reports[0], ProgressHeartbeat)


def test_checkpoint_and_numpy_write_bytes_are_unchanged(tmp_path, monkeypatch):
    now = 0.0

    def monotonic():
        nonlocal now
        now += 31
        return now

    monkeypatch.setattr(phase_progress.time, "monotonic", monotonic)
    reports = []
    payload = {"phase": "before_prediction", "weight": torch.arange(40)}
    expected = io.BytesIO()
    torch.save(payload, expected)
    path = tmp_path / "resume.pt"
    atomic_torch_save(payload, path, reporter=reports.append)
    assert path.read_bytes() == expected.getvalue()
    assert any("saving checkpoint" in message for message in reports)
    values = np.arange(40, dtype=np.float32).reshape(4, 10)
    expected, actual = io.BytesIO(), io.BytesIO()
    np.save(expected, values, allow_pickle=False)
    np.save(
        ProgressWriter(actual, PhaseProgress(reports.append), "saving prediction"),
        values,
        allow_pickle=False,
    )
    assert actual.getvalue() == expected.getvalue()


def test_alignment_progress_preserves_sample_bytes_and_order():
    values = np.arange(120, dtype=np.float32).reshape(12, 10)
    delays = np.arange(10, dtype=np.int64)
    reports = []
    expected = align_moveout(values, delays)
    actual = align_moveout(values, delays, progress=reports.append)
    assert actual.tobytes() == expected.tobytes()
    restored = unalign_moveout(actual, delays, len(values), progress=reports.append)
    assert restored.tobytes() == values.tobytes()
    assert any("aligning" in message for message in reports)
    assert any("restoring alignment" in message for message in reports)


def test_scoring_progress_preserves_metrics_and_trace_order():
    target = np.arange(2600, dtype=np.float32).reshape(260, 10) + 1
    prediction = target * np.float32(0.75)
    rows = pd.DataFrame({"cell_id": np.arange(260), "clean_target": True})
    expected, expected_traces = evaluate_traces(prediction, target, rows)
    reports = []
    actual, actual_traces = evaluate_traces(prediction, target, rows, progress=reports.append)
    assert actual == expected
    pd.testing.assert_frame_equal(actual_traces, expected_traces, check_exact=True)
    assert reports[-1] == "scoring 260/260 cells"
