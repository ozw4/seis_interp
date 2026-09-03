from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from seis_interp import run_records


@dataclass(frozen=True)
class FakeDevice:
    type: str


def test_check_new_output_directory_rejects_existing_path(tmp_path: Path) -> None:
    existing = tmp_path / "run"
    existing.mkdir()

    expected_message = re.escape(f"run output path already exists: {existing}")
    with pytest.raises(FileExistsError, match=expected_message):
        run_records.check_new_output_directory(existing)


def test_check_new_output_directory_does_not_create_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "new-run"

    run_records.check_new_output_directory(missing)

    assert not missing.exists()


def test_write_run_outputs_writes_exactly_four_round_trippable_files(tmp_path: Path) -> None:
    output_directory = tmp_path / "run"
    config = {"training": {"device": "cpu"}, "model": {"name": "siren"}}
    inputs_lock = {"seed": 42, "files": {"traces.npy": {"sha256": "abc"}}}
    metrics = {"validation_snr_db": 12.5}
    run_metadata = {"git_commit": "deadbeef", "started_at_utc": "2026-01-01T00:00:00Z"}

    run_records.write_run_outputs(output_directory, config, inputs_lock, metrics, run_metadata)

    written = sorted(path.name for path in output_directory.iterdir())
    assert written == sorted(
        [
            run_records.CONFIG_FILE_NAME,
            run_records.INPUTS_LOCK_FILE_NAME,
            run_records.METRICS_FILE_NAME,
            run_records.RUN_FILE_NAME,
        ]
    )
    config_text = (output_directory / run_records.CONFIG_FILE_NAME).read_text(encoding="utf-8")
    assert yaml.safe_load(config_text) == config
    assert list(yaml.safe_load(config_text)) == list(config)
    assert config_text == yaml.safe_dump(dict(config), sort_keys=False)
    for file_name, payload in [
        (run_records.INPUTS_LOCK_FILE_NAME, inputs_lock),
        (run_records.METRICS_FILE_NAME, metrics),
        (run_records.RUN_FILE_NAME, run_metadata),
    ]:
        text = (output_directory / file_name).read_text(encoding="utf-8")
        assert json.loads(text) == payload
        assert text.endswith("\n")
        assert not text.endswith("\n\n")
        assert text == json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


@pytest.mark.parametrize("non_finite_argument", ["inputs_lock", "metrics", "run_metadata"])
def test_write_run_outputs_rejects_non_finite_json_values(
    tmp_path: Path, non_finite_argument: str
) -> None:
    payloads: dict[str, dict[str, object]] = {
        "inputs_lock": {},
        "metrics": {},
        "run_metadata": {},
    }
    payloads[non_finite_argument] = {"value": math.nan}

    with pytest.raises(ValueError, match="Out of range float values"):
        run_records.write_run_outputs(
            tmp_path / "run",
            {},
            payloads["inputs_lock"],
            payloads["metrics"],
            payloads["run_metadata"],
        )


def test_file_hashes_returns_sha256_of_named_files(tmp_path: Path) -> None:
    payload = b"seis-interp"
    (tmp_path / "small.bin").write_bytes(payload)

    hashes = run_records.file_hashes(tmp_path, ("small.bin",))

    assert hashes == {"small.bin": {"sha256": hashlib.sha256(payload).hexdigest()}}


def test_current_git_commit_returns_stripped_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(run_records.subprocess, "run", fake_run)

    assert run_records.current_git_commit() == "abc123"


def test_current_git_commit_wraps_subprocess_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=128, cmd=["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(run_records.subprocess, "run", failing_run)

    with pytest.raises(RuntimeError, match="could not determine the current Git commit"):
        run_records.current_git_commit()


def test_current_git_commit_rejects_empty_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(run_records.subprocess, "run", empty_run)

    with pytest.raises(RuntimeError, match="git rev-parse HEAD returned an empty commit"):
        run_records.current_git_commit()


def test_utc_timestamp_uses_second_precision_utc_z_format() -> None:
    timestamp = run_records.utc_timestamp()

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp)


def test_runtime_resource_metadata_on_cpu_has_no_cuda_keys() -> None:
    metadata = run_records.runtime_resource_metadata(FakeDevice(type="cpu"))

    assert sorted(metadata) == [
        "cudnn_benchmark",
        "cudnn_deterministic",
        "process_max_rss_kib",
    ]
    assert metadata["cudnn_benchmark"] is False
    assert metadata["cudnn_deterministic"] is False
    assert isinstance(metadata["process_max_rss_kib"], int)
    assert metadata["process_max_rss_kib"] > 0
