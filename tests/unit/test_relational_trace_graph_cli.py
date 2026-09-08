"""Argument contracts and lazy imports for the new trace graph commands."""

from __future__ import annotations

import subprocess
import sys

import pytest

from seis_interp.cli import build_parser


@pytest.mark.parametrize("command", ["train", "interpolate"])
def test_help_imports_no_torch_or_pipeline(command):
    probe = (
        "import sys\nfrom seis_interp.cli import main\n"
        "try:\n"
        f"    main([{command!r}, 'relational-trace-graph', '--help'])\n"
        "except SystemExit as error:\n    assert error.code == 0\n"
        "assert 'torch' not in sys.modules\n"
        "assert not any(name.startswith('seis_interp.pipelines.') for name in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], check=False, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    for option in ("--config", "--interim", "--processed", "--output", "--device", "--json"):
        assert option in result.stdout
    if command == "train":
        for option in (
            "--validation-mask",
            "--validation-case",
            "--validation-volume",
            "--train-mask",
            "--train-case",
            "--train-volume",
        ):
            assert option in result.stdout
    else:
        assert "--checkpoint" in result.stdout and "--volume" in result.stdout


@pytest.mark.parametrize(
    "command,missing",
    [("train", "validation-case"), ("train", "validation-mask"), ("interpolate", "checkpoint")],
)
def test_required_arguments_are_checked_by_parser(command, missing):
    options = ("config", "interim", "processed", "output")
    options += (
        ("validation-mask", "validation-case")
        if command == "train"
        else ("checkpoint", "mask", "case")
    )
    argv = [command, "relational-trace-graph"]
    for option in options:
        if option != missing:
            argv += [f"--{option}", "unused"]
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(argv)
    assert error.value.code == 2
