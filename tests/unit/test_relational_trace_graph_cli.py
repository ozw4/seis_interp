"""Argument contracts and lazy imports for the observed-only trace graph command."""

from __future__ import annotations

import subprocess
import sys

import pytest

from seis_interp.cli import build_parser


def test_help_imports_no_torch_or_pipeline():
    command = "interpolate"
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
    assert "--checkpoint" not in result.stdout
    assert "--volume" in result.stdout


@pytest.mark.parametrize("missing", ["volume", "mask", "case", "config"])
def test_required_arguments_are_checked_by_parser(missing):
    options = ("config", "interim", "processed", "output", "volume", "mask", "case")
    argv = ["interpolate", "relational-trace-graph"]
    for option in options:
        if option != missing:
            argv += [f"--{option}", "unused"]
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(argv)
    assert error.value.code == 2
