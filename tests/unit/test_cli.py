from __future__ import annotations

import subprocess
import sys

import pytest

from seis_interp.cli import build_parser, main


def _help_text(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([*argv, "--help"])

    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_top_level_parser_routes_doctor_data_train_and_interpolate(capsys) -> None:
    help_text = _help_text([], capsys)

    for name in ("doctor", "data", "train", "interpolate"):
        assert name in help_text


def test_train_parser_exposes_the_four_commands(capsys) -> None:
    help_text = _help_text(["train"], capsys)
    assert "ccnet5d" not in help_text

    for name in (
        "siren",
        "neighbor-inpainter",
        "shot-gather-inpainter",
        "trace-graph",
    ):
        assert name in help_text


def test_train_and_interpolate_siren_keep_distinct_input_contracts(capsys) -> None:
    train_help = _help_text(["train", "siren"], capsys)
    interpolate_help = _help_text(["interpolate", "siren"], capsys)

    for option in ("--config", "--interim", "--processed", "--output", "--device", "--json"):
        assert option in train_help
        assert option in interpolate_help
    for option in ("--mask", "--case", "--volume"):
        assert option not in train_help
        assert option in interpolate_help


@pytest.mark.parametrize(
    "command",
    [
        "siren",
        "neighbor-inpainter",
        "shot-gather-inpainter",
        "trace-graph",
    ],
)
def test_train_commands_offer_no_overwrite_option(command: str, capsys) -> None:
    help_text = _help_text(["train", command], capsys)

    assert "--overwrite" not in help_text
    for option in ("--config", "--interim", "--processed", "--output", "--device", "--json"):
        assert option in help_text


def test_public_python_api_is_importable_from_cli() -> None:
    assert callable(build_parser)
    assert callable(main)


def test_poc_check_parser_and_dispatch(tmp_path, monkeypatch, capsys):
    import json

    from seis_interp.pipelines import check_c3_poc

    paths = {name: tmp_path / name for name in ("interim", "processed", "mask", "case", "volume")}
    received = {}
    summary = {
        "case_id": "case",
        "shape": [4, 1, 1, 1, 5],
        "observed_trace_count": 1,
        "target_trace_count": 4,
        "observed_global_rms": 2.0,
    }

    def check(**kwargs):
        received.update(kwargs)
        return summary

    monkeypatch.setattr(check_c3_poc, "check_c3_poc_inputs", check)
    arguments = ["poc", "check"]
    for name, path in paths.items():
        arguments.extend([f"--{name}", str(path)])
    for name in paths:
        index = arguments.index(f"--{name}")
        with pytest.raises(SystemExit) as error:
            build_parser().parse_args(arguments[:index] + arguments[index + 2 :])
        assert error.value.code == 2
    assert main([*arguments, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == summary
    assert received == {f"{name}_dir": path for name, path in paths.items()}
    assert list(tmp_path.iterdir()) == []


def test_ccnet5d_help_exposes_only_poc_end_to_end_run(capsys) -> None:
    inference = _help_text(["interpolate", "ccnet5d"], capsys)

    for option in ("--config", "--interim", "--processed", "--output", "--device", "--json"):
        assert option in inference
    assert "--checkpoint" not in inference
    for option in ("--mask", "--case", "--volume"):
        assert option in inference
    for option in ("--model", "--patch-shape", "--learning-rate", "--max-epochs", "--overwrite"):
        assert option not in inference


def test_ccnet5d_help_does_not_import_torch_or_pipelines() -> None:
    probe = (
        "import sys\n"
        "from seis_interp.cli import main\n"
        "try:\n"
        "    main(['interpolate', 'ccnet5d', '--help'])\n"
        "except SystemExit as error:\n"
        "    assert error.code == 0\n"
        "else:\n"
        "    raise AssertionError('help must exit')\n"
        "assert 'torch' not in sys.modules\n"
        "assert not any(name.startswith('seis_interp.pipelines.') for name in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], check=False, capture_output=True, text=True, timeout=60
    )

    assert completed.returncode == 0, completed.stderr
    assert "ccnet5d" in completed.stdout
