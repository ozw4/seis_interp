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


def test_train_parser_exposes_the_five_commands(capsys) -> None:
    help_text = _help_text(["train"], capsys)

    for name in ("siren", "neighbor-inpainter", "shot-gather-inpainter", "trace-graph", "ccnet5d"):
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
    ["siren", "neighbor-inpainter", "shot-gather-inpainter", "trace-graph", "ccnet5d"],
)
def test_train_commands_offer_no_overwrite_option(command: str, capsys) -> None:
    help_text = _help_text(["train", command], capsys)

    assert "--overwrite" not in help_text
    for option in ("--config", "--interim", "--processed", "--output", "--device", "--json"):
        assert option in help_text


def test_public_python_api_is_importable_from_cli() -> None:
    assert callable(build_parser)
    assert callable(main)


def test_ccnet5d_help_distinguishes_partition_training_and_poc_end_to_end_run(capsys) -> None:
    training = _help_text(["train", "ccnet5d"], capsys)
    inference = _help_text(["interpolate", "ccnet5d"], capsys)

    for option in ("--config", "--interim", "--processed", "--output", "--device", "--json"):
        assert option in training and option in inference
    assert "--checkpoint" not in training and "--checkpoint" not in inference
    for option in ("--mask", "--case", "--volume"):
        assert option not in training
        assert option in inference
    for option in ("--model", "--patch-shape", "--learning-rate", "--max-epochs", "--overwrite"):
        assert option not in training and option not in inference


@pytest.mark.parametrize("command", ["train", "interpolate"])
def test_ccnet5d_help_does_not_import_torch_or_pipelines(command: str) -> None:
    probe = (
        "import sys\n"
        "from seis_interp.cli import main\n"
        "try:\n"
        f"    main([{command!r}, 'ccnet5d', '--help'])\n"
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
