from __future__ import annotations

import pytest

from seis_interp.cli import build_parser, main


def _help_text(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([*argv, "--help"])

    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_top_level_parser_routes_doctor_data_and_train(capsys) -> None:
    help_text = _help_text([], capsys)

    for name in ("doctor", "data", "train"):
        assert name in help_text


def test_train_parser_exposes_the_four_commands(capsys) -> None:
    help_text = _help_text(["train"], capsys)

    for name in ("siren", "neighbor-inpainter", "shot-gather-inpainter", "trace-graph"):
        assert name in help_text


@pytest.mark.parametrize(
    "command",
    ["siren", "neighbor-inpainter", "shot-gather-inpainter", "trace-graph"],
)
def test_train_commands_offer_no_overwrite_option(command: str, capsys) -> None:
    help_text = _help_text(["train", command], capsys)

    assert "--overwrite" not in help_text
    for option in ("--config", "--interim", "--processed", "--output", "--device", "--json"):
        assert option in help_text


def test_public_python_api_is_importable_from_cli() -> None:
    assert callable(build_parser)
    assert callable(main)
