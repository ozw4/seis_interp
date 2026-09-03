"""Parser assembly and dispatch for the ``seis-interp`` command line."""

from __future__ import annotations

import argparse

from seis_interp.commands.data import add_data_commands

# The redundant alias keeps `seis_interp.cli.collect_environment` importable for
# existing callers after the doctor implementation moved to `commands.doctor`.
from seis_interp.commands.doctor import (
    add_doctor_command,
)
from seis_interp.commands.doctor import (
    collect_environment as collect_environment,
)
from seis_interp.commands.train import add_train_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seis-interp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_doctor_command(subparsers)
    add_data_commands(subparsers)
    add_train_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
