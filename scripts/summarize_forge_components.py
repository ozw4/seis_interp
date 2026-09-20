"""Inspect component evidence from a completed FORGE waveform run."""

import argparse
from pathlib import Path

from seis_interp.pipelines.audit_forge_components import run_forge_component_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform_run", type=Path)
    args = parser.parse_args()
    run_forge_component_audit(Path(__file__).resolve().parents[1], args.waveform_run)


if __name__ == "__main__":
    main()
