"""Run the FORGE header audit with the checked-in study conditions."""

import argparse
from pathlib import Path

from seis_interp.pipelines.audit_forge_headers import run_forge_header_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("studies/study_047_forge_header_audit"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_forge_header_audit(repo, repo / args.study)


if __name__ == "__main__":
    main()
