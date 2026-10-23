"""Generate an isolated study_067 rehearsal; never prepare or launch a run."""

import argparse
import json
from pathlib import Path

from seis_interp.training.study067_rehearsal_generation import make_study067_rehearsal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "studies/study_067_forge_c3_common_grid",
    )
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--no-faults", action="store_true")
    variants.add_argument("--imports-only", action="store_true")
    variants.add_argument(
        "--determinism", action="store_true", help="fault-free CCNet benchmark-off twins"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            make_study067_rehearsal(
                args.source,
                args.out,
                no_faults=args.no_faults,
                imports_only=args.imports_only,
                determinism=args.determinism,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
