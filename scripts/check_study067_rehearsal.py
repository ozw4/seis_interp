"""Check every rehearsal acceptance area using structural evidence only."""

import argparse
import json
from pathlib import Path

from seis_interp.evaluation.study067_rehearsal_checks import (
    capture_drill_snapshot,
    check_rehearsal_plan,
    check_study067_rehearsal,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3", type=Path)
    parser.add_argument("--forge", type=Path)
    parser.add_argument("--gpu-log", type=Path)
    parser.add_argument("--twin", type=Path, action="append", default=[])
    parser.add_argument("--determinism-twin", type=Path, action="append", default=[])
    parser.add_argument("--imports", type=Path, action="append", default=[])
    parser.add_argument("--launch-commit")
    parser.add_argument(
        "--plan-only",
        type=Path,
        metavar="STUDY",
        help="CPU dry run: validate generated plans without claiming runtime acceptance",
    )
    parser.add_argument("--snapshot", type=Path, metavar="EXPERIMENT")
    parser.add_argument(
        "--label", choices=("before_refusal", "after_refusal", "before_relaunch", "after_relaunch")
    )
    args = parser.parse_args()
    if args.snapshot:
        if not args.label:
            parser.error("--snapshot requires --label")
        print(capture_drill_snapshot(args.snapshot.resolve(), args.label))
        return
    if args.plan_only:
        try:
            paths = check_rehearsal_plan(args.plan_only.resolve())
        except (OSError, ValueError, KeyError) as error:
            print(f"FAIL generated plan: {error}")
            raise SystemExit(1) from error
        print(
            "PASS generated plan (runtime acceptance not evaluated): " + ", ".join(map(str, paths))
        )
        return
    if not all((args.c3, args.forge, args.gpu_log)):
        parser.error("runtime checks require --c3, --forge and --gpu-log")
    results = check_study067_rehearsal(
        args.c3,
        args.forge,
        args.gpu_log,
        twins=args.twin,
        determinism_twins=args.determinism_twin,
        imports=args.imports,
        launch_commit=args.launch_commit,
    )
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['area']}: {json.dumps(result, sort_keys=True)}")
    raise SystemExit(0 if all(result["passed"] for result in results) else 1)


if __name__ == "__main__":
    main()
