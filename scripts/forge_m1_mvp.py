"""Prepare or execute the frozen FORGE M1 MVP."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.pipelines.preflight_forge_mvp import preflight_forge_method, preflight_forge_mvp
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.run_forge_mvp import run_forge_method, run_forge_mvp
from seis_interp.processing.forge_mvp_contract import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "method", "check-method"))
    parser.add_argument(
        "--study", type=Path, help="Preparation study directory; defaults to study_052"
    )
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="run: start all six workers together; runtimes are then not comparable",
    )
    parser.add_argument(
        "--validation-holdout",
        action="store_true",
        help="prepare: hold out the study's validation traces from the outer observed set",
    )
    args = parser.parse_args()
    if args.concurrent and args.action != "run":
        parser.error("--concurrent is only valid for run")
    if args.validation_holdout and args.action != "prepare":
        parser.error("--validation-holdout is only valid for prepare")
    repo = Path(__file__).resolve().parents[1]
    if args.action == "prepare":
        if args.preparation or args.output or args.method or args.preflight:
            parser.error("prepare uses the fixed study configuration only")
        print(
            prepare_forge_mvp(
                repo,
                (args.study or repo / "studies/study_052_forge_m1_mvp").resolve(),
                validation_holdout=args.validation_holdout,
            )
        )
        return
    if args.study is not None:
        parser.error("--study is only valid for prepare")
    if args.preparation is None:
        parser.error("--preparation is required")
    preparation = args.preparation.resolve()
    if args.action in ("method", "check-method"):
        if args.output is None or args.method is None:
            parser.error("method requires --output and --method")
        function = run_forge_method if args.action == "method" else preflight_forge_method
        function(repo, preparation, args.output.resolve(), args.method)
    else:
        if args.method:
            parser.error("run executes the fixed six-method matrix")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        study_id = json.loads((preparation / "preparation_manifest.json").read_text())["study_id"]
        output = args.output or repo / "runs" / study_id / (
            stamp + "_" + execution_identity(repo)["git_commit"][:12] + "_" + args.action
        )
        if args.action == "preflight":
            result = preflight_forge_mvp(repo, preparation, output.resolve())
        else:
            if args.preflight is None:
                parser.error("run requires --preflight from the passing resource checks")
            result = run_forge_mvp(
                repo,
                preparation,
                output.resolve(),
                args.preflight.resolve(),
                concurrent=args.concurrent,
            )
        print(output)
        if result["status"] not in ("accepted", "passed"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
