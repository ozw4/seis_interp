"""Analyze immutable saved runs; never train, prepare, or write under runs/."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("studies/study_067_forge_c3_common_grid/evaluation.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("evaluate")
    analyze.add_argument("--c3", type=Path, required=True)
    analyze.add_argument("--forge", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    reproduce = commands.add_parser("reproduce-c3")
    reproduce.add_argument("--c3", type=Path, required=True)
    reproduce.add_argument("--reference-c3", type=Path, required=True)
    reproduce.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit-seeds")
    audit.add_argument("--exclude-experiment", type=Path, action="append", required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.command == "audit-seeds":
        from seis_interp.data.study067_analysis_inputs import require_output_directory
        from seis_interp.evaluation.study067_seed_audit import audit_seed_namespaces

        output = require_output_directory(repo, args.output)
        result = audit_seed_namespaces(repo, [repo / path for path in args.exclude_experiment])
        (output / "seed_namespace_audit.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            json.dumps(
                {
                    **{k: result[k] for k in ("passed", "scanned_file_count", "errors")},
                    "partition_seed_use_count": len(result["partition_seed_uses"]),
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(0 if result["passed"] else 1)
    from seis_interp.pipelines.evaluate_study067 import evaluate_study067, reproduce_c3

    evaluation = repo / args.evaluation
    if args.command == "evaluate":
        print(
            evaluate_study067(
                repo,
                evaluation,
                (repo / args.c3).resolve(),
                (repo / args.forge).resolve(),
                args.output,
            )
        )
    else:
        table = reproduce_c3(
            repo,
            evaluation,
            (repo / args.c3).resolve(),
            (repo / args.reference_c3).resolve(),
            args.output,
        )
        print(
            table.loc[
                table.metric.eq("snr_db"),
                ["estimate", "ci95_low", "ci95_high", "cluster_count", "bootstrap_resamples"],
            ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )


if __name__ == "__main__":
    main()
