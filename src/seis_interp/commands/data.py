"""The ``data`` commands: acquire, verify, inspect, and prepare external datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.error import URLError

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    ConfigurationError,
    get_required_config_value,
    load_resolved_config,
    repository_relative_config_source,
)
from seis_interp.data.data_root import DataRootError, resolve_data_root
from seis_interp.data.seg_c3_na import (
    DATASET_ID,
    DataIntegrityError,
    ManifestError,
    default_manifest_path,
    download_seg_c3_na,
    load_manifest,
    verified_source_sha256,
    verify_seg_c3_na,
)
from seis_interp.data.seg_c3_na_inspection import (
    DEFAULT_SAMPLE_TRACE_COUNT,
    SegyInspectionError,
    format_seg_c3_na_inspection,
    inspect_seg_c3_na,
    seg_c3_na_inspection_ok,
    seg_c3_na_inspection_to_dict,
)


def _download_data(args: argparse.Namespace) -> int:
    try:
        lock_path = download_seg_c3_na(
            args.manifest,
            args.data_root,
            force=args.force,
            resume=not args.no_resume,
            timeout_s=args.timeout_s,
        )
    except (DataRootError, DataIntegrityError, ManifestError, OSError, URLError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Download lock written to: {lock_path}")
    return 0


def _verify_data(args: argparse.Namespace) -> int:
    try:
        results = verify_seg_c3_na(args.manifest, args.data_root)
    except (DataRootError, DataIntegrityError, ManifestError, OSError, ValueError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(f"{result.status.upper():>17}  {result.name}: {result.detail}")
    return 0 if all(result.ok for result in results) else 1


def _inspect_data(args: argparse.Namespace) -> int:
    try:
        inspection = inspect_seg_c3_na(
            args.manifest,
            args.data_root,
            sample_trace_count=args.sample_traces,
        )
    except (DataRootError, ManifestError, SegyInspectionError, OSError, ValueError) as exc:
        print(f"Inspection failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(seg_c3_na_inspection_to_dict(inspection), indent=2, sort_keys=True))
    else:
        print(format_seg_c3_na_inspection(inspection))
    return 0 if seg_c3_na_inspection_ok(inspection) else 1


def _prepare_c3_shot(args: argparse.Namespace) -> int:
    # Imported here so that `doctor` keeps working without the data and segy extras.
    from seis_interp.pipelines.prepare_c3 import (
        C3_COMPLETE_SHOT_TRACE_COUNT,
        prepare_c3_complete_shot,
    )

    expected_traces = args.expected_traces
    if expected_traces is None:
        expected_traces = C3_COMPLETE_SHOT_TRACE_COUNT

    try:
        summary = prepare_c3_complete_shot(
            input_path=args.input,
            output_dir=args.output,
            ffid=args.ffid,
            expected_trace_count=expected_traces,
            dataset_id=args.dataset_id,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"data prepare-c3-shot failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Source file: {args.input}")
        print(f"Selected FFID: {summary['selection']['ffid']}")
        print(f"Traces: {summary['trace_count']}")
        print(f"Samples per trace: {summary['sample_count']}")
        print(f"Sample interval: {summary['sample_interval_s']} s")
        print(f"Output directory: {args.output}")
    return 0


def _prepare_c3_survey(args: argparse.Namespace) -> int:
    # Imported here so that `doctor` keeps working without the data and segy extras.
    from seis_interp.pipelines.prepare_c3 import (
        C3_COMPLETE_SHOT_TRACE_COUNT,
        C3_SURVEY_FFID_RANGE,
        prepare_c3_survey,
    )

    try:
        manifest = load_manifest(args.manifest)
        verification = verify_seg_c3_na(args.manifest, args.data_root)
        failed = [result for result in verification if not result.ok]
        if failed:
            details = "; ".join(
                f"{result.name}: {result.status} ({result.detail})" for result in failed
            )
            raise DataIntegrityError(f"source verification failed: {details}")

        data_root = resolve_data_root(args.data_root)
        source_directory = data_root / "external" / DATASET_ID
        input_paths = [source_directory / file_spec.name for file_spec in manifest.files]
        source_sha256 = verified_source_sha256(manifest, input_paths, verification)

        missing_ranges = [
            file_spec.name
            for file_spec in manifest.files
            if file_spec.ffid_min is None or file_spec.ffid_max is None
        ]
        if missing_ranges:
            raise ManifestError(f"manifest files are missing FFID ranges: {missing_ranges}")
        expected_ranges = [
            (int(file_spec.ffid_min), int(file_spec.ffid_max)) for file_spec in manifest.files
        ]
        summary = prepare_c3_survey(
            input_paths=input_paths,
            output_dir=args.output,
            dataset_id=args.dataset_id,
            expected_complete_trace_count=C3_COMPLETE_SHOT_TRACE_COUNT,
            expected_ffid_ranges=expected_ranges,
            expected_survey_ffid_range=C3_SURVEY_FFID_RANGE,
            source_sha256=source_sha256,
            overwrite=args.overwrite,
        )
    except (
        DataIntegrityError,
        DataRootError,
        FileNotFoundError,
        FileExistsError,
        ManifestError,
        OSError,
        ValueError,
    ) as error:
        print(f"data prepare-c3-survey failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        ffids = summary["ffids"]
        print(f"Source files: {summary['source_file_count']}")
        print(f"FFID range: {ffids[0]}-{ffids[-1]}")
        print(f"FFIDs: {summary['ffid_count']}")
        print(f"Complete FFIDs: {summary['complete_ffid_count']}")
        print(f"Incomplete FFIDs: {summary['incomplete_ffid_count']}")
        print(f"Traces: {summary['trace_count']}")
        print(f"Samples per trace: {summary['sample_count']}")
        print(f"Output directory: {args.output}")
    return 0


def _prepare_baseline(args: argparse.Namespace) -> int:
    # Imported here so that unrelated commands keep working without the data extras.
    from seis_interp.pipelines.prepare_baseline import (
        AMPLITUDE_NORMALIZATION_METHOD,
        COORDINATE_NORMALIZATION_METHOD,
        prepare_baseline_dataset,
    )
    from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig

    try:
        config = load_resolved_config(args.config, repository_root=REPOSITORY_ROOT)
        study_config = config.get("study")
        if isinstance(study_config, Mapping) and "random_seed" in study_config:
            raise ConfigurationError("study.random_seed is not supported; use project.random_seed")

        random_seed = _config_value_or_override(
            args.random_seed,
            config,
            "project.random_seed",
        )
        configured_split_scope = _config_value_or_optional_default(
            None,
            config,
            "sampling.split_scope",
            "global",
        )
        split_scope = args.split_scope or configured_split_scope
        split_unit_changes = (split_scope == "whole_ffid") != (
            configured_split_scope == "whole_ffid"
        )
        if split_unit_changes and args.holdout_fraction is None:
            raise ConfigurationError(
                "--holdout-fraction is required when --split-scope changes the configured "
                "split unit"
            )
        holdout_config_path = (
            "sampling.random_ffid_holdout_fraction"
            if split_scope == "whole_ffid"
            else "sampling.random_trace_holdout_fraction"
        )
        holdout_fraction = _config_value_or_override(
            args.holdout_fraction,
            config,
            holdout_config_path,
        )
        validation_fraction = _config_value_or_override(
            args.validation_fraction_of_holdout,
            config,
            "sampling.validation_fraction_of_holdout",
        )
        coordinate_normalization = _required_supported_config_value(
            config,
            "normalization.coordinates",
            COORDINATE_NORMALIZATION_METHOD,
        )
        amplitude_normalization = _required_supported_config_value(
            config,
            "normalization.amplitude",
            AMPLITUDE_NORMALIZATION_METHOD,
        )
        sampling_config = config.get("sampling")
        raw_trace_filter = (
            sampling_config.get("trace_amplitude_filter")
            if isinstance(sampling_config, Mapping)
            else None
        )
        trace_amplitude_filter = (
            TraceAmplitudeFilterConfig.from_mapping(
                raw_trace_filter,
                name="sampling.trace_amplitude_filter",
            )
            if raw_trace_filter is not None
            else None
        )
        config_source = repository_relative_config_source(
            args.config,
            repository_root=REPOSITORY_ROOT,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-baseline failed: {error}", file=sys.stderr)
        return 1

    try:
        summary = prepare_baseline_dataset(
            interim_dir=args.input,
            output_dir=args.output,
            holdout_fraction=holdout_fraction,
            validation_fraction_of_holdout=validation_fraction,
            random_seed=random_seed,
            split_scope=split_scope,
            coordinate_normalization=coordinate_normalization,
            amplitude_normalization=amplitude_normalization,
            trace_amplitude_filter=trace_amplitude_filter,
            config_source=config_source,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"data prepare-baseline failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        split_counts = summary["split_counts"]
        print(f"Configuration: {summary['config_source']}")
        print(f"Input dataset: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Traces: {summary['trace_count']}")
        trace_quality = summary.get("trace_quality")
        if isinstance(trace_quality, Mapping):
            print(f"Eligible traces: {trace_quality['eligible_trace_count']}")
            print(f"Excluded traces: {trace_quality['excluded_trace_count']}")
        print(f"Split scope: {summary.get('split_scope', 'global')}")
        print(f"FFIDs: {summary.get('ffid_count', 1)}")
        print(f"Train traces: {split_counts['train']}")
        print(f"Validation traces: {split_counts['validation']}")
        print(f"Test traces: {split_counts['test']}")
    return 0


def _prepare_mask(args: argparse.Namespace) -> int:
    # Imported here so that unrelated commands keep working without the data extras.
    from seis_interp.config_values import finite_float, nonnegative_integer
    from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
    from seis_interp.processing.interpolation_masks import MASK_KINDS
    from seis_interp.processing.trace_splits import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT

    try:
        config = load_resolved_config(args.config, repository_root=REPOSITORY_ROOT)
        study_config = config.get("study")
        if isinstance(study_config, Mapping) and "random_seed" in study_config:
            raise ConfigurationError("study.random_seed is not supported; use project.random_seed")

        random_seed = nonnegative_integer(
            get_required_config_value(config, "project.random_seed"),
            "project.random_seed",
        )
        partition = get_required_config_value(config, "interpolation_mask.partition")
        allowed_partitions = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
        if not isinstance(partition, str) or partition not in allowed_partitions:
            raise ConfigurationError(
                "interpolation_mask.partition must be one of "
                f"{list(allowed_partitions)}, got {partition!r}"
            )

        kind = get_required_config_value(config, "interpolation_mask.kind")
        if not isinstance(kind, str) or kind not in MASK_KINDS:
            raise ConfigurationError(
                f"interpolation_mask.kind must be one of {list(MASK_KINDS)}, got {kind!r}"
            )

        missing_fraction = finite_float(
            get_required_config_value(config, "interpolation_mask.missing_fraction"),
            "interpolation_mask.missing_fraction",
        )
        if not 0.0 < missing_fraction < 1.0:
            raise ConfigurationError(
                "interpolation_mask.missing_fraction must be strictly between 0 and 1"
            )
        config_source = repository_relative_config_source(
            args.config,
            repository_root=REPOSITORY_ROOT,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-mask failed: {error}", file=sys.stderr)
        return 1

    try:
        summary = prepare_interpolation_mask(
            interim_dir=args.input,
            processed_dir=args.processed,
            output_dir=args.output,
            partition=partition,
            kind=kind,
            missing_fraction=missing_fraction,
            random_seed=random_seed,
            config_source=config_source,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-mask failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        counts = summary["counts"]
        print(f"Configuration: {summary['config_source']}")
        print(f"Input dataset: {args.input}")
        print(f"Dataset partition: {args.processed}")
        print(f"Output directory: {args.output}")
        print(f"Mask kind: {summary['kind']}")
        print(f"Partition: {summary['partition']}")
        print(f"Candidate traces: {summary['candidate_trace_count']}")
        print(f"Observed traces: {counts['observed']}")
        print(f"Evaluation target traces: {counts['evaluation_target']}")
    return 0


def _prepare_benchmark_case(args: argparse.Namespace) -> int:
    # Imported here so that unrelated commands keep working without the data extras.
    from seis_interp.data.benchmark_case_store import validated_case_id
    from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case

    try:
        config = load_resolved_config(args.config, repository_root=REPOSITORY_ROOT)
        case_id = validated_case_id(get_required_config_value(config, "benchmark_case.id"))
        config_source = repository_relative_config_source(
            args.config,
            repository_root=REPOSITORY_ROOT,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-benchmark-case failed: {error}", file=sys.stderr)
        return 1

    try:
        summary = prepare_benchmark_case(
            interim_dir=args.input,
            processed_dir=args.processed,
            mask_dir=args.mask,
            output_dir=args.output,
            case_id=case_id,
            config_source=config_source,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-benchmark-case failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        mask = summary["mask"]
        counts = mask["counts"]
        print(f"Configuration: {summary['config_source']}")
        print(f"Case ID: {summary['case_id']}")
        print(f"Dataset ID: {summary['dataset_id']}")
        print(f"Partition: {summary['partition']}")
        print(f"Mask kind: {mask['kind']}")
        print(f"Observed traces: {counts['observed']}")
        print(f"Evaluation target traces: {counts['evaluation_target']}")
        print(f"Input dataset: {args.input}")
        print(f"Prepared partition: {args.processed}")
        print(f"Mask artifact: {args.mask}")
        print(f"Output directory: {args.output}")
    return 0


def _prepare_c3_volume_index(args: argparse.Namespace) -> int:
    # Imported here so that unrelated commands keep working without the data extras.
    from seis_interp.data.c3_volume_index_store import validated_volume_id
    from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index

    try:
        config = load_resolved_config(args.config, repository_root=REPOSITORY_ROOT)
        volume_id = validated_volume_id(get_required_config_value(config, "benchmark_volume.id"))
        ranges = {
            name: _required_index_range(
                config,
                f"benchmark_volume.selection.{name}",
            )
            for name in (
                "time",
                "source_line",
                "shot_in_line",
                "relative_receiver_x",
                "relative_receiver_y",
            )
        }
        config_source = repository_relative_config_source(
            args.config,
            repository_root=REPOSITORY_ROOT,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-c3-volume-index failed: {error}", file=sys.stderr)
        return 1

    try:
        summary = prepare_c3_volume_index(
            interim_dir=args.input,
            processed_dir=args.processed,
            mask_dir=args.mask,
            case_dir=args.case,
            output_dir=args.output,
            volume_id=volume_id,
            time_range=ranges["time"],
            source_line_range=ranges["source_line"],
            shot_in_line_range=ranges["shot_in_line"],
            relative_receiver_x_range=ranges["relative_receiver_x"],
            relative_receiver_y_range=ranges["relative_receiver_y"],
            config_source=config_source,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-c3-volume-index failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        binding = summary["benchmark_case"]
        counts = summary["role_counts"]
        print(f"Configuration: {summary['config_source']}")
        print(f"Volume ID: {summary['volume_id']}")
        print(f"Benchmark case ID: {binding['case_id']}")
        print(f"Dataset ID: {summary['dataset_id']}")
        print(f"Partition: {summary['partition']}")
        print(f"Axis order: {', '.join(summary['axis_order'])}")
        print(f"Selection: {summary['selection']}")
        print(f"Shape: {' x '.join(str(value) for value in summary['shape'])}")
        print(f"Traces: {summary['trace_count']}")
        print(f"Observed traces: {counts['observed']}")
        print(f"Evaluation target traces: {counts['evaluation_target']}")
        print(f"Input dataset: {args.input}")
        print(f"Prepared partition: {args.processed}")
        print(f"Mask artifact: {args.mask}")
        print(f"Benchmark case: {args.case}")
        print(f"Output directory: {args.output}")
    return 0


def _required_index_range(config: Mapping[str, object], dotted_path: str) -> tuple[int, int]:
    from seis_interp.processing.c3_volume_index import validated_index_range

    return validated_index_range(
        get_required_config_value(config, dotted_path),
        name=dotted_path,
    )


def _config_value_or_override(
    override: object | None,
    config: Mapping[str, object],
    dotted_path: str,
) -> object:
    if override is not None:
        return override
    return get_required_config_value(config, dotted_path)


def _config_value_or_optional_default(
    override: object | None,
    config: Mapping[str, object],
    dotted_path: str,
    default: object,
) -> object:
    """Resolve an optional config value below a CLI override."""
    if override is not None:
        return override
    current: object = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _required_supported_config_value(
    config: Mapping[str, object],
    dotted_path: str,
    supported_value: str,
) -> str:
    value = get_required_config_value(config, dotted_path)
    if value != supported_value:
        raise ConfigurationError(f"{dotted_path} must be {supported_value!r}, got {value!r}")
    return supported_value


def add_data_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    data = subparsers.add_parser(
        "data", help="Acquire, verify, inspect, and prepare external datasets."
    )
    data_commands = data.add_subparsers(dest="data_command", required=True)

    download = data_commands.add_parser("download", help="Download an external dataset.")
    download.add_argument("dataset", choices=(DATASET_ID,))
    download.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the tracked dataset manifest.",
    )
    download.add_argument(
        "--data-root",
        type=Path,
        help="Override SEIS_INTERP_DATA_ROOT for this command.",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Delete completed and partial files before downloading.",
    )
    download.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard partial files instead of issuing HTTP Range requests.",
    )
    download.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Per-request network timeout in seconds.",
    )
    download.set_defaults(handler=_download_data)

    verify = data_commands.add_parser("verify", help="Verify an external dataset.")
    verify.add_argument("dataset", choices=(DATASET_ID,))
    verify.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the tracked dataset manifest.",
    )
    verify.add_argument(
        "--data-root",
        type=Path,
        help="Override SEIS_INTERP_DATA_ROOT for this command.",
    )
    verify.set_defaults(handler=_verify_data)

    inspect = data_commands.add_parser("inspect", help="Inspect SEG-Y structure and content.")
    inspect.add_argument("dataset", choices=(DATASET_ID,))
    inspect.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the tracked dataset manifest.",
    )
    inspect.add_argument(
        "--data-root",
        type=Path,
        help="Override SEIS_INTERP_DATA_ROOT for this command.",
    )
    inspect.add_argument(
        "--sample-traces",
        type=int,
        default=DEFAULT_SAMPLE_TRACE_COUNT,
        help="Number of evenly spaced traces sampled per SEG-Y file.",
    )
    inspect.add_argument("--json", action="store_true", help="Print JSON output.")
    inspect.set_defaults(handler=_inspect_data)

    prepare = data_commands.add_parser(
        "prepare-c3-shot",
        help="Write one complete SEG C3 NA shot as an interim trace dataset.",
    )
    prepare.add_argument("--input", type=Path, required=True, help="Input SEG-Y file.")
    prepare.add_argument("--output", type=Path, required=True, help="Output directory.")
    prepare.add_argument(
        "--ffid",
        type=int,
        default=None,
        help="FFID to select. Defaults to the smallest complete FFID.",
    )
    prepare.add_argument(
        "--expected-traces",
        type=int,
        default=None,
        help="Trace count that marks an FFID as complete (default: 544, a complete C3 NA shot).",
    )
    prepare.add_argument(
        "--dataset-id",
        type=str,
        default=DATASET_ID,
        help="Dataset identifier stored in dataset.json.",
    )
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the generated files in an existing output directory.",
    )
    prepare.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare.set_defaults(handler=_prepare_c3_shot)

    prepare_survey = data_commands.add_parser(
        "prepare-c3-survey",
        help="Write every manifest-declared SEG C3 NA FFID as one interim dataset.",
    )
    prepare_survey.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the tracked dataset manifest.",
    )
    prepare_survey.add_argument(
        "--data-root",
        type=Path,
        help="Override SEIS_INTERP_DATA_ROOT for this command.",
    )
    prepare_survey.add_argument("--output", type=Path, required=True, help="Output directory.")
    prepare_survey.add_argument(
        "--dataset-id",
        type=str,
        default=DATASET_ID,
        help="Dataset identifier stored in dataset.json.",
    )
    prepare_survey.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated files in an existing output directory.",
    )
    prepare_survey.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare_survey.set_defaults(handler=_prepare_c3_survey)

    prepare_baseline = data_commands.add_parser(
        "prepare-baseline",
        help="Create trace splits and normalization metadata from an interim dataset.",
    )
    prepare_baseline.add_argument(
        "--input", type=Path, required=True, help="Input interim trace dataset directory."
    )
    prepare_baseline.add_argument(
        "--output", type=Path, required=True, help="Output processed dataset directory."
    )
    prepare_baseline.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Study configuration YAML to resolve, including its extends chain.",
    )
    prepare_baseline.add_argument(
        "--holdout-fraction",
        type=float,
        help=("Override the configured trace or FFID holdout fraction, according to split scope."),
    )
    prepare_baseline.add_argument(
        "--validation-fraction-of-holdout",
        type=float,
        help="Override sampling.validation_fraction_of_holdout.",
    )
    prepare_baseline.add_argument(
        "--random-seed",
        type=int,
        help="Override project.random_seed for deterministic trace-level splitting.",
    )
    prepare_baseline.add_argument(
        "--split-scope",
        choices=("global", "per_ffid", "whole_ffid"),
        help="Override sampling.split_scope (default: global).",
    )
    prepare_baseline.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files generated by this pipeline in an existing output directory.",
    )
    prepare_baseline.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare_baseline.set_defaults(handler=_prepare_baseline)

    prepare_mask = data_commands.add_parser(
        "prepare-mask",
        help="Create observed and evaluation target visibility within one dataset partition.",
    )
    prepare_mask.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Study configuration YAML containing interpolation_mask conditions.",
    )
    prepare_mask.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input interim trace dataset directory.",
    )
    prepare_mask.add_argument(
        "--processed",
        type=Path,
        required=True,
        help="Processed dataset partition directory created by prepare-baseline.",
    )
    prepare_mask.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output interpolation mask directory.",
    )
    prepare_mask.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files generated by this pipeline in an existing output directory.",
    )
    prepare_mask.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare_mask.set_defaults(handler=_prepare_mask)

    prepare_case = data_commands.add_parser(
        "prepare-benchmark-case",
        help="Bind an existing prepared partition and mask by exact file hashes.",
        description="Bind an existing prepared partition and mask by exact file hashes.",
    )
    prepare_case.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Study configuration YAML containing benchmark_case.id.",
    )
    prepare_case.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input interim trace dataset directory.",
    )
    prepare_case.add_argument(
        "--processed",
        type=Path,
        required=True,
        help="Prepared dataset partition directory.",
    )
    prepare_case.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Existing interpolation mask artifact directory.",
    )
    prepare_case.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output benchmark case directory.",
    )
    prepare_case.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace benchmark_case.json in an existing output directory.",
    )
    prepare_case.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare_case.set_defaults(handler=_prepare_benchmark_case)

    prepare_volume = data_commands.add_parser(
        "prepare-c3-volume-index",
        help="Bind a dense C3 5D crop and trace-to-cell mapping to a benchmark case.",
        description="Bind a dense C3 5D crop and trace-to-cell mapping to a benchmark case.",
    )
    prepare_volume.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Study configuration YAML containing benchmark_volume conditions.",
    )
    prepare_volume.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input interim trace dataset directory.",
    )
    prepare_volume.add_argument(
        "--processed",
        type=Path,
        required=True,
        help="Prepared dataset partition directory.",
    )
    prepare_volume.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Existing interpolation mask artifact directory.",
    )
    prepare_volume.add_argument(
        "--case",
        type=Path,
        required=True,
        help="Existing benchmark case directory.",
    )
    prepare_volume.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output C3 volume index directory.",
    )
    prepare_volume.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated volume index files in an existing output directory.",
    )
    prepare_volume.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare_volume.set_defaults(handler=_prepare_c3_volume_index)
