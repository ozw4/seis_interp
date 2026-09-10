"""Use a verified suite's fixed crop and authorized training rows with existing readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from seis_interp.data.c3_benchmark_suite import (
    VerifiedC3BenchmarkSuite,
    c3_suite_case,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.data.c3_supervised_source import (
    C3SupervisedSource,
    load_c3_supervised_source,
    load_c3_training_dataset_source,
)
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs, load_c3_volume_run_inputs
from seis_interp.data.trace_graph_domain import (
    TraceGraphDomain,
    load_benchmark_trace_graph_domain,
    load_training_trace_graph_domain,
)
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions


def load_c3_benchmark_volume_inputs(
    suite_dir: Path,
    case_id: str,
    *,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
) -> C3VolumeRunInputs:
    """The shared POCS, DRR, SIREN-5D, and CCNet-5D observed-volume input contract."""
    suite = load_c3_benchmark_input_manifest(
        suite_dir, case_id=case_id, dimensions=dimensions, verified_suite=verified_suite
    )
    entry = c3_suite_case(suite, case_id)
    config = yaml.safe_load(suite_path(suite_dir, entry["config_file"]).read_text())
    return load_c3_volume_run_inputs(config=config, **_case_paths(suite_dir, suite, entry))


def load_c3_benchmark_graph_domain(
    suite_dir: Path,
    case_id: str,
    *,
    volume_dir: Path,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
) -> TraceGraphDomain:
    """Require the suite's volume and restrict graph support and queries to that crop.

    The general arbitrary-coordinate graph API remains independent of this entry.
    """
    suite = load_c3_benchmark_input_manifest(
        suite_dir, case_id=case_id, dimensions=dimensions, verified_suite=verified_suite
    )
    entry = c3_suite_case(suite, case_id)
    paths = _case_paths(suite_dir, suite, entry)
    if volume_dir is None or Path(volume_dir).resolve() != paths["volume_dir"]:
        raise ValueError("benchmark GNN requires the case's fixed volume_dir")
    return load_benchmark_trace_graph_domain(**paths, time_samples=dimensions.time_range)


def load_c3_benchmark_training_graph_domain(
    suite_dir: Path,
    *,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
) -> TraceGraphDomain:
    """Load the exact canonical train pool and selected training time grid."""
    suite = load_c3_benchmark_input_manifest(
        suite_dir, dimensions=dimensions, verified_suite=verified_suite
    )
    domain = load_training_trace_graph_domain(
        interim_dir=suite_path(suite_dir, suite["interim"]),
        processed_dir=suite_path(suite_dir, suite["processed"]),
        pool="all_train_traces",
        time_samples=dimensions.time_range,
    )
    allowed = np.load(suite_dir / suite["train_pool"]["file"], allow_pickle=False)
    if not np.array_equal(np.sort(domain.array_rows), allowed):
        raise ValueError("GNN training rows differ from the authorized pool")
    return domain


def load_c3_benchmark_supervised_source(
    suite_dir: Path,
    *,
    fit_region: dict,
    selection_region: dict,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
) -> C3SupervisedSource:
    """Constrain CCNet teacher regions and fitted RMS to the allowed train rows/time.

    CCNet fits its existing RMS on its fit-region subset of the authorized pool;
    the graph reader fits on its unmasked training pool. Neither uses the generic
    prepared normalization (which may include other time samples).
    """
    suite = load_c3_benchmark_input_manifest(
        suite_dir, dimensions=dimensions, verified_suite=verified_suite
    )
    for name, region in (("fit", fit_region), ("selection", selection_region)):
        if (
            region.get("time") != list(dimensions.time_range)
            and region.get("time") != dimensions.time_range
        ):
            raise ValueError(f"CCNet {name} time must equal the authorized training time")
    source = load_c3_supervised_source(
        interim_dir=suite_path(suite_dir, suite["interim"]),
        processed_dir=suite_path(suite_dir, suite["processed"]),
        fit_region=fit_region,
        selection_region=selection_region,
    )
    allowed = np.load(suite_dir / suite["train_pool"]["file"], allow_pickle=False)
    for region in (source.fit, source.selection):
        if (
            not np.isin(region.array_rows, allowed).all()
            or region.time_range != dimensions.time_range
        ):
            raise ValueError("CCNet training region is outside the authorized rows/time")
    return source


def load_c3_benchmark_training_dataset_source(
    suite_dir: Path,
    *,
    selection_region: dict,
    amplitude_rms: float,
    normalization_source: dict[str, object],
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
) -> C3SupervisedSource:
    """Load the complete QC train rows while retaining the existing selection region."""
    suite = load_c3_benchmark_input_manifest(
        suite_dir, dimensions=dimensions, verified_suite=verified_suite
    )
    allowed = np.load(suite_dir / suite["train_pool"]["file"], allow_pickle=False)
    source = load_c3_training_dataset_source(
        interim_dir=suite_path(suite_dir, suite["interim"]),
        processed_dir=suite_path(suite_dir, suite["processed"]),
        authorized_train_rows=allowed,
        time_range=dimensions.time_range,
        selection_region=selection_region,
        amplitude_rms=amplitude_rms,
        normalization_source=normalization_source,
    )
    if source.inputs_lock["training_dataset"]["authorized_trace_count"] != len(allowed):
        raise ValueError("CCNet training dataset differs from the authorized train rows")
    return source


def _case_paths(suite_dir: Path, suite: dict, entry: dict) -> dict[str, Path]:
    return {
        "interim_dir": suite_path(suite_dir, suite["interim"]),
        "processed_dir": suite_path(suite_dir, suite["processed"]),
        **{
            key: suite_path(suite_dir, entry[key]) for key in ("mask_dir", "case_dir", "volume_dir")
        },
    }
