"""Fixed physical error bands and observed-only zero/IDW comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.data.trace_graph_domain import TraceGraphDomain
from seis_interp.data.trace_graph_prediction_store import trace_graph_query_positions
from seis_interp.evaluation.physical_amplitude_metrics import physical_amplitude_energies
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_idw import predict_trace_graph_idw
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph


@dataclass(frozen=True)
class TraceGraphDiagnosticBands:
    """Fixed cut points; implicit outer bands retain every finite sample/query.

    Intervals are lower-inclusive and upper-exclusive. Azimuth is in [0,360)
    and has an additional undefined bucket, independent of the supplied cuts.
    """

    time_s: tuple[float, ...] = ()
    offset_m: tuple[float, ...] = ()
    azimuth_deg: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("time_s", "offset_m", "azimuth_deg"):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1 or raw.dtype.kind not in "iuf":
                raise ValueError(f"diagnostic {name} must contain finite numeric cut points")
            values = raw.astype(np.float64)
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
                raise ValueError(f"diagnostic {name} cut points must be finite and increasing")
            if name == "offset_m" and np.any(values < 0):
                raise ValueError("offset_m cut points must be nonnegative")
            if name == "azimuth_deg" and np.any((values < 0) | (values > 360)):
                raise ValueError("azimuth_deg cut points must be inside [0, 360]")
            object.__setattr__(self, name, tuple(map(float, values)))

    def constructor_config(self) -> dict[str, object]:
        return {name: list(getattr(self, name)) for name in ("time_s", "offset_m", "azimuth_deg")}


def summarize_trace_graph_prediction_bands(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    time_s: np.ndarray,
    source_xy_m: np.ndarray,
    receiver_xy_m: np.ndarray,
    bands: TraceGraphDiagnosticBands,
    azimuth_min_offset_m: float,
) -> dict[str, object]:
    """Aggregate physical energy/counts after prediction, without fitting cuts.

    Sample counts and energies sum to the global values for each grouping.
    Offset/azimuth groups additionally have additive query counts. Time groups
    omit query counts because one query may contribute to multiple time bands.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    if predictions.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError("predictions and targets must have matching physical shape [Q, T]")
    if time_s.shape != (predictions.shape[1],) or not len(time_s):
        raise ValueError("time_s must match the unpadded prediction samples")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise ValueError("predictions and targets must be finite physical amplitudes")
    if not isinstance(bands, TraceGraphDiagnosticBands):
        raise TypeError("bands must be TraceGraphDiagnosticBands")
    geometry = compute_trace_graph_geometry(
        source_xy_m, receiver_xy_m, azimuth_min_offset_m=azimuth_min_offset_m
    )
    if len(geometry.offset_m) != len(predictions):
        raise ValueError("geometry must contain one source/receiver pair per query")
    azimuth = np.degrees(np.arctan2(geometry.azimuth_sin, geometry.azimuth_cos)) % 360.0
    result = {
        "amplitude_domain": "physical",
        "bands": bands.constructor_config(),
        "query_count": len(predictions),
        **_energy_counts(targets, predictions),
    }
    for name, coordinates in (
        ("time_s", time_s),
        ("offset_m", geometry.offset_m),
        ("azimuth_deg", azimuth),
    ):
        cuts = getattr(bands, name)
        assignments = np.searchsorted(cuts, coordinates, side="right")
        rows = []
        for index in range(len(cuts) + 1):
            selected = assignments == index
            if name == "azimuth_deg":
                selected &= geometry.azimuth_valid
            reference = targets[:, selected] if name == "time_s" else targets[selected]
            predicted = predictions[:, selected] if name == "time_s" else predictions[selected]
            row = {
                "lower_inclusive": cuts[index - 1] if index else None,
                "upper_exclusive": cuts[index] if index < len(cuts) else None,
                **_energy_counts(reference, predicted),
            }
            if name != "time_s":
                row["query_count"] = int(selected.sum())
            rows.append(row)
        if name == "azimuth_deg":
            undefined = ~geometry.azimuth_valid
            rows.append(
                {
                    "undefined_azimuth": True,
                    "query_count": int(undefined.sum()),
                    **_energy_counts(targets[undefined], predictions[undefined]),
                }
            )
        result[name] = rows
    return result


def merge_trace_graph_prediction_band_summaries(
    summaries: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Add bounded summaries from batches that share the exact fixed cuts."""
    from copy import deepcopy

    result = None
    for summary in summaries:
        if result is None:
            result = deepcopy(summary)
            continue
        if result["bands"] != summary["bands"]:
            raise ValueError("diagnostic band summaries must use identical fixed cuts")
        for name in ("query_count", "sample_count", "reference_energy", "error_energy"):
            result[name] += summary[name]
        for axis in ("time_s", "offset_m", "azimuth_deg"):
            for destination, source in zip(result[axis], summary[axis], strict=True):
                for name in ("sample_count", "reference_energy", "error_energy"):
                    destination[name] += source[name]
                if "query_count" in source:
                    destination["query_count"] += source["query_count"]
    if result is None:
        raise ValueError("at least one diagnostic band summary is required")
    return result


def evaluate_trace_graph_diagnostic_bands(
    prediction: np.ndarray,
    domain: TraceGraphDomain,
    *,
    query_trace_ids: np.ndarray,
    bands: TraceGraphDiagnosticBands,
    azimuth_min_offset_m: float,
    amplitudes: np.ndarray | None = None,
    row_chunk_size: int = 1024,
) -> dict[str, object]:
    """Read only completed prediction targets, aggregating fixed bands in chunks."""
    positions = trace_graph_query_positions(domain, query_trace_ids)
    if domain.array_rows is None or np.any(domain.array_rows[positions] < 0):
        raise ValueError("diagnostic evaluation queries require target array rows")
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, int)
        or row_chunk_size < 1
    ):
        raise ValueError("row_chunk_size must be a positive integer")
    if prediction.shape != (len(positions), len(domain.time_s)):
        raise ValueError("prediction must have physical shape [Q, T]")
    if not len(positions):
        return summarize_trace_graph_prediction_bands(
            prediction,
            np.empty_like(prediction),
            time_s=domain.time_s,
            source_xy_m=domain.source_xy_m[positions],
            receiver_xy_m=domain.receiver_xy_m[positions],
            bands=bands,
            azimuth_min_offset_m=azimuth_min_offset_m,
        )
    if amplitudes is None:
        if domain.amplitudes_path is None:
            raise ValueError("target amplitudes are required for diagnostic evaluation")
        amplitudes = np.load(domain.amplitudes_path, mmap_mode="r", allow_pickle=False)
    start, stop = domain.time_samples
    if (
        amplitudes.ndim != 2
        or stop > amplitudes.shape[1]
        or np.any(domain.array_rows[positions] >= amplitudes.shape[0])
    ):
        raise ValueError("diagnostic target rows/time are outside amplitudes")

    def chunks():
        for offset in range(0, len(positions), row_chunk_size):
            selected = positions[offset : offset + row_chunk_size]
            targets = np.asarray(amplitudes[domain.array_rows[selected], start:stop])
            yield summarize_trace_graph_prediction_bands(
                prediction[offset : offset + row_chunk_size],
                targets,
                time_s=domain.time_s,
                source_xy_m=domain.source_xy_m[selected],
                receiver_xy_m=domain.receiver_xy_m[selected],
                bands=bands,
                azimuth_min_offset_m=azimuth_min_offset_m,
            )

    return merge_trace_graph_prediction_band_summaries(chunks())


def evaluate_trace_graph_baselines(
    domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    *,
    graph_settings: TraceGraphSettings,
    query_trace_ids: np.ndarray | None = None,
    query_batch_size: int = 64,
    amplitudes: np.ndarray | None = None,
    observed_waveforms: np.ndarray | None = None,
    midpoint_scale_m: float | None = None,
    offset_scale_m: float | None = None,
) -> dict[str, object]:
    """Score zero and unique-neighbor IDW on the same physical target queries.

    Build all baseline predictions before invoking the evaluation label reader.
    D0 scales default to the fixed preprocessing position/full-offset scales.
    """
    if (
        isinstance(query_batch_size, bool)
        or not isinstance(query_batch_size, int)
        or query_batch_size < 1
    ):
        raise ValueError("query_batch_size must be a positive integer")
    ids = (
        domain.trace_ids[domain.query_mask]
        if query_trace_ids is None
        else np.asarray(query_trace_ids)
    )
    positions = trace_graph_query_positions(domain, ids)
    if not np.array_equal(domain.time_s, preprocessing.time_s):
        raise ValueError("baseline time_s must match the fixed preprocessing time grid")
    source = (
        MaskedTraceSource(domain, preprocessing, amplitudes) if observed_waveforms is None else None
    )
    observed_domain_ids = domain.trace_ids[domain.observed_mask]
    if observed_waveforms is not None and observed_waveforms.shape != (
        len(observed_domain_ids),
        len(domain.time_s),
    ):
        raise ValueError("observed_waveforms must have shape [observed traces, T]")
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m,
        domain.receiver_xy_m,
        azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
    )
    midpoint_scale = (
        preprocessing.position_scale_m if midpoint_scale_m is None else midpoint_scale_m
    )
    offset_scale = preprocessing.offset_scale_m if offset_scale_m is None else offset_scale_m
    predictions = np.zeros((len(ids), len(domain.time_s)), dtype=np.float32)
    context = np.zeros(len(ids), dtype=bool)
    for start in range(0, len(ids), query_batch_size):
        rows = positions[start : start + query_batch_size]
        queries = compute_trace_graph_geometry(
            domain.source_xy_m[rows],
            domain.receiver_xy_m[rows],
            azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
        )
        plan = build_trace_graph_subgraph(
            queries,
            ids[start : start + query_batch_size],
            geometry,
            domain.trace_ids,
            domain.observed_mask,
            rounds=1,
            **graph_settings.subgraph_kwargs(),
        )
        observed_ids = (
            plan.trace_ids[plan.observed_mask] if source is not None else observed_domain_ids
        )
        predictions[start : start + len(rows)], context[start : start + len(rows)] = (
            predict_trace_graph_idw(
                plan,
                observed_trace_ids=observed_ids,
                observed_waveforms=source.read_observed_rows(observed_ids)
                if source is not None
                else observed_waveforms,
                midpoint_scale_m=midpoint_scale,
                offset_scale_m=offset_scale,
            )
        )
    zero = np.zeros_like(predictions)
    return {
        "distance": {
            "midpoint_scale_m": midpoint_scale,
            "offset_scale_m": offset_scale,
            "minimum_distance": 1e-6,
            "power": 2,
            "support": "unique_direct_observed_union",
        },
        **{
            name: evaluate_trace_graph_prediction(
                values,
                domain,
                query_trace_ids=ids,
                has_observed_context=context,
                amplitudes=amplitudes,
            )
            for name, values in (("zero", zero), ("idw", predictions))
        },
    }


def _energy_counts(reference: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    reference_energy, error_energy = physical_amplitude_energies(reference, prediction)
    return {
        "sample_count": reference.size,
        "reference_energy": reference_energy,
        "error_energy": error_energy,
    }
