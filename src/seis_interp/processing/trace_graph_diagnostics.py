"""Query-only geometry diagnostics for exact observed dependency graphs."""

from __future__ import annotations

import resource
import sys
from itertools import combinations

import numpy as np

from seis_interp.processing.trace_graph_subgraphs import TraceGraphPlan


def trace_graph_resource_measurements(device: object = "cpu") -> dict[str, object]:
    """Read process/device counters without claiming unobserved GPU memory.

    These are process-lifetime high-water marks, not isolated operation peaks.
    Reading the CUDA peak does not reset global counters used by other runs.
    """
    import torch

    device = torch.device(device)
    cpu_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1024
    cuda = device.type == "cuda"
    return {
        "process_max_rss_bytes": int(cpu_peak * multiplier),
        "cpu_memory_scope": "process_lifetime_peak",
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if cuda
        else None,
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if cuda
        else None,
        "cuda_memory_scope": "device_process_lifetime_peak" if cuda else "unmeasured",
    }


def summarize_trace_graph_queries(plan: TraceGraphPlan) -> dict[str, object]:
    """Describe fully expanded query destinations, excluding truncated leaves.

    Spread is the sender-coordinate range along each physical axis, averaged
    only over nonempty query neighborhoods. Both-empty Jaccard is undefined;
    its count is retained separately from the defined-pair mean.
    """
    names = plan.relation_names
    query_count = len(plan.query_indices)
    neighborhoods = []
    relation_rows = []
    query_destinations = np.isin(plan.edge_index[1], plan.query_indices)
    for relation_id, name in enumerate(names):
        degrees = []
        nearest = []
        spreads = {key: [] for key in ("source", "receiver", "midpoint")}
        selected = []
        for query in plan.query_indices:
            edges = (plan.edge_index[1] == query) & (plan.edge_type == relation_id)
            senders = np.unique(plan.edge_index[0, edges])
            selected.append(set(plan.trace_ids[senders].tolist()))
            degrees.append(len(senders))
            if len(senders):
                nearest.append(float(plan.edge_distances[edges].min()))
                for key in spreads:
                    coordinates = getattr(plan.geometry, f"{key}_xy_m")[senders]
                    spreads[key].append(np.ptp(coordinates, axis=0))
        nonempty = sum(value > 0 for value in degrees)
        relation_rows.append(
            {
                "relation": name,
                "query_count": query_count,
                "degree_sum": sum(degrees),
                "mean_degree": sum(degrees) / query_count if query_count else None,
                "max_degree": max(degrees, default=0),
                "empty_query_count": query_count - nonempty,
                "empty_fraction": (query_count - nonempty) / query_count if query_count else None,
                "nearest_distance_sum": sum(nearest),
                "nearest_distance_count": nonempty,
                "mean_nearest_distance": sum(nearest) / nonempty if nonempty else None,
                "sender_spread_m": {
                    key: {
                        "nonempty_query_count": nonempty,
                        "mean_xy": np.mean(values, axis=0).tolist() if values else [None, None],
                        "max_xy": np.max(values, axis=0).tolist() if values else [None, None],
                    }
                    for key, values in spreads.items()
                },
            }
        )
        neighborhoods.append(selected)
    overlaps = []
    for left, right in combinations(range(len(names)), 2):
        values = []
        both_empty = 0
        for first, second in zip(neighborhoods[left], neighborhoods[right], strict=True):
            union = first | second
            if union:
                values.append(len(first & second) / len(union))
            else:
                both_empty += 1
        overlaps.append(
            {
                "relations": [names[left], names[right]],
                "jaccard_sum": sum(values),
                "defined_query_count": len(values),
                "both_empty_query_count": both_empty,
                "mean_jaccard": sum(values) / len(values) if values else None,
            }
        )
    direct_edges = plan.edge_index[:, query_destinations]
    context_count = sum(np.any(direct_edges[1] == query) for query in plan.query_indices)
    return {
        "destination_scope": "queries_only",
        "query_count": query_count,
        "context_query_count": int(context_count),
        "no_context_query_count": int(query_count - context_count),
        "no_context_fraction": (query_count - context_count) / query_count if query_count else None,
        "typed_edge_count": int(query_destinations.sum()),
        "unique_pair_count": int(np.unique(direct_edges, axis=1).shape[1]),
        "direct_support_node_count": int(len(np.unique(direct_edges[0]))),
        "relations": relation_rows,
        "relation_jaccard": overlaps,
        "closure": dict(plan.diagnostics),
    }
