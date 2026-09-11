"""Training-only fanout configuration, separate from inference graph settings."""

from seis_interp import config_values
from seis_interp.processing.trace_graph_settings import TraceGraphSettings


def validate_trace_graph_edge_sampling(value, graph: TraceGraphSettings) -> dict[str, int]:
    """Require an exact seed/fanout mapping for indexed multi-relation training."""
    options = config_values.exact_section(
        {"training.edge_sampling": value}, "training.edge_sampling", {"seed", "fanout_per_relation"}
    )
    seed = config_values.nonnegative_integer(options["seed"], "training.edge_sampling.seed")
    fanout = config_values.positive_integer(
        options["fanout_per_relation"], "training.edge_sampling.fanout_per_relation"
    )
    if graph.topology != "multi_relation" or graph.neighbor_search != "exact_index":
        raise ValueError("training.edge_sampling requires multi_relation and exact_index")
    if fanout > graph.neighbors_per_relation:
        raise ValueError("training.edge_sampling.fanout_per_relation must not exceed candidate K")
    return {"seed": seed, "fanout_per_relation": fanout}
