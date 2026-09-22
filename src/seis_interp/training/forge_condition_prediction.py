"""Frozen ReGSI predictions for additional held-out condition diagnostics."""

import numpy as np

from seis_interp.training.forge_mvp_methods import graph_data, initialize_model
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph


def predict_regsi_condition(inputs, config, query_cells, state):
    """Predict only held-out cells; this boundary has no target amplitude argument."""
    if not set(query_cells).issubset(inputs.test_cells):
        raise ValueError("evaluation queries must be held out")
    model = initialize_model(config)
    model.load_state_dict(state)
    domain, training, settings = graph_data(inputs, config)
    return predict_relational_trace_graph(
        model,
        domain,
        training.preprocessing,
        graph_settings=settings,
        query_trace_ids=np.asarray(query_cells),
        observed_waveforms=inputs.read_observed(domain.trace_ids[domain.observed_mask]),
        query_batch_size=config["prediction"]["query_batch_size"],
        device=config["device"],
        measure_resources=False,
    )
