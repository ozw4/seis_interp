"""Train ReGSI-real and score fixed held-out geometry in a condition search."""

import numpy as np

from seis_interp.training.forge_mvp_methods import graph_data, initialize_model, model_state_hash
from seis_interp.training.relational_trace_graph_poc_trainer import train_relational_trace_graph_poc
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.trace_graph_ema import TraceGraphEMA


def train_regsi_condition(inputs, config, query_cells, reporter=print):
    """Use the existing pseudo-mask trainer; held-out waveforms are not accepted."""
    if not set(query_cells).issubset(inputs.test_cells):
        raise ValueError("evaluation queries must be held out")
    model = initialize_model(config)
    initialization_hash = model_state_hash(model)
    domain, training, settings = graph_data(inputs, config)
    tr = config["training"]
    ema = TraceGraphEMA(tr["ema_decay"]) if tr.get("ema_decay") else None
    result = train_relational_trace_graph_poc(
        model,
        training,
        graph_settings=settings,
        random_seed=config["model_seed"],
        inner_mask_fraction=tr["inner_mask_fraction"],
        max_steps=tr["max_steps"],
        query_batch_size=tr["query_batch_size"],
        learning_rate=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
        gradient_clip_norm=tr["gradient_clip_norm"],
        report_interval=tr["report_interval"],
        loss=config["loss"],
        mixed_precision=tr["mixed_precision"],
        cudnn_benchmark=False,
        ema=ema,
        device=config["device"],
        reporter=reporter,
    )
    states = {"raw": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if ema is not None:
        states["ema"] = ema.state_dict()
    predictions = {}
    context = None
    for variant, state in states.items():
        model.load_state_dict(state)
        prediction = predict_relational_trace_graph(
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
        predictions[variant] = prediction.prediction
        context = prediction.has_observed_context
    return {
        "states": states,
        "predictions": predictions,
        "context": context,
        "history": result.history,
        "initialization_hash": initialization_hash,
        "query_presentations": result.query_count,
        "no_context_query_count": result.no_context_query_count,
    }
