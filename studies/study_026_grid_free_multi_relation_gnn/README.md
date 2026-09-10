# Study 026: grid-free multi-relation GNN

Status: `draft`; analytic smoke complete, C3 runs not executed

## Purpose

- Compare separate source, receiver, CMP, and full-offset graph relations against matched untyped and single-graph controls.

## Conditions

- training pool: all canonical train-partition traces; fixed training RMS and geometry origin fitted on that pool
- validation case: existing seed-42 `random_trace` case; other declared case identities remain `not_prepared`
- training seeds: 42, 43, 44; artificial masks: `random_trace` and `random_whole_ffid`
- primary metric: physical target-only global S/N; context-free queries remain in the metric with zero prediction
- starting model: width 64, two rounds, time factor 2, eight neighbors per relation, relation scales 160/640 m, feature scales 1,000 m
- starting budget: 10,000 updates on time `[0,384)`; not executed
- comparisons: learned relation gate, equal mean, row-normalized plain GCN, untyped graph, single 4D graph, one-relation removals, and explicit-azimuth-feature removal
- configs: [starting condition](config.yaml), [variants](variants/), [seed overrides](seeds/)

## Results

- analytic CPU integration smoke completed three updates for each mask kind, reloaded checkpoints, and predicted four arbitrary-coordinate queries
- fixture: two analytic Ricker arrivals on irregular observation lists; one context-free nonzero target retained in scoring
- smoke model: width 8, two neighbors; temporary test artifacts only
- no C3 preflight, C3 training, frozen C3 prediction, GPU calibration, multi-seed result, or field-data result exists

## Decision

- Treat the analytic run as execution evidence only; no C3 performance or generalization claim is recorded.
