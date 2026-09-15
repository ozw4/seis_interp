# Study 026: grid-free multi-relation GNN

Status: `draft` — analytic smoke complete (execution evidence only), C3 runs not executed

## Purpose

- Compare separate source, receiver, CMP, and full-offset graph relations against matched untyped and single-graph controls.

## Conditions

Executable condition: [starting condition](config.yaml), [variants](variants/), [seed overrides](seeds/).

- declared case identities other than the seed-42 `random_trace` case remain `not_prepared`
- training pool: all canonical train-partition traces; fixed training RMS and geometry origin fitted on that pool
- primary metric: physical target-only global S/N; context-free queries remain in the metric with zero prediction

## Results

- analytic CPU integration smoke completed three updates for each mask kind, reloaded checkpoints, and predicted four arbitrary-coordinate queries
- fixture: two analytic Ricker arrivals on irregular observation lists; one context-free nonzero target retained in scoring
- smoke model: width 8, two neighbors; temporary test artifacts only
- no C3 preflight, C3 training, frozen C3 prediction, GPU calibration, multi-seed result, or field-data result exists
