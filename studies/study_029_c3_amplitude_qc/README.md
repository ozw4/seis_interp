# Study 029: C3 amplitude QC

Status: `qc_and_numerical_validation_complete`

## Purpose

- Trace the extreme Study 028 GNN amplitudes to source data and validate a QC-bound training scale.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`probe.yaml`](probe.yaml), [`gnn_probe.yaml`](gnn_probe.yaml).

- benchmark geometry, time range, partitions, cases, and mask seeds: inherited from Study 027
- numerical probe: width-32 GNN, two neighbors per relation, AdamW, first four hidden queries, CPU

## Results

- excluded traces: 437, all in FFID 1746 and train partition; QC train traces 1,146,366; fixed train RMS 28.6279
- nonzero traces with lost float32 squared energy: 0
- all 20 old/new cases retained identical evaluation masks, volume indices, and held-out split membership
- four-query probe physical RMSE: 11.4754 before training; 10.5172 after 100 updates
- SIREN rerun with the same seed and 2,000 updates: target S/N -0.0003 dB; RMSE 9.9390
- reports: [QC](../../reports/c3_amplitude_qc_20260908.md), [SIREN rerun](../../reports/c3_siren_qc_rerun_20260909.md)

## Decision

- Adopt the QC suite for subsequent benchmark experiments; keep valid zero traces and do not clip or repair waveforms.
