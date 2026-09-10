# Study 030: SIREN per-trace scaling

Status: `per_trace_scaling_evaluated`

## Purpose

- Evaluate per-observed-trace RMS normalization with observed-only RMS interpolation at missing positions.

## Conditions

- suite: Study 029 QC suite
- case: `c3_benchmark_validation_random_trace_80_seed142`; volume `[384,9,32,8,32]`
- traces: 14,729 observed; 58,999 evaluation targets
- scale interpolation: 8-neighbor inverse-distance weighting, power 2, coordinate scales `[160,80,40,40]` m
- model / training: Study 029 SIREN condition; batch 16,384, seed 20260908, 2,000 updates
- evaluation: all target samples in physical amplitude; test unused
- config / method: [`config.yaml`](config.yaml), [`methods/siren.yaml`](methods/siren.yaml)

## Results

- target S/N: -0.0002 dB
- target RMSE: 9.9389
- RMS interpolation RMSE: 0.5290
- RMS interpolation relative L2: 5.3229%
- median RMS relative absolute error: 1.9647%
- report: [trace-scaling record](../../reports/c3_siren_trace_scaling_20260909.md)

## Decision

- The per-trace scaling condition did not materially improve the Study 029 global-RMS SIREN result.
