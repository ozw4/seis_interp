# Study 030: SIREN per-trace scaling

Status: `per_trace_scaling_evaluated` — no material change from the Study 029 global-RMS SIREN

## Purpose

- Evaluate per-observed-trace RMS normalization with observed-only RMS interpolation at missing positions.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`methods/siren.yaml`](methods/siren.yaml).

- suite: Study 029 QC suite
- 14,729 observed traces; 58,999 evaluation targets
- evaluation: all target samples in physical amplitude; test unused

## Results

- target S/N -0.0002 dB; target RMSE 9.9389
- RMS interpolation RMSE 0.5290; relative L2 5.3229%; median RMS relative absolute error 1.9647%
- report: [trace-scaling record](../../reports/c3_siren_trace_scaling_20260909.md)
