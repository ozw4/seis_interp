# Study 010: full-FFID temporal patches

Status: `completed`

## Purpose

- Test shared 64-sample temporal patches on all 435 FFID 2348 training traces.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- dataset / split / normalization: Study 007 condition

## Results

- classification / decision: `full_ffid_near_zero`
- best step 10,000: median training-trace S/N -0.001837 dB, global S/N -0.000147 dB, median trace correlation 0.000669, prediction/target RMS ratio 0.007035
- final median training-trace S/N: -0.146711 dB
- run / summary: `20260826T074127Z_b7bfe87_patch64_trace78_trace435`, `20260826T074127Z_b7bfe87_summary.json`
