# Study 008: full-FFID complete-trace batches

Status: `completed`

## Purpose

- Test complete-trace sampling on all 435 FFID 2348 training traces.

## Conditions

- dataset / split / normalization: Study 007 condition
- model: seed-42 six-input SIREN, width 256, four sine layers, `omega_0=300`
- training: MSE, Adam, learning rate `1e-3`, eight distinct complete traces per update (5,000 points), 50,000 updates, report every 500, `cuda:0`
- config: [`config.yaml`](config.yaml)

## Results

- classification / decision: `full_ffid_near_zero`
- best step: 3,500
- best median training-trace S/N: -0.001160 dB
- best global S/N: approximately 0 dB
- best median trace correlation: 0.002370
- best prediction/target RMS ratio: 0.005476
- final median training-trace S/N: -0.04134 dB
- run: `20260826T065417Z_fa548ba_tracebatch8_trace435`
- summary: `20260826T065417Z_fa548ba_summary.json`

## Decision

- Eight complete traces per update did not fit the full training pool under this condition.
