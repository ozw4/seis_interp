# Study 010: full-FFID temporal patches

Status: `completed`

## Purpose

- Test shared 64-sample temporal patches on all 435 FFID 2348 training traces.

## Conditions

- dataset / split / normalization: Study 007 condition
- model: seed-42 six-input SIREN, width 256, four sine layers, `omega_0=300`
- training: MSE, Adam, learning rate `1e-3`, 78 traces x 64 samples per update (4,992 points), 50,000 updates
- patch starts: `[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 561]`
- config: [`config.yaml`](config.yaml)

## Results

- classification / decision: `full_ffid_near_zero`
- best step: 10,000
- best median training-trace S/N: -0.001837 dB
- best global S/N: -0.000147 dB
- best median trace correlation: 0.000669
- best prediction/target RMS ratio: 0.007035
- final median training-trace S/N: -0.146711 dB
- run: `20260826T074127Z_b7bfe87_patch64_trace78_trace435`
- summary: `20260826T074127Z_b7bfe87_summary.json`

## Decision

- This temporal-patch condition did not fit the full training pool.
