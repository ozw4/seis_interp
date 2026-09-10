# Study 028: C3 first results

Status: `first_results_complete`; GNN cross-run tolerance unmet

## Purpose

- Produce finite-budget full-volume validation results for POCS, DRR, SIREN-5D, CCNet-5D, and RelationalTraceGraphInterpolator on one fixed Study 027 case.

## Conditions

- suite SHA-256: `6447a35cc7d4de43532ee8e2e0de3245ac0e93d855efa7ff59c98a85e1631cab`
- case: `c3_benchmark_validation_random_trace_80_seed142`
- volume: shape `[384,9,32,8,32]`; time `[0,384)`, source lines `[41,50)`, shots `[32,64)`, receiver x `[0,8)`, receiver y `[18,50)`
- evaluation: 58,999 target traces, 22,655,616 samples; physical-amplitude global S/N and RMSE
- training seed: 20260908; mask seed: 142
- method budgets:
  - POCS: 20 iterations, threshold 0.9 to 0.01, windows `[128,4,8,4,16]`, overlaps `[64,2,4,2,8]`
  - DRR: rank 4, damping 4, 5 iterations, windows `[4,8,4,8]`, overlaps `[2,4,2,4]`, 0 Hz to Nyquist
  - SIREN: six features, width 256, four hidden layers, omega 30, learning rate `1e-4`, batch 16,384, 2,000 steps
  - CCNet: hidden/intermediate width 8, kernel 3, patches `[32,2,4,2,8]`, fit 64, selection 16, batch 1, two epochs
  - GNN: width 32, two rounds, attention 16, embedding 8, two neighbors per relation, train query batch 4, validation/prediction batch 32, 200 steps
- config / native method fragments: [`config.yaml`](config.yaml), [`methods/`](methods/)

## Results

| Method | Target S/N | RMSE |
|---|---:|---:|
| zero-fill | 0.0000 dB | 9.9386 |
| POCS | 12.8654 dB | 2.2597 |
| DRR | 7.2542 dB | 4.3114 |
| SIREN-5D | -0.0003 dB | 9.9390 |
| CCNet-5D | -0.0027 dB | 9.9417 |
| RelationalTraceGraphInterpolator | -567.8483 dB | `2.4532e29` |

- GNN: 200 steps, 800 training queries, 307,200 samples, zero completed full-train episodes, zero missing or duplicate query indices, zero context-free queries
- GNN full-train-pool RMS: approximately `4.1921e34`; 710 finite extreme samples contained almost all amplitude energy
- GNN train-time versus independent-prediction error-energy relative difference: `2.0509e-4`; S/N difference: 0.0009 dB; failed `rtol=1e-6, atol=1e-12`
- resource revision: initial four-neighbor/batch-8 estimate 8,135.5484 s training and 3,792.0788 s prediction; adopted two-neighbor/batch-32 estimate 2,803.8469 s and 1,197.0107 s; measured 2,945.3956 s and 973.2474 s
- adopted artifacts: [CSV](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.csv), [JSON](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.json), [manifest](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/manifest.json)

## Decision

- Retain all five declared-budget results, including low scores.
- Classify the pilot as complete, but do not treat the GNN cross-run audit as passed.
- These reduced models and unequal training-information regimes are pilot results, not a model-ranking conclusion.
