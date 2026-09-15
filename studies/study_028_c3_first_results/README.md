# Study 028: C3 first results

Status: `first_results_complete` — GNN cross-run tolerance unmet; pilot budgets, not a model ranking

## Purpose

- Produce finite-budget full-volume validation results for POCS, DRR, SIREN-5D, CCNet-5D, and RelationalTraceGraphInterpolator on one fixed Study 027 case.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`methods/`](methods/).

- evaluation: 58,999 target traces, 22,655,616 samples

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
- GNN full-train-pool RMS approximately `4.1921e34`; 710 finite extreme samples contained almost all amplitude energy
- GNN train-time versus independent-prediction error-energy relative difference `2.0509e-4`; S/N difference 0.0009 dB; failed `rtol=1e-6, atol=1e-12`
- resource revision: initial four-neighbor/batch-8 estimate 8,135.5484 s training and 3,792.0788 s prediction; adopted two-neighbor/batch-32 estimate 2,803.8469 s and 1,197.0107 s; measured 2,945.3956 s and 973.2474 s
- adopted artifacts: [CSV](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.csv), [JSON](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.json), [manifest](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/manifest.json)
