# C3 proposed GNN observed-RMS FP32 result — 2026-09-09

## Conditions

- fixed QC validation case; 58,999 targets / 22,655,616 samples
- width 64, 101,701 parameters, two rounds, two neighbors per relation, query batch 128
- observed-trace RMS inputs and IDW query gain; physical `masked_mse`, not trace-relative MSE
- AdamW `1e-3`, seed 20260908, 5,000 updates, 640,000 query / 245,760,000 sample presentations
- TF32 overrides zero; training `cudnn_benchmark=true`, prediction false

## Results

| Condition | Target S/N | RMSE | Cross-run / CPU audit |
|---|---:|---:|---|
| previous global-RMS TF32 run | 7.4002 dB | 4.2395 | failed |
| observed-RMS FP32 run | **7.2797 dB** | **4.2987** | passed |

- validation curve: 2.8327, 3.5276, 3.8076, 5.1418, 7.2797 dB
- trainer-final / independent-frozen error-energy relative difference: `1.4420e-10`
- CPU restoration: normalized RMSE `1.7837e-7`, max difference `9.7939e-6`, physical relative L2 `6.6866e-7`; all limits passed
- native training / prediction: 4,341.5328 / 103.4689 s; GPU peak allocated 47.9984 / 1.4480 GB

## Decision

- Do not adopt this below-10-dB model.
- Its successful audit does not change the failed audit status of the previous global-RMS run.
