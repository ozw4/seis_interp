# Study 032: C3 proposed GNN above 10 dB

Status: `final_5000_adopted_10db_achieved`

## Purpose

- Evaluate the proposed relational GNN with bounded amplitude handling and a relative-MSE training objective on the fixed QC validation case.

## Conditions

- suite SHA-256: `f707a2e09dc0c0c2e137c1c2aef3ed57ade7bf8a3f05e9b784f531d70ed3a3ee`
- case: `c3_benchmark_validation_random_trace_80_seed142`; 14,729 observed and 58,999 target traces, 22,655,616 target samples
- train pool: 1,146,366 QC-authorized canonical train traces, time `[0,384)`
- model: width 64, attention width 32, four relations x two neighbors, two rounds, time downsample 2, query batch 128, no fixed shear or learned lag
- training: AdamW, learning rate `1e-3`, seed 20260908, 5,000 updates, 640,000 query presentations
- adopted amplitude path: observed traces normalized by their RMS; query gain reconstructed by deduplicated direct-neighbor D0-IDW; global RMS 28.6279 retained for physical units
- loss: `masked_trace_relative_mse`; hidden-trace RMS is used only as a loss divisor
- numerical settings: both TF32 environment overrides zero; `cudnn_benchmark=false`
- config: [`config.yaml`](config.yaml); [executed adopted condition](config_observed_trace_rms_relative_mse_fp32_no_benchmark_5k.yaml)

## Results

| Amplitude condition | Checkpoint | Target S/N | RMSE | Auxiliary best |
|---|---|---:|---:|---|
| observed-trace RMS + IDW gain | final 5,000 | **11.9354 dB** | **2.5151** | step 3,000: 12.1987 dB |
| train-global RMS | final 5,000 | 10.3033 dB | 3.0350 | step 4,000: 11.0417 dB / 2.7877 |

- both final predictions passed full saved-output rescoring and fixed CPU checkpoint-restoration audits
- primary comparison uses the declared final 5,000-update checkpoint, not the auxiliary best checkpoint
- reports: [adopted evaluation](../../reports/c3_proposed_gnn_relative_mse_5k_20260909.md), [global-RMS comparison](../../reports/c3_proposed_gnn_global_rms_relative_mse_5k_20260910.md)

## Decision

- Adopt the observed-trace-RMS final model at 11.9354 dB.
- Retain `masked_trace_relative_mse` and the no-benchmark numerical condition.
- The train-global-RMS result exceeds 10 dB but does not replace the adopted model.
