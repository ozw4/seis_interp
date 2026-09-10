# C3 proposed GNN amplitude-mode ablation — 2026-09-10

The authoritative conditions and adoption are in
[Study 032](../studies/study_032_c3_proposed_gnn_10db/README.md).

| Amplitude mode, relative-MSE loss | Final target S/N | Final RMSE | Auxiliary best |
|---|---:|---:|---:|
| observed-trace RMS | 11.9354 dB | 2.5151 | step 3,000: 12.1987 dB |
| train-global RMS | 10.3033 dB | 3.0350 | step 4,000: 11.0417 dB / 2.7877 |

- final difference: -1.6321 dB
- global RMS: 28.627922455065246
- global-RMS validation curve: best 11.0417 dB at step 4,000; final 10.3033 dB at step 5,000
- global-RMS CPU restoration: normalized RMSE `1.4623e-7`, max difference `1.2859e-5`, physical relative L2 `5.1474e-7`; all limits passed
- global-RMS native training / prediction: 2,919.0425 / 121.0184 s
- global-RMS GPU peak allocated/reserved: training 9.1865/38.1241 GB; prediction 1.4479/8.6549 GB

## Decision

- Keep the observed-trace-RMS model as the adopted Study 032 result.
