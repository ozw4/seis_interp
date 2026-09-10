# C3 proposed GNN global-RMS 5,000-update result — 2026-09-09

## Conditions

- fixed QC case; 58,999 targets / 22,655,616 samples
- width 64, 101,701 parameters, two rounds, two neighbors per relation, query batch 128
- train-global RMS 28.6279, AdamW `1e-3`, seed 20260908, 5,000 updates
- 640,000 query / 245,760,000 sample presentations; no completed full mask episode

## Results

| Method | Target S/N | RMSE |
|---|---:|---:|
| zero-fill | 0.0000 dB | 9.9386 |
| GNN, 200 updates | 0.0144 dB | 9.9222 |
| GNN, 5,000 updates | **7.4002 dB** | **4.2395** |
| direct physical-waveform IDW | -0.7444 dB | 10.8280 |
| unit-waveform IDW x IDW gain | -0.7434 dB | 10.8267 |

- validation curve at 1,000-step intervals: 2.8512, 3.0524, 3.8284, 4.2245, 7.4001 dB
- full saved-output rescoring passed
- trainer-final / independent-frozen error-energy relative difference: `3.1848e-5`; failed `rtol=1e-6`
- CPU restoration: normalized RMSE `1.3034e-4` and max difference `8.9693e-3` failed; physical relative L2 `4.9432e-4` passed
- native training / prediction: 3,592.8192 / 98.9156 s; GPU peak allocated 47.7253 / 1.4479 GB

## Decision

- Do not adopt this model; retain the saved-output metric as valid and the cross-run/CPU audit failures as unresolved for this run.
