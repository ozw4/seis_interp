# C3 proposed GNN relative-MSE result — 2026-09-09

The authoritative comparison conditions and adoption are in
[Study 032](../studies/study_032_c3_proposed_gnn_10db/README.md).

## Results

- final 5,000-update target S/N / RMSE: **11.9354 dB / 2.5151**
- 1,000-step validation curve: 7.5269, 11.0035, 12.1987, 12.0540, 11.9354 dB
- auxiliary best at step 3,000: 12.1987 dB / RMSE 2.4400
- final truth / error energy: `2.2378e9` / `1.4331e8`
- trainer-final / independent-frozen error-energy relative difference: `1.4331e-10`
- CPU restoration on 1,143 queries: normalized RMSE `1.0269e-7`, max difference `6.1962e-6`, physical relative L2 `3.3955e-7`; all limits passed
- native training / prediction: 3,444.5858 / 98.2014 s
- GPU peak allocated/reserved: training 9.1872/41.7753 GB; prediction 1.4480/8.6549 GB

## Decision

- Adopt the declared final checkpoint; retain the step-3,000 checkpoint as auxiliary only.
