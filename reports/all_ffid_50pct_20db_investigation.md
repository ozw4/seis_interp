# All-FFID 50% / 20 dB Investigation

The canonical experiment record is [Study 018](../studies/study_018_all_ffid_50pct_neighbor_inpainter/README.md).

## Conditions

- effective train / validation / test samples: 1,151,731 / 287,933 / 863,801
- seed: 42
- normalization: per-trace RMS
- optimizer: AdamW, learning rate 5e-4 with cosine decay to 1.5e-5
- batch size: 96
- weight decay: 1e-5
- dropout: 0.05
- gradient clipping: 1
- precision: bfloat16
- screening loss: MSE + 0.1 × time-derivative MSE
- evaluation: raw-domain `oracle_per_trace_unit_rms_global_snr_db`

## Screening results

| Stage | Change | Parameters / batch / steps | Validation S/N (dB) | Delta (dB) | Decision |
|---:|---|---|---:|---:|---|
| 01 | Study 017 model | 104 / 128 / 2,500 | 15.3760 | — | Baseline |
| 02 | Add source-x coordinate | — | 15.5931 | +0.2171 | Retain |
| 03 | Same-line neighborhood, K=274 | — | 15.7029 | +0.1098 | Retain |
| 04 | Crossline neighborhood, K=272 | — | 15.3473 | -0.2458 | Reject |
| 05 | Stage 03 at 10,000 steps | — | 17.4240 | +1.7211 | Retain training budget |
| 06 | FiLM conditioning | — | 15.7760 | +0.0731 | Retain |
| 07 | Width 256 and full temporal context | — | 16.3666 | +0.5906 | Retain bundle |
| 08 | Pure MSE and no dropout | — | 16.3555 | -0.0111 | Reject bundle |
| 09 | Stage 07 at 10,000 steps | — | 18.2484 | +1.8819 | Diagnostic best |
| 10 | Epoch sampler | — | 17.4221 | -0.0019 | Reject |
| 11 | Kernel size 31 | — | 16.1662 | -0.2003 | Reject |
| 12 | Gating | — | 16.4172 | +0.0507 | Retain |
| 13 | Width 384 | — | 16.6023 | +0.2357 | Retain |
| 14 | FIR-31 front end | — | 16.4975 | +0.1309 | Retain |
| 15 | Retained combination | — | 16.8037 | +0.4371 | Formal architecture |

## Formal result

| Step | Validation S/N (dB) |
|---:|---:|
| 1 | -3.2054 |
| 5,000 | 17.8862 |
| 10,000 | 18.7106 |
| 15,000 | 19.2885 |
| 20,000 | 19.6402 |
| 25,000 | 19.8833 |
| 30,000 | 20.1082 |
| 35,000 | 20.2611 |
| 40,000 | 20.3675 |
| 45,000 | 20.4340 |
| 50,000 | 20.4604 |

- model parameters: 9,210,121
- training-audit S/N: 20.7557 dB
- run ID and checkpoint: see Study 018
- implementation commit: `ee3d9e5d5fce73e3ce0450b3471fe3284af616a1`
- signal/error energy and audit results: see Study 018; all required audits passed

## Decision

- Accept the 50,000-step formal run as the Study 018 comparison result.
