# C3 proposed GNN numerical and lag diagnostics — 2026-09-09

## Numerical reproduction

Fixed checkpoint subset: 1,143 queries / 438,912 samples.

| GPU condition vs CPU | TF32 overrides | cuDNN benchmark | Normalized RMSE | Max difference | Physical relative L2 | Result |
|---|---|---|---:|---:|---:|---|
| A | 1 / unset | true | `2.0540e-4` | `7.8512e-3` | `7.7893e-4` | fail |
| B | 1 / unset | false | `1.3034e-4` | `8.9693e-3` | `4.9429e-4` | fail |
| C | 0 / 0 | true | `1.4861e-7` | `6.0201e-6` | `5.6448e-7` | pass |
| D | 0 / 0 | false | `1.4499e-7` | `6.0201e-6` | `5.5073e-7` | pass |

Fixed limits were `1e-4`, `1e-3`, and `1e-3`. A/B physical relative L2 was `7.2325e-4`; C/D was `1.5004e-7`.

## FP32 resource probe

- observed-trace RMS, width 64, query batch 128, one update
- graph/index 1.4459 s; reads 0.4192 s; forward 1.1596 s; backward/optimizer 2.1234 s
- worker / outer time 9.2005 / 11.2383 s
- GPU peak allocated/reserved 8.2587/8.9632 GB; process RSS 3.0092 GB

## Learned-lag diagnostic

| Four-query condition, 1,000 updates | S/N | RMSE |
|---|---:|---:|
| observed-trace RMS, no lag | 21.1003 dB | 1.0110 |
| observed-trace RMS, learned edge lag | 20.3853 dB | 1.0977 |

- difference: -0.7150 dB
- maximum learned shift: 0.2574 sample / 2.0589 ms; zero saturated edges

## Decision

- Disable both TF32 environment overrides for subsequent audited runs.
- Do not include learned edge lag in the full condition.
