# Study 024: per-volume C3 NA SIREN

Status: `draft` — CPU and GPU smokes complete (execution-contract evidence only), formal volume not run

## Purpose

- Evaluate a SIREN fitted only to observed samples of each shared benchmark volume.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`config_smoke.yaml`](config_smoke.yaml).

- partition, case, mask, and formal/smoke volumes: Study 022
- features: time, CMP x/y, offset, azimuth sine/cosine; geometry bounds from known volume coordinates; one amplitude RMS from observed samples only
- selection: final checkpoint only; target metrics do not affect training

## Results

Both smokes used shape `[64,4,8,8,16]`, 821 observed traces / 52,544 samples, 3,275 targets / 209,600 samples, observed-only RMS 4.1126.

| Device / run | Target S/N | RMSE / relative L2 | Final batch loss | Train / predict | Peak memory |
|---|---:|---:|---:|---:|---:|
| CPU `20260907T054518Z_9c07ccf_smoke` | 1.2691 dB | 3.4626 / 0.8641 | 0.5018 | 3.8628 / 1.2181 s | 765.5781 MiB RSS |
| H100 `20260907T062042Z_a7c4986_gpu_smoke` | 1.2455 dB | 3.4720 / 0.8664 | 0.4991 | 1.1673 / 0.0677 s | 162.2725 / 186.0000 MiB CUDA |

- CPU commit `9c07ccfa0236e7c40cb3412dd1348e803dd77a4f`; GPU commit `a7c4986b98325f16cc426cc6d3173408edf32de8`; both clean
- both predictions finite, zero uncovered samples, observed traces reinserted exactly, all 262,144 saved points reproduced bitwise after same-device checkpoint reload
