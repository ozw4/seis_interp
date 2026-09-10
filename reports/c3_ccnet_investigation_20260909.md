# C3 CCNet-5D baseline diagnostics — 2026-09-09

Primary Study 033 conditions and the adopted result are in
[the study record](../studies/study_033_c3_ccnet_10db/README.md).

## Conditions

- case: `c3_benchmark_validation_random_trace_80_seed142`
- evaluation: 58,999 targets x 384 samples
- model: four Conv3D/Conv2D modules, width 8, kernel 3, linear output, 7,257 parameters
- patches: `[32,2,4,2,8]`; 64 fit and 16 selection descriptors
- training: seed 20260908, Adam `1e-4`, batch 1, two epochs / 128 updates, complete-patch MSE
- normalization: fit-region global RMS 8.2957

## Results

| Condition | Target S/N | RMSE | Model RMSE before reinsertion |
|---|---:|---:|---:|
| Study 028 baseline | -0.0027 dB | 9.9417 | 9.9443 |
| Study 033 QC baseline | -0.0027 dB | 9.9417 | 9.9443 |
| zero-fill | 0.0000 dB | 9.9386 | — |

- exact unrounded S/N change after QC rebinding: `+3.9088e-6 dB`
- final interval MSE: 0.2210; internal-selection S/N: -0.0053 dB
- native train / prediction: 3.1538 / 5.7112 s; outer actions: 35.05364 / 16.3646 s
- CPU restoration on tiles 0, 288, and 575 over 119,872 missing-core samples: normalized RMSE `5.4090e-9`, maximum difference `3.4129e-8`, physical relative L2 `1.5885e-7`; all fixed tolerances passed
- adopted later teacher regions: fit / selection RMS 8.3516 / 8.2394, 32,768 traces each, no overlap and no QC-excluded rows
- fit / selection energy in samples `[64,192)`: 92.5866% / 88.8248%; all-zero traces: 0 / 0

## Decision

- The 128-update baseline is not adopted.
- The QC rebinding did not materially change the baseline result.
