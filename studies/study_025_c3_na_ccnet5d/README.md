# Study 025: C3 NA CCNet-5D

Status: `draft`; CPU smoke and GPU calibration complete, formal training/inference not run

## Purpose

- Validate supervised train-partition CCNet-5D training and frozen inference on the shared C3 benchmark.

## Conditions

- partition / case / mask: Study 022; CCNet additionally uses complete labels from train-only regions
- formal teacher regions: fit time `[0,384)`, source lines `[0,16)`, shots `[27,59)`; selection shots `[59,75)`; receiver x `[0,8)`, receiver y `[18,34)`
- method: four cross-convolution modules; full-width candidate 64/64 channels, kernel 5, signed linear output, 1,853,249 parameters
- formal starting budget: 10,000 fit / 1,000 selection patches of `16x16x16x8x16`, batch 1, 20 epochs, complete-patch MSE
- selection: missing-sample global S/N on disjoint train-partition selection patches; benchmark validation targets used only in frozen evaluation
- configs: [formal training](config_train.yaml), [CPU smoke](config_train_smoke.yaml), [GPU calibration](config_train_calibration.yaml)

## Results

- teacher preflight RMS, fit / selection: formal 7.8755 / 8.2312; calibration 17.2473 / 16.9012; CPU smoke 0.0658 / 0.0277
- CPU smoke training / inference: `20260907T140759Z_d783669_train_smoke` / `20260907T140839Z_d783669_infer_smoke`
- CPU smoke model: 7,829 parameters, four updates; internal selection S/N -0.4997 dB
- CPU benchmark smoke: target S/N approximately 0 dB; RMSE 4.0073; relative L2 1.0000; zero-fill 0.0000 dB / 4.0074
- CPU timing: training + selection 0.8119 s; prediction 0.1193 s; peak RSS 3,513,176 / 589,504 KiB
- CPU prediction: all 821 observed traces reinserted exactly; all 3,275 targets covered; same-environment reload reproduced all 262,144 samples byte-for-byte
- duplicate CPU run `20260907T085141Z_d783669_train_smoke` has a mistyped directory timestamp; `run.json` start is `2026-09-07T14:07:09Z`; weights and histories match the correctly named run
- GPU calibration: `20260907T141201Z_df66605_gpu_calibration`, commit `df666052bf76fa8324c175762a8c44815973bab3`, H100, one update
- GPU calibration: training loss 1.2045; internal selection S/N -0.0002 dB; 15.0282 s; CUDA peak allocated/reserved 3,597.6099 / 3,878.0000 MiB

## Decision

- Treat the CPU run as an execution smoke and the GPU run as resource calibration only.
- No formal benchmark performance, test result, or full-paper reproduction is recorded.
