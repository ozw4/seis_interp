# Study 025: C3 NA CCNet-5D

Status: `draft` — CPU smoke and GPU calibration complete (execution and resource evidence only), formal training/inference not run

## Purpose

- Validate supervised train-partition CCNet-5D training and frozen inference on the shared C3 benchmark.

## Conditions

Executable condition: [formal training](config_train.yaml), [CPU smoke](config_train_smoke.yaml), [GPU calibration](config_train_calibration.yaml).

- partition / case / mask: Study 022; CCNet additionally uses complete labels from train-only regions
- full-width candidate has 1,853,249 parameters
- method: four cross-convolution modules, signed linear output
- selection: missing-sample global S/N on disjoint train-partition selection patches; benchmark validation targets used only in frozen evaluation

## Results

- teacher preflight RMS, fit / selection: formal 7.8755 / 8.2312; calibration 17.2473 / 16.9012; CPU smoke 0.0658 / 0.0277
- CPU smoke training / inference: `20260907T140759Z_d783669_train_smoke` / `20260907T140839Z_d783669_infer_smoke`
- CPU smoke model 7,829 parameters, four updates; internal selection S/N -0.4997 dB
- CPU benchmark smoke: target S/N approximately 0 dB; RMSE 4.0073; relative L2 1.0000; zero-fill 0.0000 dB / 4.0074
- CPU timing: training + selection 0.8119 s; prediction 0.1193 s; peak RSS 3,513,176 / 589,504 KiB
- CPU prediction: all 821 observed traces reinserted exactly; all 3,275 targets covered; same-environment reload reproduced all 262,144 samples byte-for-byte
- duplicate CPU run `20260907T085141Z_d783669_train_smoke` has a mistyped directory timestamp; `run.json` start is `2026-09-07T14:07:09Z`; weights and histories match the correctly named run
- GPU calibration `20260907T141201Z_df66605_gpu_calibration`, commit `df666052bf76fa8324c175762a8c44815973bab3`, H100, one update: training loss 1.2045; internal selection S/N -0.0002 dB; 15.0282 s; CUDA peak allocated/reserved 3,597.6099 / 3,878.0000 MiB
