# Study 033: C3 CCNet-5D above 10 dB

Status: `final_2048_adopted_10db_achieved`

## Purpose

- Evaluate an expanded CCNet-5D condition on the fixed QC validation case.

## Conditions

Executable condition: [`config.yaml`](config.yaml); [executed condition](config_context_width32_batch2_2k.yaml).

- 14,729 observed traces, 58,999 targets
- train-only regions: fit shots `[32,48)`, internal selection `[48,64)`, time `[0,384)`, source lines `[0,8)`, receiver x `[0,8)`, receiver y `[17,49)`; 32,768 dense traces each
- global RMS 8.3516 fitted only on the complete fit region
- 111,969 model parameters

## Results

- physical target S/N **11.6607 dB**; RMSE **2.5959**; relative L2 **0.2612**
- internal patch-instance S/N 9.2691 dB; final checkpoint was also internal-selection best
- training + internal selection 580.4847 s; frozen prediction 2.3190 s; training peak allocated / reserved memory 4.4728 / 5.5512 GB
- CPU restoration on tiles 0, 12, 23 and 2,838,720 target samples: normalized RMSE `1.5475e-7`, max difference `2.0554e-6`, physical relative L2 `4.3232e-7`
- saved-output rescoring matched exactly; no fixed shear or fitted time alignment
- training run `20260909T130443857538Z_5df55f375e66_ccnet-train/native`; prediction run `20260909T131457466479Z_f2dccac6d1e3_ccnet-predict/native`
- final checkpoint SHA-256 `b8fb3a07e7a9d4a5b3c5540667abbfff8d87f4e24a6358f8cc9cd5e6cfa48697`; publication source commit `f2dccac6d1e33294156710dc688fc21667aff89a`

## Decision

- Adopt the final 2,048-update checkpoint as the Study 033 reference. The condition jointly changes context, width, batch size, learning rate, training-region split, and budget; no single-factor effect is assigned.
