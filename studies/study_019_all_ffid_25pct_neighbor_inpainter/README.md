# Study 019: 25% within-FFID neighbor inpainter

Status: `completed`; strict 25 dB threshold not reached

## Purpose

- Test a leakage-safe neighbor inpainter when 25% of traces within every eligible FFID are assigned to training.

## Conditions

- dataset: all four SEG C3 NA sources, 4,780 eligible FFIDs; FFID 1746 excluded
- split: seed 42; 25% train, 18.75% validation, 56.25% test within each FFID
- prepared counts: 575,870 / 431,890 / 1,295,720; effective canonical counts: 575,864 / 431,887 / 1,295,714
- baseline model: accepted Study 018 architecture; each stage changes one declared mechanism
- metric: `oracle_per_trace_unit_rms_global_snr_db`; only train amplitudes enter inputs
- config / variants: [`config.yaml`](config.yaml), [`variants/`](variants/)

## Results

| Stage | Condition | Validation S/N | Change from Stage 01 |
|---:|---|---:|---:|
| 01 | Study 018 architecture, 2,500 updates | 14.2228 dB | — |
| 02 | aligned-neighbor residual reference | 14.2289 dB | +0.0061 dB |
| 03 | shared offset-aware attention | 9.8196 dB | -4.4032 dB |
| 04 | K734 aperture | 14.0899 dB | -0.1330 dB |
| 05 | K274 coarse receiver-y shift | 14.2043 dB | -0.0185 dB |
| 06 | width 512 | 14.4385 dB | +0.2157 dB |
| 07 | width 512, 10,000 updates | 16.3489 dB | +2.1261 dB |

- Stage 07 training audit: 16.5923 dB; checkpoint reproduced exactly; scope and leakage checks passed
- threshold shortfall: 8.6511 dB
- Stage 07 missed the 50,000-step promotion gate by 6.9013 dB
- runs: `runs/study_019_all_ffid_25pct_neighbor_inpainter/`
- best run: `20260831T030617Z_05dc7d9_stage07_width512_k274_10000_steps`
- best model: 14,898,313 parameters; commit `05dc7d9bbd913aec9c0caf33d9559273a5a61333`
- best validation signal / error energy: 269,929,375.0038355 / 6,256,857.8796
- best-run runtime: 2,048 s; CUDA peak allocated/reserved: 13.308/20.805 GiB

## Decision

- Do not promote Stages 02-05.
- Promote width 512 to the 10,000-update diagnostic; do not run a 50,000-update budget-only extension.
- The 25 dB target remains unmet.
