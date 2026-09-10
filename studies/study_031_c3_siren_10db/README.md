# Study 031: C3 SIREN above 10 dB

Status: `fixed_shear_reference_adopted_no_shear_10db_unmet`

## Purpose

- Improve physical target S/N above 10 dB on the fixed QC validation case and isolate the recorded temporal shear condition.

## Conditions

- suite: Study 029 QC suite; case `c3_benchmark_validation_random_trace_80_seed142`
- evaluation: 58,999 target traces and 22,655,616 samples; test unused
- observed inputs: 14,729 traces normalized by their own 384-sample RMS
- missing-position gain: observed-only 8-neighbor IDW, power 2, coordinate scales `[160,80,40,40]` m
- adopted model: Cartesian CMP + half-offset five-input SIREN, width 256, four layers, omega 30/30
- adopted training: all 14,729 complete observed traces per update, Adam, learning rate `1e-4`, 5,000 updates, seed 20260908
- adopted coordinate transform: time scale 12 and `tau = time_s + 0.0006 * relative_receiver_y_m`
- config: [`config.yaml`](config.yaml); [no-shear control](config_omega30_time12_shear0_batch14729_5k.yaml)

## Results

| Condition | Target S/N | Target RMSE | Model RMSE before reinsertion | Run |
|---|---:|---:|---:|---|
| Study 030, random points, 2,000 updates | -0.0002 dB | 9.9389 | 9.9414 | `20260909T003230820286Z_1819109f28e1_siren` |
| Cartesian5, 435 traces, 50,000 updates, time scale 1 | -0.7700 dB | 10.8599 | 9.0161 | `20260909T005658230199Z_1819109f28e1_siren` |
| Cartesian5, 435 traces, 50,000 updates, time scale 4 | -2.5383 dB | 13.3119 | 5.0725 | `20260909T010327892987Z_1819109f28e1_siren` |
| Cartesian5, 435 traces, 5,000 updates, time scale 4 | -1.7051 dB | 12.0943 | 6.7069 | `20260909T011656120058Z_1819109f28e1_siren` |
| Cartesian5, 14,729 traces, 5,000 updates, time scale 4 | -2.9480 dB | 13.9549 | 5.0412 | `20260909T011832015888Z_1819109f28e1_siren` |
| previous row + shear 0.0006 s/m | 7.8831 dB | 4.0102 | 3.3105 | `20260909T015011064447Z_1819109f28e1_siren` |
| omega 30, time scale 12, shear 0.0006 s/m | **11.3422 dB** | **2.6929** | 2.3861 | `20260909T021337632365Z_1819109f28e1_siren` |
| omega 30, time scale 12, no shear | -3.1486 dB | 14.2809 | 5.2413 | `20260909T030400486453Z_1819109f28e1_siren` |
| no shear + initial time weight x3 | -3.0870 dB | 14.1799 | 5.2205 | `20260909T041346896156Z_1819109f28e1_siren` |
| no shear + observed-envelope loss | -1.6301 dB | 11.9903 | 5.2220 | `20260909T043544755153Z_1819109f28e1_siren` |
| no shear + both additions | -2.7047 dB | 13.5693 | 5.5394 | `20260909T050203109381Z_1819109f28e1_siren` |

- observed-only shear diagnostic: 2,048 adjacent receiver-y pairs had a three-sample (24 ms per 40 m) shift; median Pearson correlation changed from -0.7132 to 0.9343 after alignment
- adopted run: full saved-output rescoring matched; fixed 16-target CPU checkpoint restoration passed its threshold
- A/B/AB saved-output rescoring matched; their standard CPU checkpoint-restoration checks did not pass

## Decision

- Adopt `20260909T021337632365Z_1819109f28e1_siren` only as an explicitly shear-transformed SIREN reference.
- The shear-free 10 dB goal remains unmet; do not adopt the A/B/AB models.
- Do not claim matched preprocessing against other methods and do not add fixed shear to them.
- Adoption record: [adoption decision](../../results/study_031_c3_siren_10db/20260909T055026000000Z_e39df16d563c_shear_reference_adoption/adoption_decision.json).
