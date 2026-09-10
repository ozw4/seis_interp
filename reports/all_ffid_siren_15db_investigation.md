# All-FFID SIREN 15 dB Investigation

The canonical experiment record is [Study 017](../studies/study_017_all_ffid_neighbor_inpainter/README.md).

## Conditions

- benchmark: 25% spatial trace sampling, 15 dB noise
- evaluation: held-out missing traces, `oracle_per_trace_unit_rms_global_snr_db`
- final method: leakage-safe neighboring-trace interpolation followed by time-domain smoothing
- configuration and run provenance: see Study 017

## Results

| Experiment | Data range | Validation S/N (dB) | Status |
|---|---:|---:|---|
| Random-point SIREN, width 512 | full survey | -0.000041 | Invalid: duplicate leakage found later |
| Full-FFID correlation sampling, threshold 0.3, width 512 | full survey | -0.001481 | Invalid: duplicate leakage found later |
| Complete-trace legacy 6D coordinates | FFID 2348 | 1.479477 | Leakage-safe subset |
| Range expansion | FFID 2348–2351 | 4.593618 | Intermediate |
| Width 256 | FFID 2348–2363 | 5.138709 | Intermediate |
| Width 512 | FFID 2348–2363 | 4.493958 | Intermediate |
| Cartesian CMP + half-offset coordinates | FFID 2348–2363 | 10.230463 | Intermediate |
| Cartesian CMP + half-offset, cosine schedule | FFID 2348–2363 | 10.324141 | Intermediate |
| Cartesian + offset radius | FFID 2348–2363 | 9.844906 | Intermediate |
| Cartesian width 512, cosine schedule | FFID 2348–2363 | 10.397155 | Intermediate |
| Cartesian, time scale 4 | FFID 2348–2363 | 10.473193 | Intermediate |
| Cartesian, width 256, omega0 90 | FFID 2348–2363 | 10.680488 | Best direct SIREN variant |
| Layer omega 90 to 30, four layers | FFID 2348–2363 | 10.444614 | Intermediate |
| Exponential omega 5 to 50 | FFID 2348–2363 | 0.000696 | Rejected |
| Exponential omega 30 to 90 | FFID 2348–2363 | 8.940770 | Rejected |
| Dense skip, omega 90 to 30 | FFID 2348–2363 | 9.804371 | Rejected |
| Fixed omega 30, 12 layers | FFID 2348–2363 | 0.010177 | Rejected |
| Dense skip, 12 layers | FFID 2348–2363 | 0.000344 | Rejected |
| Profile direct-convolution, latent 64, five bands | FFID 2348–2363 | 10.024054 | Intermediate |
| Profile direct-convolution, latent 64, five bands | FFID 2314–2382 | 10.345880 | Intermediate |
| ISR Fourier-ReLU, K=10 | FFID 2348–2363 | 4.522785 | Rejected |
| Masked 3D gather | FFID 2348–2363 | 7.730520 | Rejected |
| Source-relative coordinates | FFID 2348–2363 | 10.275549 | Intermediate |
| Cartesian global coordinates | FFID 2348–2363 | 10.140656 | Intermediate |
| Time mixture, five experts | FFID 2348–2363 | approximately 11.0311 | Intermediate |
| Leakage-safe moveout interpolation | FFID 2348–2363 | approximately 10.8098 | Intermediate |
| Target-referenced 32-neighbor least squares | FFID 2348–2363 | approximately 15.157 | Invalid: target leakage |

### Neighbor-interpolation scaling

| Training range | Validation S/N (dB) | Status |
|---|---:|---|
| 16 FFIDs | 16.1824 | Leakage-safe |
| 69 FFIDs | 16.5134 | Leakage-safe |
| Initial full survey | 18.0608 | Invalid: duplicate leakage |
| Full survey excluding two duplicate validation rows | 18.0608 | Diagnostic |
| Canonical formal run | 18.1119 | Accepted |

### Canonical formal run

| Step | Validation S/N (dB) |
|---:|---:|
| 1 | 0.6228 |
| 500 | 15.8008 |
| 1,000 | 16.9219 |
| 1,500 | 17.6202 |
| 2,000 | 17.9918 |
| 2,500 | 18.1119 |

- signal energy: 71,556,250.0032
- error energy: 1,105,250.1186
- prediction self-normalized S/N: 18.1034 dB
- training audit S/N: 18.1044 dB

## Decision

- Use only the canonical formal result for comparison.
- Classify target-leaking and pre-audit duplicate-contaminated results as invalid.
