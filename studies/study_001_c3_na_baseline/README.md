# study_001_c3_na_baseline

## Status

`draft`

## Research question

Can a sinusoidal coordinate MLP trained on observed SEG C3 Narrow-Azimuth traces reconstruct complete held-out traces more accurately than simple nearest-neighbor and inverse-distance-weighted trace interpolation?

## Hypothesis

The INR model will improve held-out-trace S/N or NRMSE over the simple physical-coordinate baselines while preserving the principal event spectrum.

## Inputs

The study uses a controlled subset of SEG C3 Narrow-Azimuth. Raw SEG-Y files remain outside Git. `inputs.yaml` locks the source file and its SHA-256 together with the selected shot: FFID 2348 of `SEG_C3NA_ffid_1201-2400.sgy`, 544 traces of 625 samples at 8 ms, no time window. Header conventions are in [`docs/coordinate_conventions.md`](../../docs/coordinate_conventions.md).

## Method

The POC treats SEG-Y as a trace table, audits SEG-Y headers, derives CMP/offset/azimuth coordinates, assigns complete traces to train/validation/test splits, and fits coordinate ranges and amplitude RMS from training traces only. It trains a SIREN-style MLP and compares predictions with the original held-out traces. The stored physical mapping remains 5D with `azimuth_deg`; model inputs contain six features and trace-baseline spatial inputs contain five because azimuth is encoded on demand as dimensionless sine and cosine components. Random and structured trace masks are separate evaluation conditions.

## Expected outputs

The initial accepted output set is a metrics JSON file, trace-level metrics table, reconstruction comparison figure, residual figure, and amplitude-spectrum comparison.

## Acceptance criteria

For the random holdout experiment, the provisional go/no-go criterion is either at least 2 dB higher global S/N than inverse-distance weighting or at least 10% lower NRMSE, with no dominant coherent event leakage in the residual. Structured holdout performance must be no worse than the baseline under the same qualitative checks.

## Limitations

SEG C3 NA has narrow azimuth coverage and is synthetic. A successful POC does not establish performance for wide-azimuth or noisy field surveys. The paper's architecture and training details are not completely specified, so repository choices must be documented as POC-specific decisions.

Historical rationale is recorded in [`decisions.md`](decisions.md).
