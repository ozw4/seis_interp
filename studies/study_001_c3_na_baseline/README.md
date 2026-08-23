# study_001_c3_na_baseline

## Status

`draft`

## Research question

Can a sinusoidal coordinate MLP trained on observed SEG C3 Narrow-Azimuth traces reconstruct complete held-out traces more accurately than simple nearest-neighbor and inverse-distance-weighted trace interpolation?

## Hypothesis

The INR model will improve held-out-trace S/N or NRMSE over the simple physical-coordinate baselines while preserving the principal event spectrum.

## Inputs

The study uses a controlled subset of SEG C3 Narrow-Azimuth. Raw SEG-Y files remain outside Git. The selected sail lines, shots, time window, file checksums, and header conventions must be locked in `inputs.yaml` before the first formal run.

## Method

The POC will audit SEG-Y headers, derive CMP/offset/azimuth coordinates, hold out complete traces, normalize coordinates and amplitudes, train a SIREN-style MLP, and compare predictions with the original held-out traces. Random and structured trace masks will be evaluated separately.

## Expected outputs

The initial accepted output set is a metrics JSON file, trace-level metrics table, reconstruction comparison figure, residual figure, and amplitude-spectrum comparison.

## Acceptance criteria

For the random holdout experiment, the provisional go/no-go criterion is either at least 2 dB higher global S/N than inverse-distance weighting or at least 10% lower NRMSE, with no dominant coherent event leakage in the residual. Structured holdout performance must be no worse than the baseline under the same qualitative checks.

## Limitations

SEG C3 NA has narrow azimuth coverage and is synthetic. A successful POC does not establish performance for wide-azimuth or noisy field surveys. The paper's architecture and training details are not completely specified, so repository choices must be documented as POC-specific decisions.

## Decision log

- 2026-08-21: Use SEG C3 Narrow-Azimuth and prioritize proof of interpolation over exact paper reproduction.
- 2026-08-23: Step 1 treats SEG-Y as a trace table instead of a fixed 5D array. The coordinate rules are documented in [`docs/coordinate_conventions.md`](../../docs/coordinate_conventions.md) and produced with `python -m seis_interp.cli data prepare-c3-shot`. `inputs.yaml` is not locked yet.
