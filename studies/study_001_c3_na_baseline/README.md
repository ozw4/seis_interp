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

The POC will audit SEG-Y headers, derive CMP/offset/azimuth coordinates, hold out complete traces, normalize coordinates and amplitudes, train a SIREN-style MLP, and compare predictions with the original held-out traces. The stored physical mapping remains 5D, while numerical inputs contain six features because `azimuth_deg` is encoded on demand as dimensionless sine and cosine components. Random and structured trace masks will be evaluated separately.

## Expected outputs

The initial accepted output set is a metrics JSON file, trace-level metrics table, reconstruction comparison figure, residual figure, and amplitude-spectrum comparison.

## Acceptance criteria

For the random holdout experiment, the provisional go/no-go criterion is either at least 2 dB higher global S/N than inverse-distance weighting or at least 10% lower NRMSE, with no dominant coherent event leakage in the residual. Structured holdout performance must be no worse than the baseline under the same qualitative checks.

## Limitations

SEG C3 NA has narrow azimuth coverage and is synthetic. A successful POC does not establish performance for wide-azimuth or noisy field surveys. The paper's architecture and training details are not completely specified, so repository choices must be documented as POC-specific decisions.

## Decision log

- 2026-08-21: Use SEG C3 Narrow-Azimuth and prioritize proof of interpolation over exact paper reproduction.
- 2026-08-23: Step 1 treats SEG-Y as a trace table instead of a fixed 5D array. The coordinate rules are documented in [`docs/coordinate_conventions.md`](../../docs/coordinate_conventions.md) and produced with `python -m seis_interp.cli data prepare-c3-shot`.
- 2026-08-24: Lock the formal input to FFID 2348 of `SEG_C3NA_ffid_1201-2400.sgy`. The shot was checked on the real SEG-Y: 544 traces, 625 samples, 8 ms, `time_s` 0.0-4.992 s, all geometry and amplitudes finite. Mid-survey shots away from the sail-line ends are used instead of the smallest complete FFID, which sits at the start of the first line.
- 2026-08-24: PR2 implements the SIREN core following the paper equations: only the first layer applies `omega_0`, later sine layers use `omega=1`. Depth 4, width 256, and the standard SIREN weight initialization are POC-specific choices; training on real data follows in a later PR. Because `omega_0` and the layer sizes stay outside `state_dict()`, the later checkpoint must save the `Siren` constructor arguments next to the weights and rebuild the model from them; saving weights alone would silently restore a different function.
- 2026-08-24: Keep the interim `float64` physical coordinates and time axis unchanged and convert model inputs and training targets to `float32` at the training boundary with `to_model_tensors()`; `Siren.forward()` adds no implicit cast.
- 2026-08-24: PR3 uses whole-trace random train/validation/test splits and fits coordinate ranges and amplitude RMS from training traces only. The simple baselines use nearest-neighbor and inverse-distance weighting with Euclidean distance in numerical spatial features derived from the four physical spatial coordinates. These choices define the POC evaluation design rather than reproduce a paper-specified protocol.
- 2026-08-24: Baseline preparation resolves `configs/default.yaml`, this study's `config.yaml`, and explicit CLI overrides in that order. The processed metadata records the resolved split values and config source. Structured masking and IDW evaluation settings remain conditions for their future execution commands because preparation does not run either algorithm.
- 2026-08-24: Keep `azimuth_deg` in the interim trace table for physical provenance, and derive dimensionless `sin(deg2rad(azimuth_deg))` / `cos(deg2rad(azimuth_deg))` features on demand for model and baseline inputs. The unit-circle representation is continuous across the 0°/360° boundary. The scientific mapping remains physical 5D, while its numerical representation has six model features (five spatial features for trace baselines). The paper uses raw azimuth rather than this encoding, so the periodic representation is a POC-specific decision instead of a paper-reproduction requirement. Existing interim data remains valid; regenerate the processed normalization and its companion preparation outputs rather than regenerating SEG-Y-derived interim data or hand-editing generated files.
- 2026-08-25: PR4 samples training traces and time indices uniformly with replacement and defines one epoch as `steps_per_epoch` Adam updates with L1 loss. The paper motivates Adam, L1, and early stopping; the POC-specific choices are the sampling details, epoch length, validation batch size, device, selection of the best model by global S/N over every point of the whole validation traces, and patience-based stopping. Only the best checkpoint is saved, including the SIREN constructor arguments, weights, and normalization. Resume state, optimizer state, and test-split evaluation remain outside PR4.
