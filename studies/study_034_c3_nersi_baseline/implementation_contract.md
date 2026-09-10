# NeRSI repository reimplementation contract

This contract separates what Gao et al., *NeRSI: Neural implicit representations for 5D
seismic data interpolation*, pp. V32–V35 (especially equation (7), Figure 5, and the C3
experiment description) specifies from choices made for this repository. The method label is
`NeRSI (repository reimplementation)`; neither the official implementation nor complete
paper reproduction is claimed.

## Paper-specified

| Concern | Fixed C3 contract |
|---|---|
| Data | NMO-uncorrected SEG C3_NA 5D seismic data. |
| Axes | The paper represents the axes as `(time, s_x, s_y, r_x, r_y)`. |
| Profile mapping | Fix `(s_x, s_y, r_x)` and output the complete two-dimensional `(time, r_y)` profile. |
| Supervision | Minimize squared error at observed locations only; missing zero-filled locations are not training labels. |
| Coordinate encoding | Exponential Fourier feature mapping with `K=40` components for each input coordinate in the C3 experiment. |
| Encoder | Two fully connected layers. |
| Decoder | Three NeRSI modules, each with upsampling rate 2. |
| NeRSI module | Convolution, PixelShuffle, and activation. |
| Regularization | The C3 experiment does not use nuclear-norm regularization. |

## Repository reimplementation choice

| Concern | Fixed repository contract |
|---|---|
| Axis correspondence | `(time, source_line, shot_in_line, relative_receiver_x, relative_receiver_y)` corresponds to paper `(time, s_x, s_y, r_x, r_y)`. |
| Profile key | `(source_line, shot_in_line, relative_receiver_x)` in stable C order. |
| Profile axes | `(time, relative_receiver_y)`; the fixed validation profile shape is `(384, 32)`. |
| Key coordinates | Normalize each discrete selected-volume index axis independently to `[0, 1]`; a singleton axis is all zero. No physical coordinates or shear are used. |
| Fourier frequencies | `omega_i = pi * beta**i` for `i=1..K`, with baseline `beta=1.25`. Beta is not represented as a paper-complete-reproduction value. |
| Network details | GELU hidden activation, kernel size 3, linear signed-amplitude output, and PyTorch default Linear/Conv2d initialization. |
| Amplitude scaling | One global RMS accumulated in float64 from all time samples of observed traces in the selected volume only. |
| Objective | Adam with observed-sample-weighted masked MSE; unobserved samples are absent from numerator and denominator. |
| Stopping and checkpoint | Exactly the declared optimizer updates and a fixed-final checkpoint bound to the verified case and volume hashes; no early stopping, best selection, or cross-volume reuse. |
| Initial finite plan | Candidates A–D fix capacity and learning rate in their native fragments; all use 16 profiles per step, 5,000 updates, and seed `20260908`. |
| Evaluation boundary | Evaluation-target physical amplitudes are read only after the full prediction exists, by the existing target-only evaluator. |

## Unspecified / unresolved

| Concern | Status |
|---|---|
| Encoder width, latent channels, decoder channels | Not specified by the cited C3 description. Four repository candidates are declared before validation; the primary candidate remains unresolved until all four runs complete. |
| Activation, kernel size, initialization | Not specified by the cited C3 description; the repository choices above are not attributed to the paper. |
| Optimizer, learning rate, profile batch, update count | Not specified by the cited C3 description; the finite Study 034 choices are comparison settings, not reproduction claims. |
| Frequency base | The C3 experiment section does not restate beta. `1.25` is an explicit baseline choice rather than a paper-complete-reproduction value. |
| Prediction batch size | Fixed at 64 for all initial candidates, subject only to a separately recorded resource revision made without target metrics. |
| Validation outcome | No candidate score, adopted candidate, or test result exists before execution and independent audit. |

## Explicitly out of scope

| Concern | Initial C3 baseline contract |
|---|---|
| Nuclear norm | **nuclear norm: out of scope for initial C3 baseline**; the cited C3 experiment does not use it. |
| Grid and preprocessing variants | Off-grid prediction, NMO correction, and physical-coordinate remapping are excluded. |
| Windowing | Sliding windows and separate regional models are excluded. |
| Training lifecycle | Resume training, early stopping, best-checkpoint selection, and validation-driven candidate generation are excluded. |
| SIREN-specific behavior | Fixed receiver-y time shear, per-trace RMS/IDW gain restoration, and envelope loss are not inherited. |
| Data changes | No NeRSI-specific mask, benchmark case, volume artifact, additional labels, or target-derived normalization is created. |
| Evaluation | Test-partition execution and paper-value acceptance thresholds are excluded. |
