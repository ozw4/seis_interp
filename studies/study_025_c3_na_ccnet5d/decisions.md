# Study 025 method decisions

## Signed final output

The supplied method specification identifies ReLU after both convolutions in Figure 4 and
Appendix B of Fang et al. (2023), but does not establish how a final ReLU represents signed
seismic amplitudes. The comparison candidate therefore removes only the last Conv2D ReLU.
Its variant is `supervised_train_partition_linear_output`. The
`supervised_train_partition_paper_relu_output` variant retains that activation for an audit;
neither variant claims to reproduce undocumented preprocessing. A positive RMS inverse cannot
restore negative amplitudes from a nonnegative output.

## Boundary padding and amplitude scale

Every convolution uses stride-one same zero padding. Boundary mode is an implementation choice,
not an established paper setting. A single global RMS fitted from complete fit-region labels
normalizes both fit and internal-selection patches and is saved with the model. Inference uses
that checkpoint RMS without refitting on the benchmark. There is no shift, clipping, absolute
value transform, or per-trace scaling. This explicitly uses additional supervised information.

## Internal selection region and score

Fit and internal-selection crops share train source lines but occupy disjoint shot-in-line
ranges. Separation by spatial trace, rather than only by time, keeps a trace from serving both
roles. Neither crop includes validation/test benchmark source lines. Their exact ranges are
our choices because the paper's patch start positions are not available in the specification.

Checkpoint selection minimizes relative squared error over all artificial missing samples
across the fixed internal-selection patch instances. This is equivalent to maximizing their
global S/N, not averaging patch S/N as in the paper's described selection rule. Overlapping
selection patches count repeated samples as separate patch instances. Ties retain the earlier
step. The independent benchmark target metric never selects a checkpoint inside a run.

## Halo/core inference

Each disjoint output core is evaluated with the model's receptive-field halo clipped to actual
volume bounds. Individual convolutions provide boundary padding, preserving the full-forward
boundary behavior even with bias and ReLU. Pre-padding the whole volume would create artificial
outside activations in later layers. Halo predictions are discarded, not blended. Observed
physical traces are reinserted exactly only after prediction and inverse RMS scaling.

## Real-C3 teacher crop feasibility

The initial CPU teacher time `[64,80)` contained only zero amplitudes in both regions and
was rejected before training. Adopt `[128,144)` for fit and internal selection while keeping
their spatial ranges, patch shape/counts, and seeds fixed. A read-only diagnostic found finite
values and at least one nonzero sample on every trace in both regions (RMS 0.0658/0.0277,
rounded). This is an input-feasibility choice, not selection by model score. Zero patches are
not removed and artificial masks are not redrawn.

For formal and full-width calibration teachers, restrict source lines to `[0,16)` and receiver y
to `[18,34)`, preserving the original disjoint shot ranges `[27,59)` / `[59,75)` and receiver x
`[0,8)`. This contains the original `(16,16,16,8,16)` patch. Source line 16 has discontinuous
shot spacing; source-line restriction alone still leaves missing cells at global indices
`(3,53,0,49)` and `(5,64,7,35)`. The narrower receiver-y subset excludes both without filling
labels or relaxing density checks. Shifting shot ranges instead was rejected after an additional
hole at `(5,76,0,21)` was found in the candidate selection range `[65,81)`.

The dense calibration crop with time `[64,128)` still places the fixed seed-42 selection patch
at `[67,83)`, where its artificial-missing reference energy is zero. Use `[128,192)` for both
calibration teacher regions, retaining the 64-sample extent, original patch size, seeds, and
region-local descriptors. These are geometry and label-energy feasibility checks, not searches
over model scores. Formal time remains `[0,384)`; calibration inherits its spatial crop.
