# Decisions

## 2026-09-10 — Define the initial C3 NeRSI reimplementation boundary

Use the profile-wise C3 structure stated by Gao et al. while naming the method
`NeRSI (repository reimplementation)`. The available paper description does not fix every
network and optimization value needed by an executable implementation, so beta, activation,
kernel, channel, optimizer, and budget choices are recorded as repository choices rather
than attributed to the paper.

Do not add nuclear-norm regularization to the initial baseline. The cited C3 experiment does
not use it, and adding it would change rather than clarify the first comparison. Likewise,
do not inherit the SIREN receiver-y time shear, per-trace RMS restoration, or envelope loss;
those are prior SIREN study choices, not parts of the NeRSI profile contract.

Use local regular-grid key indices because the NeRSI input identifies a profile rather than
an individual physical trace point. Normalize source-line, shot-in-line, and receiver-x
indices independently within the fixed selected volume. This is deterministic, matches the
regular-grid scope, handles singleton axes explicitly, and does not imply a physical-distance
encoding.

Select any primary setting from validation only after all four declared candidates have
finished. Evaluation target amplitudes may contribute to final target-only metrics and that
single finite selection; they may not affect normalization, gradients, stopping, checkpoint
choice, resource revisions, or creation of additional candidates. Do not resolve, read, or
score the test partition in Study 034.

## 2026-09-10 — Freeze a finite validation matrix and fixed-final budget

Declare a two-by-two matrix before seeing Study 034 validation results: capacity
`(encoder=256, latent=64, decoder=[64,32,16])` or
`(encoder=384, latent=96, decoder=[96,48,24])`, crossed with learning rate `1e-3` or
`3e-4`. Hold `K=40`, beta 1.25, three rate-two decoder blocks, GELU, kernel 3, linear output,
observed-volume global RMS, 16 profiles per step, 5,000 Adam updates, training seed 20260908,
and prediction batch size 64 fixed across candidates.

The 5,000-update budget is a repository-side finite comparison budget. Its similarity to an
existing SIREN or GNN update count does not imply equal profile exposure, sample exposure,
training information, or compute. The primary artifact is the final fixed-step checkpoint;
do not produce or select a best checkpoint. Record a failed candidate without automatic
quality retry and do not add a fifth candidate after inspecting the four results.

## 2026-09-10 — Separate disposable resource preflight from validation

Use Candidate A architecture for a disposable ten-update smoke measurement and predict at
most one configured batch. Do not complete a full validation prediction or target scoring,
and do not reuse preflight state in a candidate run. Extrapolated training and prediction
costs are estimates rather than full-run measurements.

If the estimate exceeds the declared timeout or memory budget, do not let code silently
change channels, batch size, or update count. Record the evidence run and before/after values
here, add a new revision file instead of overwriting a declared config, and apply any primary
architecture reduction coherently across the four-candidate matrix. Make resource revisions
without consulting target metrics.

After all four fixed validation runs, use target-only physical global S/N as the primary
selection input. Treat RMSE, observed pre-reinsertion fit, coverage/non-finite status,
runtime, and memory as diagnostics. Paper-reported 15.16 dB and 18.67 dB values are not
acceptance thresholds for this repository reimplementation.
