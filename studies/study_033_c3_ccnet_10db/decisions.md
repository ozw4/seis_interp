# Study 033 decisions

## Fixed comparison and authorization

The user requests CCNET above 10 dB and has authorized automatic progression
through subsequent work. The Study 025 starting-budget restriction does not
prevent the explicitly requested CCNet investigation. Each resource and training
condition is nevertheless declared before its own execution so costs and
comparisons remain reviewable.

The Study 029 QC suite, validation geometry, all-target metric, and anomaly
exclusion are retained. SIREN's adopted fixed-shear result remains a separately
identified reference; CCNet receives no fixed shear. Final-budget checkpoints
are primary. Failed or below-threshold attempts are retained without changing
their recorded predictions or relaxing engineering thresholds.

## Baseline and spatial context

The earlier Study 028 CCNet used width 8, kernel 3 and 128 updates. Its fit and
selection regions exclude the location of the 437 anomalous traces, but a new
QC-bound run is needed to verify the common data contract. The new baseline also
disables TF32 environment overrides and uses physical GPU 0; it is not an
otherwise bitwise reproduction claim of the earlier GPU execution.

With four kernel-3 modules, each axis has a receptive field of nine samples.
The earlier `(32,2,4,2,8)` training patch therefore has artificial boundaries
throughout several spatial axes. Frozen volume prediction already acquires
the appropriate halo from the true volume. Increasing spatial context in
training is a motivated condition change; it is not yet evidence of a measured
improvement.

The first larger-context geometry splits the original fit volume by shot:
fit `[32,48)`, internal selection `[48,64)`, both source lines `[0,8)`, receiver
x `[0,8)`, receiver y `[17,49)`, and time `[0,384)`. Geometry, authorized-row
membership and disjointness were checked before reading waveform values.
Both regions contain 32,768 dense traces; they permit patches
`(64,8,16,8,16)`. No alternative region was selected by its model score.
The feasibility evidence is in
`runs/study_033_c3_ccnet_10db/20260909T125002842211Z_5df55f375e66_training_region_feasibility/`.

The resulting fit and internal-selection RMS are 8.3516 and 8.2394. The
near-offset energy domination found in the GNN's much broader training pool
cannot be assumed here. CCNet also has no GroupNorm layer. Complete-patch MSE
and the existing global RMS contract are therefore retained for the first
larger-context experiment. Short time-patch relative RMS needs separate
justification, especially for zero or very small-energy patches.

## Adopted final 2,048 updates

The first extended condition achieved physical global target SNR **11.6607 dB**
and RMSE **2.5959**. All 58,999 targets and 384 time samples were retained.
The saved-output rescore exactly reproduced native metrics and the fixed CPU
restoration met its original thresholds. Adopt final 2,048; best internal
selection happens to select the same step. No further quality retries or
training-budget extensions are needed for the requested 10 dB objective.

The condition retains complete-patch MSE and global fit RMS. Width, spatial
context, batch size, teacher-region split, learning rate and budget changed
jointly from the baseline. Sample presentations rose from 524,288 to
4,294,967,296, a factor of 8,192 including overlaps; this is not a single-factor
causal ablation. Neither internal train-selection SNR nor a subset waveform
score is substituted for the full physical validation metric.

Fixed qualitative locations reuse the earlier geometry-selected GNN gather
and three held-out traces, declared before extended CCNet predictions existed.
Truth from its previously published CSV is reused for visualization only.
Display scales, amplitudes and sample times are unchanged; no fitted gain or
alignment is applied.

This Study 033 reference adoption relies on frozen source identity, complete
saved outputs and independent audits. Actual dirty-worktree provenance and
native nonformal warnings remain visible. Publication source commit f2dccac
matches all 183 frozen Python files; this does not rewrite the training-time
Git state or assert the separate Study 025 clean-formal designation.
