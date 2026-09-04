# study_019_all_ffid_25pct_neighbor_inpainter

## Status

`completed` — the 25 dB threshold was not reached

This study applies the 25% ratio inside every FFID. The whole-FFID variant of the
same ratio is a separate scope, covered by
[`study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter`](../study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/README.md).

## Research question

Can a leakage-safe geometry-conditioned trace inpainter exceed
`oracle_per_trace_unit_rms_global_snr_db > 25 dB` when exactly one quarter of
the amplitude-eligible traces in every FFID are assigned to training?

## Split contract

"25% FFID" means a deterministic whole-trace split inside every eligible FFID,
not a split of whole FFIDs. Seed 42 assigns 25% to train. The remaining 75%
keeps the established allocation: 25% of holdout (18.75% overall) is validation
and 75% of holdout (56.25% overall) is test. No time samples from a trace cross
split boundaries.

All four locked SEG C3 Narrow-Azimuth sources and all 4,780 amplitude-eligible
FFIDs remain in scope. FFID 1746 is wholly excluded by amplitude QC. Global
physical-coordinate canonicalization retains the lowest `array_row` without
consulting split or amplitude values.

Prepared train/validation/test counts are 575,870 / 431,890 / 1,295,720.
Canonicalization removes 6 / 3 / 6 rows, giving effective counts of 575,864 /
431,887 / 1,295,714. Every eligible FFID contributes all three splits.

## Evaluation and staged method

Only train amplitudes may populate neighbor inputs. Validation amplitudes are
used only for checkpoint selection and metrics; test and excluded amplitudes are
not materialized. Raw predictions are compared with oracle per-trace unit-RMS
validation targets.

Stage 01 changes only the split relative to the accepted Study 018 architecture
and evaluates it after 2,500 updates. Stage 02 keeps that condition fixed and
tests an explicit aligned-neighbor reference with a zero-initialized CNN
residual. Stage 03 tests offset-aware shared temporal encoding and time-dependent
masked attention at the same K274 aperture. Stage 04 changes only the accepted
CNN's same-line aperture from K274 to K734. Stage 05 returns to K274 and tests a
deterministic train-derived receiver-y moveout shift before the existing gate,
FIR, and temporal CNN. Stage 06 returns to the unshifted K274 model and changes
only hidden width from 384 to 512. Stage 07 extends the promoted width-512
condition to a fresh 10,000-update cosine horizon. A 50,000-update formal run is
allowed only if the Stage 07 absolute result leaves an evidence-backed path to
25 dB. A final candidate must be frozen before a fresh acceptance run.

## Acceptance criteria

- All 4,780 eligible FFIDs contribute train, validation, and test traces.
- Effective train/validation/test counts equal 575,864 / 431,887 / 1,295,714.
- Duplicate physical cells are canonicalized before amplitude routing.
- The target offset is absent and every neighbor amplitude comes from train.
- Test and excluded amplitude values are not materialized.
- The selected checkpoint reproduces its raw validation metric.
- `oracle_per_trace_unit_rms_global_snr_db` is strictly greater than 25 dB.
- Run provenance is complete and repository quality gates pass.

## Current result

The strict 25 dB target was not reached. Stage 01 reached
`14.2228 dB` after 2,500 updates. This is 2.5808 dB below
the same architecture and budget at 50% density. Stage 02's explicit neighbor
reference reached `14.2289 dB`, only 0.0061 dB above
Stage 01, and was not promoted. Stage 03's single shared-attention fusion fell
to `9.8196 dB` (`-4.4032 dB` versus Stage 01), so early
compression of K274 neighbors was rejected. Stage 04 increased mean validation
availability from 54.788 to 132.690 traces with K734 but reached only
`14.0899 dB` (`-0.133 dB` versus Stage 01), so aperture
growth was also rejected. Stage 05's deterministic coarse shift reached
`14.2043 dB` (`-0.0185 dB` versus Stage 01), so it was not
promoted. All five full-scope runs passed the complete leakage and scope audit
and reproduced their saved checkpoint metric. Stage 06 reached
`14.4385 dB`, a `+0.2157 dB` gain over Stage 01 and just
passed the predeclared `+0.20 dB` gate for a fresh 10,000-update diagnostic.

That Stage 07 run reached the study best of `16.3489 dB` at step
10,000, `8.6511 dB` below the strict target. Its late 2,500-step gain
had fallen to `+0.2158 dB`, and it missed the evidence-based
50,000-step promotion gate by `6.9013 dB`. The budget-only formal
extension was therefore stopped. The best checkpoint reproduced exactly,
the training audit reached `16.5923 dB`, and all scope and leakage
checks passed.

Immutable staged runs are stored under
`runs/study_019_all_ffid_25pct_neighbor_inpainter/` and are summarized in
`reports/all_ffid_25pct_25db_investigation.md`.

Historical rationale is recorded in [decisions.md](decisions.md).
