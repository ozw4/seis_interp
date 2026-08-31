# study_019_all_ffid_25pct_neighbor_inpainter

## Status

`in_progress`

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
FIR, and temporal CNN. Later stages will promote only measured gains and isolate
capacity and training budget. A final candidate must be frozen before a fresh
full-scope acceptance run.

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

No 25% training run has been accepted yet. Stage 01 reached
`14.222844294958843 dB` after 2,500 updates. This is 2.580838795167327 dB below
the same architecture and budget at 50% density. Stage 02's explicit neighbor
reference reached `14.22890961173312 dB`, only 0.006065316774276797 dB above
Stage 01, and was not promoted. Stage 03's single shared-attention fusion fell
to `9.819645233036228 dB` (`-4.403199061922615 dB` versus Stage 01), so early
compression of K274 neighbors was rejected. Stage 04 increased mean validation
availability from 54.788 to 132.690 traces with K734 but reached only
`14.089875885195529 dB` (`-0.132968409763314 dB` versus Stage 01), so aperture
growth was also rejected. All four full-scope runs passed the complete leakage
and scope audit and reproduced their saved checkpoint metric.

Immutable staged runs are stored under
`runs/study_019_all_ffid_25pct_neighbor_inpainter/` and will be summarized in
`reports/all_ffid_25pct_25db_investigation.md`.

Historical rationale is recorded in [decisions.md](decisions.md).
