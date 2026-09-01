# Decisions

The current contract is defined by `README.md`, `config.yaml`, `inputs.yaml`,
code, tests, and immutable runs. This file records rationale relevant to changing
that contract.

## 2026-08-31 — Interpret 25% FFID as a disjoint whole-FFID split

**Status:** active

**Decision:** Select exactly 25% of the 4,780 amplitude-eligible FFIDs for
training. Assign every eligible trace of one FFID to the same split. Allocate
25% of the held-out FFIDs to validation and the remainder to test.

**Reason:** The user explicitly corrected the experimental condition from a
25% within-FFID trace ratio to selecting 25% of FFIDs. A separate study keeps
the earlier Study 019 run facts immutable and prevents the two estimands from
being conflated.

## 2026-08-31 — Start with the accepted K274 architecture and probe source coverage

**Status:** active

**Decision:** Stage 01 changes only the split relative to the accepted Study 018
architecture and trains for 2,500 updates. Geometry-only diagnostics precede
aperture changes. Subsequent stages change one mechanism at a time and use
fresh initialization.

**Reason:** Whole-FFID withholding changes source coverage qualitatively. The
matched baseline measures that shift, while the geometry probe identifies
whether same-line or crossline aperture is the first bottleneck without using
validation amplitudes.

**Geometry evidence:** K274 has 132,336 zero-neighbor validation traces
(30.28%). Same-line K494 and K934 reduce this to 10.17% and 1.21% respectively.
Crossline K714 reduces it to 3.56%, while crossline K1374 reaches zero and has a
minimum of 9 available train traces. Stage 02 isolates crossline support at the
original source-y radius; Stage 03 then isolates complete source-y coverage.

## 2026-08-31 — Match training context to held-out-FFID evaluation

**Status:** active

**Decision:** For every training target, mask all neighbor offsets whose exact
FFID ID equals the target FFID ID. Apply the same deterministic mask during
training audit and validation.

**Reason:** A validation FFID contributes no neighbor amplitudes, whereas an
unmasked training target can use hundreds of traces from its own FFID. That
train/evaluation context shift would optimize receiver interpolation within a
known shot instead of the requested held-out-shot interpolation. The mask
creates a pseudo-held-out FFID using train amplitudes only.

## 2026-08-31 — Use evidence-based promotion gates

**Status:** active

**Decision:** A 2,500-step model change must improve the matched Stage 01 result
by at least 0.20 dB, have a finite reproducible checkpoint metric, and preserve
all scope/leakage checks to be promoted. A 10,000-step winner is extended only
if its measured late slope and a documented empirical tail leave a credible
path to the strict 25 dB target.

**Reason:** Full validation covers 437,087 traces and longer GPU runs are
expensive. Predeclared gates keep the staged search interpretable and prevent
budget-only escalation after a clearly flattened curve.

## 2026-08-31 — Isolate an exact-receiver shot-axis prediction reference

**Status:** active

**Decision:** After the finite-aperture coverage stages, compare Stage 01 with a
K274 model whose initial prediction is the distance-weighted interpolation of
the nearest strict lower/upper train shots on the same source-x line and at the
same relative receiver coordinates. Use the nearest side without extrapolation
when only one side exists. Append the synthesized reference after local-neighbor
dropout and zero-initialize the residual head.

**Reason:** Whole-FFID withholding removes an entire source point. A
geometry-only diagnostic found exact-receiver two-sided brackets for 397,535 of
437,087 validation traces and a one-sided train source for every remaining
trace. The deterministic reference alone measured 5.4785273632 dB, versus
4.3999236270 dB for nearest-shot copying. This is not a 25 dB candidate by
itself, but it supplies a source-axis-aligned waveform while leaving the K274
CNN to learn the residual. Formal audits require zero unresolved rows, zero
non-train sources, zero target-FFID sources, and zero same-source-y sources.

## 2026-08-31 — Combine coverage and source reference only after independent promotion

**Status:** superseded for post-Stage-05 work by the continuation decision below

**Decision:** Run a 2,500-step Stage 05 with K1374 plus the exact-receiver
bracketing reference only if K1374 and K274-plus-reference each improve the
matched Stage 01 result by at least 0.20 dB and pass every scope check. Change no
other model or training setting. Do not promote capacity or budget if the
combined result remains below the predeclared evidence-backed 25 dB path.

**Reason:** The two stages address distinct failure modes: K1374 removes missing
finite-aperture context, while the reference supplies a direct long-range
source-axis interpolation. Testing them independently before their combination
separates their main effects from interaction and avoids adding mechanisms only
after seeing the combined metric.

## 2026-08-31 — Stop budget-only escalation after Stage 05

**Status:** superseded by the continuation decision below

**Decision:** Complete the study without a 10,000- or 50,000-step extension.
Retain Stage 03 K1374 as the best validated candidate at
8.719953365995504 dB. Record the strict 25 dB criterion as not reached.

**Reason:** Stage 03 was 16.280046634004496 dB below the threshold. The matching
Study 018 architecture gained 3.656672439472694 dB from 2,500 to 50,000
updates, so a short-run candidate needed 21.343327560527307 dB to retain that
empirical path. Stage 03 was 12.623374194531802 dB below the promotion gate.
Stage 05 did not provide positive interaction: 8.595997409114656 dB was
0.123955956880849 dB below Stage 03. More budget or the previously observed
small width gain has no evidence-backed route across the remaining gap.

## 2026-08-31 — Reopen the study until the strict threshold is measured

**Status:** active

**Decision:** Keep the exact 1,195 / 896 / 2,689 whole-FFID split and the raw
`oracle_per_trace_unit_rms_global_snr_db > 25` rule unchanged. Treat Stages
01--05 as immutable failed evidence and continue with mechanisms specific to a
wholly missing shot. First isolate without-replacement target coverage at the
same update count, then cover at least one complete train sweep. In parallel,
diagnose train-only f-k/POCS and a model that shares receiver coherence across
an entire 8 x 68 trace shot gather. Do not warm-start from Study 017--019,
because those checkpoints consumed amplitudes from FFIDs held out here.

**Reason:** The user explicitly requires an observed value above 25 dB, not an
evidence-based early stop. The original stop remains a valid conclusion about
budget-only scaling, but it cannot close the new objective. Reopening preserves
the split and evaluation estimand while moving to architectures that match the
whole-shot missingness pattern.

## 2026-08-31 — Promote one complete trace-model train sweep

**Status:** active

**Decision:** Change Stage 03 target sampling to epoch-without-replacement at
matched budget in Stage 06, then train Stage 07 for 6,030 updates so its
`96 trace/update` batches cover at least all 578,685 TRAIN traces once. Keep
K1374, width, split, target-FFID masking, loss, and metric unchanged.

**Reason:** Stage 06 measured 8.715600689719826 dB and showed that sampler order
alone did not explain Stage 03. Stage 07 improved from 8.970943588197336 dB at
step 3,015 to 9.099802401746661 dB at step 6,030. This is the new formal best,
but the second half added only 0.128858813549325 dB and still leaves
15.900197598253339 dB to the strict threshold.

## 2026-08-31 — Isolate a leakage-safe joint shot-gather path

**Status:** active

**Decision:** Add a thin formal pipeline that predicts each complete 8 x 68
target shot gather from the nearest TRAIN source gathers. Load only TRAIN and
validation amplitudes, exclude the target FFID, mask missing receiver cells,
zero-initialize an inverse-distance residual decoder, save the input feature
schema in the checkpoint, and revalidate the selected checkpoint on all 437,087
validation traces. Isolate receiver-y dilation, ordered source waveforms,
capacity, temporal field, receiver-cell FiLM, distance power, and objective
regularization one mechanism at a time.

**Reason:** Whole-FFID withholding removes a complete shot rather than isolated
traces, so receiver coherence is potentially useful. The path also provides a
formal home for train-only source-lattice diagnostics. All completed formal runs
in Stages 09--14 and 16--17 passed the same split, amplitude-access, collision,
target-mask, and checkpoint checks. Stage 15 had no formal run. The strongest
short joint-shot result so far is 7.028111512586028 dB, below the trace-model
best.

## 2026-08-31 — Promote capacity and squared-distance weighting only

**Status:** active

**Decision:** Promote Stage 12 width 128 and Stage 16 inverse-distance power 2
because each improves the matched Stage 09 condition by at least 0.20 dB. Test
their combination at 2,500 updates, and extend width 128 to five TRAIN sweeps.
Do not promote K16/K32, receiver-y dilation, full temporal field, ordered-raw
features, pure MSE, or receiver-cell FiLM as independent winners.

**Reason:** Width 128 measured 7.010553558041961 dB, `+0.227865449443491 dB`
over Stage 09. Squared-distance weighting measured 6.998159535214238 dB,
`+0.215471426615768 dB`; its zero-step reference was independently reproduced
at 6.443429389385823 dB. In contrast, K16/K32 geometry diagnostics degraded the
reference, and the rejected mechanisms changed Stage 09 or Stage 12 by at most
0.036650567075181 dB. Combination and budget runs preserve the original split
and acceptance rule.

## 2026-09-01 — Promote the observed joint-shot interactions

**Status:** active

**Decision:** Record neighbor-dropout removal as rejected. Promote the
time-varying, zero-initialized source-attention ablation as Stage 21. Because
Stage 19 exceeds Stage 12 by more than 0.20 dB, combine its 6,000-step budget
with the already promoted squared-distance reference as Stage 22. Stage 21
changes source weighting only; Stage 22 changes budget only relative to Stage
20. Keep the split, raw metric, checkpoint selection, and scope rules fixed.

**Reason:** Stage 18 measured 6.782754397212449 dB, only
+0.000066288613979 dB above Stage 09. Stage 20 measured
7.284502495688106 dB, +0.273948937646145 dB over Stage 12. Stage 19 measured
7.409153447682268 dB after 6,000 steps, +0.398599889640307 dB over Stage 12;
its second half still added 0.264818095201537 dB. This supplies the predeclared
promotion evidence for Stage 22 while Stage 21 addresses the independently
diagnosed time-, receiver-, and source-dependent weighting error.
