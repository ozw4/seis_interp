# Current experiment choice

The user explicitly requested starting the best configuration at 50000 updates.
Authorize this named budget exception for attention RMS with mean gate pooling;
retain the 20000 cap for other candidates. Start a fresh run from the existing
initialization seed with constant LR 0.001 and evaluate the final EMA 0.999.
Only max_steps and execution device change. The initial GPU1 launch was stopped
after its log reached update 100 following a memory attribution error: NVIDIA
process IDs differ from container process IDs, so the increased GPU1 usage could
include this run itself. Restart from initialization on the less occupied GPU0;
do not reuse the interrupted run weights. Other jobs were not stopped.
This is an additional-budget experiment, separate from
the original 20000-update goal and canonical result. Equal update counts with
CCNet/NeRSI do not imply equal FLOPs or exactly equal teacher presentations.

Verify the completed RMS experiments on 2026-09-18. Attention-only RMS reaches
19.2251 dB (+0.1802 versus geometry500), the highest measured fair-comparison
result. Gate-only RMS scores 18.7845 dB (-0.2604) and raises peak allocated CUDA
memory to 43.9766 GiB, compared with 41.6913 GiB for attention RMS. Retain mean
pooling for the gate in the next baseline; a combined RMS change is not supported
by these independent results. Single runs do not establish repeatability or
prove the cancellation hypothesis. Both retain constant LR 0.001, 20000 updates,
363269 parameters, 1253344 teacher presentations and final EMA 0.999. Input locks,
artifact hashes, strict EMA loads, query/coverage arrays and per-step graph/query
counts pass verification, and saved-prediction rescoring reproduces all metrics
exactly. Graph counts are not proof that every graph array is identical. The
20 dB goal remains unmet by 0.7749 dB; canonical adoption is unchanged. No new
experiment is launched by this review.

Prepare independent attention-RMS and gate-RMS candidates from geometry500 on
the user's request. The hypothesis is that temporal averaging can cancel
oscillatory latent features before neighbor or relation weighting. RMS retains
feature magnitude but loses sign, so improvement is unproven. Candidate 1 changes
only pooling of sender/destination latents for attention. Candidate 2 changes
only pooling of node latents and relation messages for the learned gate.
Message values, waveform normalization, loss, teacher sampling, graph settings,
initialization seed and final EMA evaluation remain fixed. The changes add no
parameters or random draws. They preserve 363269 parameters, constant LR 0.001,
20000 updates and the original 7 / 5 kernels. Compare against geometry500's
19.0449 dB and measure actual memory/runtime. Default mean pooling is retained
for existing experiments. Preparation does not launch or adopt either candidate.

On 2026-09-18, verify both completed temporal-kernel experiments. A scores
18.6468 dB and B 18.8211 dB, respectively 0.3981 and 0.2238 dB below geometry500.
Neither replaces the highest fair-comparison candidate (19.0449 dB); 20 dB
remains unmet. Both retain constant LR 0.001, 20000 updates, 363269 parameters,
1253344 supervised presentations and final EMA 0.999. Input locks, artifact
hashes, strict EMA loads, query/coverage arrays and per-step query/graph counts
pass verification, and all saved metrics are exactly reproduced from predictions.
Graph-count checks do not establish equality of every graph array. Allocated
GPU memory is effectively unchanged; measured training times differ, but these
single runs do not isolate architecture cost from shared-machine load. The
joint kernel changes and changed initialization shapes do not identify a causal
effect of either kernel alone. Canonical adoption remains unchanged; this review
does not launch another experiment.

Also prepare proposal B on the user's request: geometry500 with stem kernel
11 and message-round temporal kernels 3. Relative to geometry500, the stem
adds 512 parameters and the two depthwise kernels remove 512, retaining
363269. This tests the opposite allocation from A, toward the input encoder.
B inherits geometry500 directly and shares A's fixed comparison and final EMA
evaluation contract below. The two kernel changes are a joint comparison;
memory and runtime remain to be measured. Preparation does not launch a run.

On 2026-09-17, prepare proposal A from geometry500: reduce the encoder stem
kernel from 7 to 3 and expand both message-round temporal kernels from 5 to 7.
At width128 the stem removes 512 parameters and the two depthwise kernels
add 512, retaining exactly 363269. This tests allocating more temporal capacity
to message passing; it is a joint architecture comparison, not an isolated
effect of either kernel. Keep constant LR 0.001, 20000 updates, final EMA 0.999,
the original teacher sampling, graph construction, geometry scales and v3
evaluation contract. Start from initialization, not an existing checkpoint.
The fixed seed does not imply identical initial weights after shapes change.
Measure peak memory and runtime in the completed run; equal parameter counts
do not imply equal compute or memory. Compare all-target physical-amplitude
mean trace SNR against geometry500, verifying input lock, coverage, observed
reinsertion and artifact hashes. This request authorizes preparation and a
start command only; no run is launched or queued and canonical adoption is unchanged.

The separate feature-scale comparisons completed on 2026-09-17. Offset-only
500 m scored 19.0023 dB (-0.0167 versus neighbors6); position-only 500 m scored
19.0417 dB (+0.0226). The latter is only 0.0032 dB below geometry500 at 19.0449 dB.
Both runs retained constant LR 0.001, 20000 updates, 363269 parameters,
1253344 supervised trace presentations and final EMA 0.999. Input locks,
artifact hashes, strict EMA loads, query/coverage arrays and per-update
query/graph counts pass verification. Saved-prediction rescoring reproduces
all metrics exactly. The sampled outcomes suggest no large benefit from this
feature-scale change; single runs do not establish a repeatable ordering or
causal contribution of each feature family. Do not attribute the combined
gain entirely to position scaling. Geometry500 remains the highest measured
fair-comparison candidate, below 20 dB. No new run is launched by this review.

On the user's subsequent result-review request, geometry500 was found completed
in run `20260915T232958Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_geometry500_20k`.
Its mean trace SNR is 19.0449 dB, 0.0259 dB above neighbors6, making it the current
highest completed fair-comparison candidate. Constant LR 0.001, 20000 updates,
parameter/data budgets, input lock and final EMA evaluation were verified.
All saved metrics were exactly reproduced from the saved prediction. This small
single-run gain does not establish repeatability, and the 20 dB goal is still unmet.
The explicitly adopted canonical result is unchanged. This review does not
launch or queue another experiment.

The user requested stopping work after confirming the reference result and
providing the next experiment's start command. At that handoff, no geometry500
training was launched or queued by the assistant. Further launches require a
subsequent user instruction.
The reference-only LR 0.002 run completed at 19.3961 dB mean trace SNR.
Saved-prediction rescoring exactly reproduced every metric; input lock,
artifact hashes, strict EMA loading, query/coverage arrays and per-update
query/graph counts were verified. This score does not enter the fair-comparison
ranking or establish achievement of the 20 dB goal.

Rationale for the completed fair-comparison candidate, neighbors6 geometry500: reduce the position
and offset feature length scales together from 1000 to 500 m, leaving relation
search scales unchanged. This doubles the physical spatial frequencies of
the existing 16-component node Fourier encoding without adding parameters;
raw node and edge geometric features also scale accordingly. Test whether
finer spatial encoding helps represent variation across the fixed window.
This is a representation change, with no gain assumed. Preserve constant
LR 0.001, AdamW/zero decay, 20000 updates from initialization, 363269 parameters,
teacher sampling, MSE, O-only amplitude normalization and final EMA 0.999.
Graph topology should match neighbors6; feature tensors intentionally differ.
This candidate uses initialization, without using the reference-only LR 0.002 weights.

The user explicitly clarified that learning-rate changes break the required
fair comparison and must be retained only as reference records. This supersedes
the earlier decision to treat LR as a tuned comparison hyperparameter below.
Require constant LR 0.001 for every subsequent fair-comparison candidate.
Classify neighbors6 LR 0.002, neighbors6 late cosine, and the earlier width64
cosine experiment as reference-only. Do not count their scores toward the
20 dB goal or the fair-comparison ranking. The completed LR 0.002 run is
retained as a reference record; it does not replace the canonical result.
Constant-rate neighbors6 at 19.0190 dB remains the fair-comparison best.

Historical rationale for the reference-only experiment: constant LR 0.002
on neighbors6, changing only LR from 0.001.
The constant-rate reference's last-1000-update mean training loss fell from
0.0194483 at update 15000 to 0.0164379 at update 20000. Late cosine reached
0.0149908 but did not improve target SNR. Training loss is therefore not a
substitute for the full-target metric. A larger constant LR tests a different
optimization trajectory within the fixed budget; improvement is not assumed.
Retain initialization, query sampling, graph, 363269 parameters, MSE, O-only
normalization, AdamW/zero decay and final EMA 0.999. Start from initialization
for exactly 20000 updates. LR is explicitly tuned, rather than matched to
CCNet/NeRSI; the input/objective and approximate per-update data/parameter
comparison remains unchanged. The canonical result is not replaced.

Neighbors6 late cosine completed at 18.9676 dB, 0.0515 dB below constant-rate
neighbors6. Saved-prediction rescoring reproduces all metrics exactly. The
complete learning-rate history, input lock, artifact hashes, EMA loading,
query IDs, target coverage and per-update query/graph counts were verified.
This schedule did not improve the final EMA result in this fixed-seed run.
Neighbors6 at 19.0190 dB remains the highest completed candidate; the 20 dB
goal is still unmet and the explicitly adopted 18.2795 dB lock is retained.

Neighbors6 dilation11 completed at 18.7930 dB, 0.2260 dB below neighbors6.
Return to neighbors6 with [1, 2] dilations and test a late cosine learning-rate
schedule: hold 0.001 for 15000 updates, then decay to 0.0001 at update 20000.
This tests whether smaller final updates improve the final EMA state while
retaining most of the constant-rate optimization budget. The earlier width64
cosine candidate degraded performance; its decay started at 10000 and reached
0.00003, so it is evidence against aggressive early decay, not proof that this
late, milder schedule will improve the width128 EMA model. No gain is assumed.
The full experiment starts from initialization, with unchanged model, graph,
teacher stream, MSE, normalization, AdamW/zero decay and EMA. The LR trajectory
is now explicitly a tuned hyperparameter; comparison does not claim identical
optimizer trajectories across methods. The 20000-update GNN budget stays fixed.

Neighbors8 completed at 18.8649 dB, 0.1541 dB below neighbors6, with higher
peak GPU allocation. Do not continue increasing neighbor count as the immediate
next test. The following candidate used neighbors6 and changed only temporal
dilations from [1, 2] to [1, 1], testing local temporal processing without changing parameters,
graphs, teacher sampling or the optimizer. C's [1, 4] result motivates testing
the opposite direction, but C used neighbors2; it does not establish the effect
at neighbors6. The full run starts from initialization and uses exactly 20000
updates and final EMA evaluation. All other comparison conditions below remain.

The user authorized continued exploration toward at least 20 dB with exactly
20000 optimizer updates per full run. Neighbors6 completed at 19.0190 dB,
improving B by 0.4803 dB. Neighbors8 tested only the neighbor count after an
observed-only memory preflight. The preflight weights were discarded and never
evaluated on T; the full run started from initialization.
Keep the fixed v3 input, MSE, O-only Global RMS, no shear, 363269 parameters,
original query sampling, AdamW with zero decay and final EMA 0.999.
Constant LR 0.001 is mandatory for fair comparison; LR changes above are reference-only.
CCNet has 363292 parameters and exactly 64 supervised presentations/update;
the ten-profile NeRSI condition has 363269 parameters and approximately 64.
The GNN retains approximately 64, including its existing episode-tail batches.
These are matched input/objective and approximately matched parameter/data
budgets, not equal FLOPs or equal total training: CCNet/NeRSI use 50000 updates.
Do not enlarge the GNN update budget or change its loss/normalization to reach
the threshold. Target-informed selection remains exploratory evidence.

Prepare neighbors6 from B: B improved the adopted mean trace SNR by 0.2593 dB,
whereas C reduced it by 0.3010 dB. Test whether the neighbor-count improvement
continues while retaining B's dilation, parameter count, update budget and
final EMA evaluation. Do not combine B with C in this candidate. Additional
neighbors may raise memory and runtime; measure them rather than assuming
equal compute. This preparation does not adopt B or launch another run.

The next target is mean trace SNR above 20 dB. The user explicitly retained
the current update budget for the preceding B and C experiments. Both
start from the adopted EMA configuration with the original initialization
seed, not from its checkpoint. B isolates additional observed neighbors while
keeping two message-passing rounds; the earlier direct8 candidate also removed
a round and therefore did not isolate neighbor count. C expands temporal
receptive field through dilation without adding parameters. These changes
preserve the existing data and evaluation contract but do not imply equal
FLOPs, wall time or update budgets across the three methods.

The user explicitly adopted the completed EMA result on 2026-09-15 as the
current GNN canonical result. `gnn_v3_result.lock.json` binds the exact run and
evaluated EMA checkpoint. Its mean trace SNR is 18.2795 dB and exceeds the study
threshold. Raw final weights and all preceding runs remain retained. Adoption
does not turn target-informed exploration into independent evaluation evidence.

Use the mask10 width128 result as the reference for the EMA candidate. Increasing
width to 192 and reducing the inner mask to 5% did not improve their completed
final-state results.

The mask05 candidate tested whether reducing the inner pseudo-mask brings the
training context density closer to inference. It also changed episode length,
the sampled query sequence and episode-cache reuse despite retaining the seed
and optimizer-update budget. Its result does not justify replacing mask10.

The EMA candidate tests whether averaging successive optimizer states improves
final prediction. Use decay 0.999, initialized at the first successful update;
the averaging has an approximate 1000-update horizon. The training objective,
optimizer, mask sequence and budget stay fixed. Preserve raw final weights for
comparison and evaluate the final EMA state without validation or checkpoint
selection. This replaces raw final-state evaluation only for the EMA candidate.

Selection among completed candidates still uses the fixed target population
and remains target-informed exploration.
Judge the candidate by full-population physical-amplitude mean trace SNR, with
the same input lock, complete target coverage and exact observed reinsertion.
