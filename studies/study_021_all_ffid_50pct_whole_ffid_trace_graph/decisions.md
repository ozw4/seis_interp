# Decisions

## Adopt the 50% whole-FFID split with the established seed-42 rule

The user requires that 50% of all target FFIDs be selected wholly into train,
with no FFID shared between train, validation, and test. The split reuses
`assign_random_whole_ffid_splits` exactly as accepted in study 020, changing
only `random_ffid_holdout_fraction` to 0.5. The holdout keeps the established
25% validation / 75% test allocation so the validation share stays comparable
with earlier studies. Observed counts: 2,390 / 598 / 1,792 FFIDs, effective
1,155,304 / 293,151 / 855,010 traces after the same global physical-coordinate
canonicalization; FFID 1746 remains fully excluded by amplitude QC.

## Model the task as trace-node graph message passing over whole gathers

The user requires one trace per node with time as an in-trace sequence or
latent feature, never as a graph coordinate. Study 020 evidence shows the
whole-FFID task is dominated by cross-shot information transport, and its
per-trace K-aperture models reached at most 8.72 dB at 25% density while joint
shot-gather reconstruction preserved gather structure end to end. The GNN
therefore operates on the joint gather node set — the 544 target-FFID trace
nodes plus 544 nodes per nearest train shot — and reuses the audited
shot-gather batch/evaluation machinery (`nearest_train_source_gathers`,
exact-FFID exclusion, availability masking) unchanged.

Edges are factorized into two sets applied alternately: receiver-lattice edges
within each shot, and source-axis edges between shots at the same
relative-receiver cell conditioned on inter-source deltas. This keeps the
per-round cost linear in nodes while every node can reach every other node
within a few rounds. A `source_receiver_bipartite` mode implements the
requested heterogeneous formulation: source nodes and receiver nodes aggregate
their incident observed-trace edges, and the target shot is a missing-edge set
reconstructed from its source node, receiver nodes, and geometry.

## Composite objective with isolated weights

`L = L_mask + lambda_spec L_spectrum + lambda_slope L_slope + lambda_amp
L_amplitude` follows the user's requirement. `L_mask` is the masked gather MSE
in the unit-RMS training domain (the artificially hidden target FFID).
`L_spectrum` uses rFFT log-magnitude matching plus magnitude-weighted phase
error. `L_slope` estimates local plane-wave slope from the target gather with
a detached structure tensor and penalizes the prediction's
destruction-residual mismatch along the receiver-y axis. `L_amplitude` matches
windowed RMS envelopes. All weights default to zero in the study config so
each stage isolates exactly one term against the mask-only baseline before any
combination stage.

## Control stages before GNN stages

Stage 01 and Stage 02 rerun the accepted study 020 architectures
(`shot_gather_inpainter` K8 moments and `neighbor_trace_inpainter` crossline
K1374) unchanged on the 50% split. They calibrate how much of any GNN result
comes from the easier split rather than the model, and give the matched
comparison every promotion decision requires. The promotion gate (+0.20 dB at
2,500 updates) and the evidence-based budget-extension rule follow studies
018-020.

## Keep the mask-only trace-lattice objective after loss isolation

Stages 04-06 isolated each requested composite-loss term against the
mask-only Stage 03 baseline at 2,500 updates. Spectrum (-0.09 dB), slope
(-0.02 dB), and amplitude (-0.25 dB) all failed the +0.20 dB promotion gate.
The primary metric is itself an L2 quantity and every model in this regime is
capacity-limited (train audit equals validation), so auxiliary shape losses
cannot move it. The terms remain implemented and configurable for later
budgets.

## Reject the bipartite mode and per-frame attention at this budget

The source-receiver bipartite mode (8.181 dB) lost 0.55 dB against matched
trace-lattice message passing, and per-frame (8.026 dB) and per-frame-shifted
(7.752 dB) attention lost up to 0.98 dB: attention weights that vary per
latent frame add degrees of freedom the 2,500-update budget cannot calibrate.
The pooled trace-lattice formulation remains the promoted graph.

## Promote width, depth, and batch; adopt activation checkpointing

Width 128 (+0.27 dB), twelve rounds (+0.31 dB), their combination
(+0.55 dB, additive), and batch 4 (+1.04 dB) all passed the gate, revealing a
data-scaling law: about +1.0 dB per doubling of gathers seen early, decaying
to +0.4 dB (Stage 13 budget curve 9.145 / 9.982 / 10.506 / 10.691 dB at
2.5k/5k/7.5k/10k). Scaling past batch 2 exceeded the 93 GB GPU, so the model
gained optional activation checkpointing; unit tests pin its outputs and
gradients to the uncheckpointed path.

## Record the same-offset correlation probe as a design constraint

Deterministic probes showed the inverse-distance same-relative-cell reference
reaches 7.089 dB at K=8 while aligning neighbors at the same physical receiver
collapses to -0.996 dB. Cross-shot edges must therefore connect equal
relative-receiver cells, which the trace-lattice source-axis attention already
does. Effective same-line shot spacing is 80 m (staggered 40 m grid).

## Reject two-pass recurrent refinement at this budget

Stage 20 fed the first-pass prediction back as the target-node waveform and
reran the shared rounds (8.777 dB, +0.04 dB against Stage 03), below the
+0.20 dB gate. Iterating the same weights does not add usable capacity here;
the data-scaling axis (batch and steps) remains the productive direction.

## Accept the scaled trace graph as study best; threshold remains unmet

Stage 19 (width 128, twelve rounds, batch 4, 10,000 updates, activation
checkpointing) reached 12.2393 dB with checkpoint revalidation and
the full formal scope audit passing, exceeding the strongest control
(per-trace K1374 at 11.318 dB) by 0.92 dB. Its budget curve
(10.700 / 11.684 / 12.068 / 12.239 dB at 2.5k/5k/7.5k/10k) still decays, so
budget alone leaves no evidence-backed path to the strict 20 dB requirement;
the study stays open with the continuation plan recorded in the report.

## Close the budget axis after Stage 21 convergence

Stage 21 (width 128, six rounds, batch 4, 25,000 updates) reached
12.4692 dB with all audits passing, the study best. Its final
increments (+0.09 / +0.06 / +0.03 dB per 2,500 updates) show the current
model family converging near 12.5-13 dB on this split, so further
budget-only extensions are not evidence-backed. Reaching the strict 20 dB
requirement needs a mechanism change (documented in the report's
continuation plan), not more updates.
