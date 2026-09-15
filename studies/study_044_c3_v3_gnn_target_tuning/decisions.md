# Current experiment choice

The next target is mean trace SNR above 20 dB. The user explicitly retained
the current update budget and requested preparation of B and C only. Both
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
