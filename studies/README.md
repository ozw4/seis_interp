# Studies

Each numbered study directory is the authoritative record of one research question. The role of each study file is defined in [`docs/repository_layout.md`](../docs/repository_layout.md). Start a new study from [`_template/`](_template/README.md).

Numbered studies are listed in ascending ID order. Gaps in the numbering (for example `study_002`) are real: those IDs were never created.

## Numbered studies

| Study | Recorded status | Scope / model | Research question |
|---|---|---|---|
| [study_001_c3_na_baseline](study_001_c3_na_baseline/README.md) | `draft` | single-FFID trace split / SIREN | Coordinate SIREN vs. nearest-neighbor and inverse-distance baselines |
| [study_003_omega0_sensitivity](study_003_omega0_sensitivity/README.md) | `active` | single-FFID trace split / SIREN | `omega_0` × learning-rate sensitivity on FFID 2348 |
| [study_004_domain_scaling](study_004_domain_scaling/README.md) | `active` | single-FFID trace subsets / SIREN | Training fit vs. training-trace count |
| [study_005_correlation_loss_ablation](study_005_correlation_loss_ablation/README.md) | `completed` | eight-trace subset / SIREN | Correlation auxiliary loss against the near-zero predictor |
| [study_006_batching_ablation](study_006_batching_ablation/README.md) | `completed` | eight-trace subset / SIREN | Random-replacement batches vs. exact point coverage |
| [study_007_full_ffid_large_batch](study_007_full_ffid_large_batch/README.md) | `completed` | full FFID 2348 / SIREN | 435-trace fit under 5,000-point random batches, 50,000 updates |
| [study_008_full_ffid_trace_batches](study_008_full_ffid_trace_batches/README.md) | `completed` | full FFID 2348 / SIREN | 435-trace fit under eight complete traces per update |
| [study_009_full_ffid_trace_batch_correlation](study_009_full_ffid_trace_batch_correlation/README.md) | `completed` | full FFID 2348 / SIREN | Correlation loss under complete-trace batches |
| [study_010_full_ffid_temporal_patches](study_010_full_ffid_temporal_patches/README.md) | `completed` | full FFID 2348 / SIREN | 435-trace fit under shared 64-sample temporal patches |
| [study_011_trace_pool_continuation](study_011_trace_pool_continuation/README.md) | `completed` | full FFID 2348 / SIREN | 435-trace fit by nested training-pool expansion |
| [study_012_official_siren_baseline](study_012_official_siren_baseline/README.md) | `completed` | full FFID 2348 / SIREN | Official SIREN parameterization against the near-zero predictor |
| [study_013_amplitude_balancing](study_013_amplitude_balancing/README.md) | `completed` | full FFID 2348 / SIREN | Per-trace RMS balancing against the near-zero predictor |
| [study_014_full_trace_batch_ablation](study_014_full_trace_batch_ablation/README.md) | `completed` | full FFID 2348 / SIREN | Which ingredient of the 435-trace escape is necessary |
| [study_015_strong_fit_budget_extension](study_015_strong_fit_budget_extension/README.md) | `completed` | full FFID 2348 / SIREN | Strong fit from the Study 014 recipe with a longer budget |
| [study_016_all_ffid_siren](study_016_all_ffid_siren/README.md) | `draft` | per-FFID trace split / SIREN | One SIREN across every eligible FFID, held out within each |
| [study_017_all_ffid_neighbor_inpainter](study_017_all_ffid_neighbor_inpainter/README.md) | `completed` | per-FFID trace split / neighbor trace inpainter | Geometry-conditioned temporal network > 15 dB from train-only neighbors |
| [study_018_all_ffid_50pct_neighbor_inpainter](study_018_all_ffid_50pct_neighbor_inpainter/README.md) | `completed` | per-FFID trace split / neighbor trace inpainter | Leakage-safe inpainter > 20 dB at 50% train traces per FFID |
| [study_019_all_ffid_25pct_neighbor_inpainter](study_019_all_ffid_25pct_neighbor_inpainter/README.md) | `completed` — 25 dB not reached | per-FFID trace split / neighbor trace inpainter | Leakage-safe inpainter > 25 dB at 25% train traces per FFID |
| [study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter](study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/README.md) | `blocked` — 25 dB not reached, no promotable candidate | whole-FFID split / neighbor trace inpainter | Leakage-safe inpainter > 25 dB at 25% whole FFIDs as train |
| [study_021_all_ffid_50pct_whole_ffid_trace_graph](study_021_all_ffid_50pct_whole_ffid_trace_graph/README.md) | `running` | whole-FFID split / trace graph | Trace-node graph > 20 dB at 50% whole FFIDs, validation shots unobserved |
| [study_022_c3_na_pocs](study_022_c3_na_pocs/README.md) | `draft` | validation C3 volume / Fourier POCS-5D | How well does fixed hard-threshold Fourier POCS-5D reconstruct missing traces? |
| [study_023_c3_na_drr](study_023_c3_na_drr/README.md) | `draft` | validation C3 volume / damped rank-reduction 5D | How well does fixed damped rank-reduction reconstruct missing traces on the shared POCS benchmark? |
| [study_024_c3_na_siren_volume](study_024_c3_na_siren_volume/README.md) | `draft` | validation C3 volume / per-volume SIREN | How well does fixed-step observed-only SIREN internal learning reconstruct missing traces on the shared benchmark? |
| [study_025_c3_na_ccnet5d](study_025_c3_na_ccnet5d/README.md) | `draft` | supervised train-partition CCNet5D / frozen validation inference | How well does pretrained CCNet5D reconstruct the shared benchmark when additional complete train labels are used? |
| [study_026_grid_free_multi_relation_gnn](study_026_grid_free_multi_relation_gnn/README.md) | `draft` | masked train-partition trace GNN / arbitrary-coordinate queries | Do separate source, receiver, CMP and offset relations improve physical target reconstruction over matched graph controls? |
| [study_027_c3_na_benchmark](study_027_c3_na_benchmark/README.md) | `materialized_locked` | fixed C3 cases and shared train pool | Which immutable data contract supports the five-method comparison? |
| [study_028_c3_first_results](study_028_c3_first_results/README.md) | `first_results_complete` — GNN cross-run tolerance unmet | fixed validation / five-method pilot | What are the first full-volume results under finite declared budgets? |
| [study_029_c3_amplitude_qc](study_029_c3_amplitude_qc/README.md) | `qc_and_numerical_validation_complete` | raw SEG-Y provenance / amplitude QC / GNN numerical probe / SIREN rerun | Does source-bound QC restore a meaningful fixed training scale while preserving the evaluation cases? |
| [study_030_c3_siren_trace_scaling](study_030_c3_siren_trace_scaling/README.md) | `per_trace_scaling_evaluated` | observed per-trace RMS / interpolated physical scale | Does observed-only scale interpolation make per-trace SIREN useful in physical amplitude? |
| [study_031_c3_siren_10db](study_031_c3_siren_10db/README.md) | `fixed_shear_reference_adopted_no_shear_10db_unmet` — declared-shear SIREN reference adopted at 11.3422 dB; no-shear goal unmet | Cartesian SIREN / complete observed traces / fixed QC validation case | Physical target SNR with an explicitly recorded coordinate transform; preserve the [no-shear initialization and envelope comparison](../reports/c3_siren_time_learning_20260909.md) and [current adoption decision](../results/study_031_c3_siren_10db/20260909T055026000000Z_e39df16d563c_shear_reference_adoption/adoption_decision.json). |
| [study_032_c3_proposed_gnn_10db](study_032_c3_proposed_gnn_10db/README.md) | `final_5000_adopted_10db_achieved` — adopted 11.9354 dB; global RMS comparison 10.3033 dB; both audited | QC train-partition proposed relational GNN / fixed full validation | No fixed shear; relative MSE retained. Adopted observed trace RMS with interpolated gain; see the [adopted evaluation](../reports/c3_proposed_gnn_relative_mse_5k_20260909.md) and [global RMS comparison](../reports/c3_proposed_gnn_global_rms_relative_mse_5k_20260910.md). |
| [study_033_c3_ccnet_10db](study_033_c3_ccnet_10db/README.md) | `final_2048_adopted_10db_achieved` — final 11.6607 dB; saved-output and CPU audits passed | QC train-partition CCNet-5D / fixed full validation | No fixed shear; width 32, larger patches, batch two and complete-patch MSE. See the [final evaluation](../reports/c3_ccnet_context_2048_20260909.md). |
| [study_034_c3_nersi_baseline](study_034_c3_nersi_baseline/README.md) | `validation_complete_candidate_c_selected` — Candidate C 15.9189 dB; selected-run audit passed | per-volume observed-only profile-wise NeRSI repository reimplementation / fixed QC validation | Four predeclared candidates completed; Candidate C is the fixed-final validation primary. Test remains unused. See the [validation report](../reports/c3_nersi_validation_20260910.md). |

## Scratch workspaces

`study_all_ffid_temp` and `study_temp` are not numbered studies and are not immutable research records. They are overwriteable, throwaway workflows for informal runs and are excluded from the formal study index above. [`study_all_ffid_temp/README.md`](study_all_ffid_temp/README.md) describes its workspace; `study_temp` has no README.
