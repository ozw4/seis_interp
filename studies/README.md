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

## Scratch workspaces

`study_all_ffid_temp` and `study_temp` are not numbered studies and are not immutable research records. They are overwriteable, throwaway workflows for informal runs and are excluded from the formal study index above. [`study_all_ffid_temp/README.md`](study_all_ffid_temp/README.md) describes its workspace; `study_temp` has no README.
