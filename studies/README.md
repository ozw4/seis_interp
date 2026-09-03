# Studies

Each numbered study directory is the authoritative record of one research question. Its `README.md` states the question, method, and recorded status; `config.yaml` holds the executable conditions; `inputs.yaml` holds the input contract; `decisions.md` exists only where important decision rationale is recorded. Start a new study from [`_template/`](_template/README.md).

Numbered studies are listed in ascending ID order. Gaps in the numbering (for example `study_002`) are real: those IDs were never created.

## Numbered studies

| Study | Recorded status | Scope / model | Research question |
|---|---|---|---|
| [study_001_c3_na_baseline](study_001_c3_na_baseline/README.md) | `draft` | single-FFID trace split / SIREN | Can a coordinate SIREN reconstruct held-out traces better than nearest-neighbor and inverse-distance baselines? |
| [study_003_omega0_sensitivity](study_003_omega0_sensitivity/README.md) | `active` | single-FFID trace split / SIREN | How do `omega_0` and learning rate affect validation convergence on FFID 2348? |
| [study_004_domain_scaling](study_004_domain_scaling/README.md) | `active` | single-FFID trace subsets / SIREN | How does training fit change as the number of training traces increases? |
| [study_005_correlation_loss_ablation](study_005_correlation_loss_ablation/README.md) | `completed` | eight-trace subset / SIREN | Does a trace-wise correlation auxiliary loss help the first failing Study 004 subset escape the near-zero predictor? |
| [study_006_batching_ablation](study_006_batching_ablation/README.md) | `completed` | eight-trace subset / SIREN | Is a 5,000-point random-replacement batch sufficient, or is exact coverage of all trace-time points required per update? |
| [study_007_full_ffid_large_batch](study_007_full_ffid_large_batch/README.md) | `completed` | full FFID 2348 / SIREN | Can one SIREN fit all 435 training traces under 5,000-point random-replacement batches and a 50,000-update budget? |
| [study_008_full_ffid_trace_batches](study_008_full_ffid_trace_batches/README.md) | `completed` | full FFID 2348 / SIREN | Can one SIREN fit all 435 training traces when every update uses eight uniformly selected complete traces? |
| [study_009_full_ffid_trace_batch_correlation](study_009_full_ffid_trace_batch_correlation/README.md) | `completed` | full FFID 2348 / SIREN | Does the Study 005 correlation loss help under the Study 008 complete-trace batches? |
| [study_010_full_ffid_temporal_patches](study_010_full_ffid_temporal_patches/README.md) | `completed` | full FFID 2348 / SIREN | Can one SIREN fit all 435 training traces using shared 64-sample temporal patches across 78 random traces? |
| [study_011_trace_pool_continuation](study_011_trace_pool_continuation/README.md) | `completed` | full FFID 2348 / SIREN | Can nested expansions of the training-trace pool, starting from the Study 006 subset, reach a strong 435-trace fit? |
| [study_012_official_siren_baseline](study_012_official_siren_baseline/README.md) | `completed` | full FFID 2348 / SIREN | Can the official SIREN parameterization escape the near-zero predictor under the Study 007 budget? |
| [study_013_amplitude_balancing](study_013_amplitude_balancing/README.md) | `completed` | full FFID 2348 / SIREN | Can per-trace RMS amplitude balancing escape the near-zero predictor under the Study 007 budget? |
| [study_014_full_trace_batch_ablation](study_014_full_trace_batch_ablation/README.md) | `completed` | full FFID 2348 / SIREN | Which ingredient of the first 435-trace escape—full complete-trace batches, correlation loss, or per-trace RMS—is necessary? |
| [study_015_strong_fit_budget_extension](study_015_strong_fit_budget_extension/README.md) | `completed` | full FFID 2348 / SIREN | Does the Study 014 recipe reach a strong fit when the update budget is extended? |
| [study_016_all_ffid_siren](study_016_all_ffid_siren/README.md) | `draft` | per-FFID trace split / SIREN | Can one SIREN trained across every amplitude-eligible FFID reconstruct traces held out within each FFID? |
| [study_017_all_ffid_neighbor_inpainter](study_017_all_ffid_neighbor_inpainter/README.md) | `completed` | per-FFID trace split / neighbor trace inpainter | Can a geometry-conditioned temporal network exceed 15 dB oracle global S/N using only training-trace neighbors? |
| [study_018_all_ffid_50pct_neighbor_inpainter](study_018_all_ffid_50pct_neighbor_inpainter/README.md) | `completed` | per-FFID trace split / neighbor trace inpainter | Can a leakage-safe inpainter exceed 20 dB when exactly half of the eligible traces in every FFID are training? |
| [study_019_all_ffid_25pct_neighbor_inpainter](study_019_all_ffid_25pct_neighbor_inpainter/README.md) | `completed` — the 25 dB threshold was not reached | per-FFID trace split / neighbor trace inpainter | Can a leakage-safe inpainter exceed 25 dB when exactly one quarter of the eligible traces in every FFID are training? |
| [study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter](study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/README.md) | `blocked` — the strict 25 dB threshold was not reached and no unchanged-scope candidate passes the evidence-backed promotion rule | whole-FFID split / neighbor trace inpainter | Can a leakage-safe inpainter exceed 25 dB when one quarter of the eligible FFIDs are assigned wholly to training? |
| [study_021_all_ffid_50pct_whole_ffid_trace_graph](study_021_all_ffid_50pct_whole_ffid_trace_graph/README.md) | `running` | whole-FFID split / trace graph | Can a trace-node graph network exceed 20 dB when half of the eligible FFIDs are training and every validation FFID is a completely unobserved shot? |

## Scratch workspaces

`study_all_ffid_temp` and `study_temp` are not numbered studies and are not immutable research records. They are overwriteable, throwaway workflows for informal runs and are excluded from the formal study index above. [`study_all_ffid_temp/README.md`](study_all_ffid_temp/README.md) describes its workspace; `study_temp` has no README.
