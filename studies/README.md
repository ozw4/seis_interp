# Studies

Each numbered study README is the authoritative record of its purpose, comparison conditions,
results, and decisions. Config and input files remain the executable condition sources. Files in
`reports/` contain only supplemental measurements or links back to the study record. The role of
each study file is defined in [`docs/repository_layout.md`](../docs/repository_layout.md). Start a
new study from [`_template/`](_template/README.md).

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
| [study_031_c3_siren_10db](study_031_c3_siren_10db/README.md) | `fixed_shear_reference_adopted_no_shear_10db_unmet` | Cartesian SIREN / complete observed traces / fixed QC validation case | Physical target SNR with a declared coordinate transform |
| [study_032_c3_proposed_gnn_10db](study_032_c3_proposed_gnn_10db/README.md) | `final_5000_adopted_10db_achieved` | QC train-partition relational GNN / fixed full validation | Observed-trace RMS scaling and relative-MSE training against global RMS |
| [study_033_c3_ccnet_10db](study_033_c3_ccnet_10db/README.md) | `final_2048_adopted_10db_achieved` | QC train-partition CCNet-5D / fixed full validation | Wider-context supervised CCNet-5D under the fixed benchmark |
| [study_034_c3_nersi_baseline](study_034_c3_nersi_baseline/README.md) | `validation_complete_candidate_c_selected` — Candidate C 15.9189 dB | observed-only profile-wise NeRSI / fixed QC validation | Four predeclared NeRSI candidates on the fixed validation case |
| [study_035_c3_ccnet_full_train_sampling](study_035_c3_ccnet_full_train_sampling/README.md) | `completed_degraded` | QC full-train patch sampling / CCNet-5D | Full-train patch sampling against the adopted Study 033 model |
| [study_036_c3_random80_observed_only_poc](study_036_c3_random80_observed_only_poc/README.md) | `stage_1_baseline_frozen` | fixed random-80 volume / five observed-only methods | Independent reconstruction under common inputs, target coverage, and compute reporting |
| [study_037_c3_neural_mse_loss_ablation](study_037_c3_neural_mse_loss_ablation/README.md) | `results_frozen` — MSE adopted for prior-research alignment | fixed Stage-1 inputs / three neural methods | MSE-only loss ablation with all other settings fixed |
| [study_039_c3_drr_target_tuning](study_039_c3_drr_target_tuning/README.md) | `target_achieved` — 12.1328 dB | DRR / target-informed parameter selection | Fixed study_036 inputs and full-target scoring; >11.3 dB achieved, not an independent evaluation |
| [study_040_c3_nersi_target_tuning](study_040_c3_nersi_target_tuning/README.md) | `target_unmet` — 64 verified single-model runs + 6 local-model candidates; physical best 12.3879 dB; normalized reference best 12.4088 dB | NeRSI / target-informed tuning, time alignment, EMA and gradient accumulation | Can NeRSI exceed 15 dB on the fixed target? Not an independent evaluation |

## Window-selection studies

| Study | Status | Scope |
|---|---|---|
| [study_041_c3_nersi_translated_window](study_041_c3_nersi_translated_window/README.md) | `target_reached` — mean trace SNR 15.6282 dB | Fixed source lines and shape; target-informed window selection, not original-window improvement |
| [study_042_c3_v3_five_method_comparison](study_042_c3_v3_five_method_comparison/README.md) | `running` | Fixed v3 inputs / five methods / physical mean trace SNR; unequal compute, target-informed |
| [study_043_c3_v3_ccnet_target_tuning](study_043_c3_v3_ccnet_target_tuning/README.md) | `completed` | v3 CCNet, Global RMS, no shear; target-informed mean trace SNR 14.4375 dB |
| [study_044_c3_v3_gnn_target_tuning](study_044_c3_v3_gnn_target_tuning/README.md) | `adopted_18db_achieved` | Fixed v3 GNN canonical EMA result: mean trace SNR 18.2795 dB; Global RMS, no shear; target-informed |

- [study_045_c3_v3_nersi_parameter_matched](study_045_c3_v3_nersi_parameter_matched/README.md): running — NeRSI width reduction to approximately match CCNet parameter count on fixed v3 inputs.

- [study_046_c3_v3_nersi_gnn_parameter_budget](study_046_c3_v3_nersi_gnn_parameter_budget/README.md): running — NeRSI at exactly 363,269 parameters, 50,000 updates, no shear, constant learning rate.

## Real-data EDA

- [study_047_forge_header_audit](study_047_forge_header_audit/README.md): full header audit complete — 1,106 SEG-Y files, 1,919,731 candidates; added ZIP members verified against extracted files.
- [study_048_forge_waveform_qc](study_048_forge_waveform_qc/README.md): full waveform audit complete — 35 all-zero and 100 nonzero constant traces; 1,919,596 numerically usable traces, component orientation unconfirmed.

- [study_049_forge_geometry_eda](study_049_forge_geometry_eda/README.md): full geometry EDA complete — 13 source lines, 1,106 source stations and 24 grid comparisons; benchmark region not selected.

- [study_050_forge_qc_review](study_050_forge_qc_review/README.md): review complete — fixed exclusions of 35 zero and 100 nonzero constant traces; distributions and 16 waveform examples; additional QC decisions pending.

- [study_051_forge_region_search](study_051_forge_region_search/README.md): search complete — 1,575 local grids, 1,200 source/receiver combinations, 15 region candidates; strict geometric guidelines unmet; benchmark adoption pending.

## Real-data comparison

- [study_052_forge_m1_mvp](study_052_forge_m1_mvp/README.md): prepared — frozen M1 input, random-80 mask, six method configs, observed-only preflight; production metrics pending.

- [study_053_forge_m1_trace_rms_idw](study_053_forge_m1_trace_rms_idw/README.md): follow-up — study_052 rerun with per-trace RMS normalization and projected-grid IDW amplitude restoration; designed after the study_052 test set was observed.

- [study_054_forge_regsi_conditions](study_054_forge_regsi_conditions/README.md): completed — ReGSI-real condition search over missing fraction and time sampling (14 conditions); the main-experiment condition is fixed (clean relative MSE 0.8788).

- [study_055_forge_amplitude_compression](study_055_forge_amplitude_compression/README.md): runs recorded — ReGSI-real without per-trace RMS, comparing linear scaling and asinh compression with an observed-only scale, plus an asinh scale sweep.

- [study_056_forge_2ms_random80_trace_rms](study_056_forge_2ms_random80_trace_rms/README.md): completed — all six methods predicted and evaluated; oracle trace RMS and matched neural update counts.

- [study_057_forge_2ms_random70_trace_rms](study_057_forge_2ms_random70_trace_rms/README.md): completed — six-method FORGE comparison at 70% missing, retaining the original hash order and all method settings; all methods predicted and evaluated. Follow-up tuning on the original test set (not an independent estimate) reached POCS 0.8381, DRR 0.8373 and CCNet5D 0.8175; NeRSI's best was 0.9212.

- [study_058_forge_2ms_random80_align500_matched20k](study_058_forge_2ms_random80_align500_matched20k/README.md): first matrix complete — six-method FORGE comparison on the study_056 mask with a shared 500 m/s moveout canvas; neural methods at 20,000 updates, 64 loss traces per update and about 363k parameters. POCS, DRR and ReGSI-real reach relative MSE ≤ 0.9; ReGSI-grid, CCNet5D and NeRSI do not.

- [study_059_forge_2ms_random70_align500_matched20k](study_059_forge_2ms_random70_align500_matched20k/README.md): completed — study_058 conditions on the study_057 70% mask; CCNet5D, NeRSI and ReGSI each use exactly 64 loss traces per update; CCNet5D patches span several lines so that every kernel tap is trained. All six methods reach relative MSE ≤ 0.9.

- [study_060_forge_fair_grid](study_060_forge_fair_grid/README.md): primary comparison completed — all searches, selections and 14 final predictions evaluated on mask B. Clean-test relative MSE: ReGSI-real 0.5779, ReGSI-grid 0.5884, CCNet5D 0.6526, POCS 0.7310, DRR 0.7754, NeRSI 0.8865. All six pre-registered contrasts favour ReGSI with cluster-bootstrap 95% CIs excluding zero and Holm-adjusted p = 0.0006. Density and 10,000-update sensitivity supplements remain pending.

- [study_061_forge_nersi_profile](study_061_forge_nersi_profile/README.md): completed — post-hoc re-run of NeRSI in the paper's profile-wise form on the re-sealed study_060 inputs with the study_060 search protocol. All three final seeds stayed at near-zero output (training loss ≈ 1.0 throughout): clean-test relative MSE 1.0001 ± 0.0001, worse than all six study_060 methods (exploratory contrasts, Holm p = 0.0006). Two of six seeds of the selected setting trained (validation 0.9697 / 0.9698).

- [study_063_c3_trace_grid](study_063_c3_trace_grid/README.md): partially stopped — C3 full-crop trace assignments, 10% observed validation holdout following study_060, five methods with 25 configurations and three additional assignment seeds.

- [study_064_c3_reference_grid](study_064_c3_reference_grid/README.md): running / queued — replacement ReGSI / CCNet5D / profile NeRSI 25-candidate grids with matched MSE, AdamW lr=1e-3, 50k updates, 64 traces and EMA; POCS / DRR continue from study_063.

- [study_065_forge_expanded_grid](study_065_forge_expanded_grid/README.md): stopped by user — fixed C3 S1 winners for five methods on the one-to-one FORGE extension [6,56,16,48], at 4 ms with 50,000 neural updates. NeRSI, POCS and DRR results are retained; CCNet5D training and the ReGSI retry queue are stopped. Source/receiver grid areas are 1.75/3 times the original, with 248,125 eligible traces and no collisions.

- [study_066_forge_c3_reference](study_066_forge_c3_reference/README.md): running CCNet / ReGSI queued — original study060 region [6,32,8,32] at 4 ms with the same trace assignment. CCNet uses patch [4,8,4,8], BF16 and 64-trace updates across independent full-time patches. ReGSI waits for GPU 0, using its original frozen code/config and fresh initialization because no intermediate checkpoint exists. Both target 50,000 optimizer steps; NeRSI, POCS and DRR completed.

- [study_067_forge_c3_common_grid](study_067_forge_c3_common_grid/README.md): prepared specification — C3 / FORGE 共通 5×5 grid、validation-only S1、全手法 barrier と 3-seed 評価。POCS は 16 反復 + receiver [4,8]、ReGSI は P2 V3（inner mask 0.25、node Fourier 0）を両 dataset に採用済み。

- [study_068_forge_shot50](study_068_forge_shot50/README.md): completed — study_067 の FORGE 選択設定を periodic whole-shot 50% へ移した固定設定5手法比較。標準5手法は全て完了。ReGSI 1.2758 dB、CCNet5D −0.5667 dB、profile NeRSI / POCS / DRR は approximately 0 dB。1 mask・1 model seed の記述的比較。

- [study_069_forge_random_shot50](study_069_forge_random_shot50/README.md): completed — 同じ選択設定を random whole-shot 50%（seed 7302）へ移した固定設定5手法比較。標準5手法は全て予測・採点を完了し、失敗・未実行は0件。ReGSI 2.5702 dB、CCNet5D 1.2157 dB、DRR 1.0033 dB、POCS 0.7105 dB、profile NeRSI は approximately 0 dB。periodic の study_068 とは独立した結果として保持する。

- [study_070_forge_c3_trace_field](study_070_forge_c3_trace_field/README.md): completed — FORGE の床 0.4 dB 未達、代表手法の置き換えは不採用。 study_067 と同一の入力・mask・validation holdout・予算・seed・選択規則で、新規のトレース出力 implicit field `inr_real` を共通 5×5 grid で測定。test 3 seed 平均は C3 11.3884 ± 0.0140 dB / relMSE 0.0941、FORGE 0.1011 ± 0.0073 dB / relMSE 1.0071 で、同じ test における study_067 の profile NeRSI（C3 12.5611 dB、FORGE 0.3126 dB）を両 dataset で下回る。FORGE は 25 候補すべてが validation relMSE > 1。縮退はしていない（相関中央値 0.2185）が補間に至らない。

- [study_071_forge_c3_context_field](study_071_forge_c3_context_field/README.md): prepared — study_070 の測定（同一 FORGE test 上で、観測トレースを読む4手法は相関中央値 0.4544–0.7275、重みのみの2手法は 0.2159–0.2600）を受けて、neural-implicit を観測近傍で条件付ける。近傍は field が読む4正規化座標だけで決まり、query は自分自身を文脈から除外する。時間表現は提案法と同じ共有 trace codec を使うため、パラメータ数が時間長に依存せず width 120 / 362,642 parameters が両 dataset で同一、宣言する手法差分は実行時に上書きされる `model.time_sample_count` の1件だけ。探索軸は近傍数 × Fourier 帯域。この baseline は予測時に観測データを読むため機構が提案法に近づくことを明示して報告する。比較結果は未確定。

## Scratch workspaces

`study_all_ffid_temp` and `study_temp` are not numbered studies and are not immutable research records. They are overwriteable, throwaway workflows for informal runs and are excluded from the formal study index above. [`study_all_ffid_temp/README.md`](study_all_ffid_temp/README.md) describes its workspace; `study_temp` has no README.

`study_forge_nersi_temp` is a scratch FORGE NeRSI parameter workspace, not a numbered study and not formal evidence. Each experiment writes an immutable timestamped run under `runs/study_forge_nersi_temp/`; its `summary.csv` is the overwriteable current output. [`study_forge_nersi_temp/README.md`](study_forge_nersi_temp/README.md) describes the knobs and commands.
