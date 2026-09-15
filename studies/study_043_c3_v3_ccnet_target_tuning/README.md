# v3 CCNet target-informed tuning

固定v3入力の全Tに対するphysical-amplitude trace SNR平均で14 dB超を目指す。
O-only Global RMS、MSE、Shearなし。Tは学習ラベルに使用しないが、
試行選択に参照するため独立testではない。既存の比較runを上書きしない。

入力はStudy 042のinputs.yamlとv3 condition lockを使用する。
mask10_5k.yamlはCCNetのinner maskを10%とし、他の設定を維持する。
外側の欠損率・配置・評価対象は変更しない。

mask10_20k.yamlは同じモデル・学習率・patchで20,000 updateを実行する。
重みは初期化seed 101から学習し、checkpoint再開は行わない。

| Config | Mean trace SNR (dB) | Global SNR (dB) | RMSE |
|---|---:|---:|---:|
| mask10_5k | 13.1816 | 12.5600 | 2.1824 |
| mask10_20k | 14.4375 | 13.8567 | 1.8797 |

結果run: `runs/study_043_c3_v3_ccnet_target_tuning/20260914T012855Z_03c2fc83df18_mask10_5k/ccnet5d`。

達成run: `runs/study_043_c3_v3_ccnet_target_tuning/20260914T013304Z_03c2fc83df18_mask10_20k/ccnet5d`。
全104,750 target traceのSNR平均は14.437486631276457 dBで、14 dB超を達成。
Global SNRは14 dB未満。O-only Global RMSは9.27911442626805、Shearなし、
全target coverage、Oのexact reinsertionを維持する。保存predictionの再採点はmetrics.jsonと一致。
これはTを参照した探索結果であり、凍結済みv3比較の正本を自動置換しない。

## Exact 64-trace AdamW experiment

[`mask10_ema0999_adamw_trace64_50k.yaml`](mask10_ema0999_adamw_trace64_50k.yaml)
uses AdamW and exactly 64 supervised trace presentations per update, as explicitly
requested by the user. It inherits the EMA 50k model, input lock, patch shape,
inner mask, learning rate and seeds. Weight decay is zero, matching the GNN.
There are 3,200,000 supervised presentations over the fixed training budget.

Each update stacks enough sampled patches to supply 64 hidden traces, then
uniformly selects exactly 64 hidden entries across the entire stacked batch.
Every patch keeps its original inner mask; unselected hidden entries stay hidden.
Overlapping patches can present the same physical trace more than once. These
are presentations, not a guarantee of 64 distinct physical trace IDs. The loss
uses all 384 time samples of each selected trace. There is one forward, one
backward and one optimizer step per update, with no gradient accumulation.
The selection RNG is independent of placement and masking and uses the inner
mask seed. Cumulative supervised presentations are recorded in the loss history.
Optimizer and batch semantics are recorded in resolved config and run metadata;
the final inference checkpoint retains the established EMA format.

```bash
bash scripts/run_c3_v3_ccnet_tuning.sh mask10_ema0999_adamw_trace64_50k
```

## GNN parameter-budget candidate

[`mask10_ema0999_adamw_trace64_param363k_50k.yaml`](mask10_ema0999_adamw_trace64_param363k_50k.yaml)
is prepared, not yet run. It changes only the two channel widths from the
exact-64 AdamW candidate. Its 363292 trainable parameters exceed the GNN/NeRSI
budget of 363269 by 23 (0.0063%); this is an approximate, not exact, match.
For the existing four-block kernel-3 architecture the count is
`108 * hidden * intermediate + 40 * intermediate + 3 * hidden + 1`.
No positive integer channel pair gives exactly 363269. The candidate retains
the architecture without dummy parameters or changes to convolution biases.

The inherited training, input and evaluation contract remains fixed. It starts
from initialization and evaluates the final EMA weights in `final.pt`.
Parameter matching does not match FLOPs or the GNN's 20000-update budget.
After completion, verify input/artifact hashes, full target coverage, exact O
reinsertion, 3200000 supervised presentations, memory and timing before comparison.

```bash
cd /workspace
bash scripts/run_c3_v3_ccnet_tuning.sh mask10_ema0999_adamw_trace64_param363k_50k
```

## EMA results

20,000 updatesのEMA結果はtrace SNR平均15.4961 dB、global SNR 14.7970 dB、
RMSE 1.6869。EMAなしに対してtrace SNR平均は+1.0586 dB。
結果runは `runs/study_043_c3_v3_ccnet_target_tuning/20260915T002456Z_684c4e4c659a_mask10_ema0999_20k/ccnet5d`。
全target coverage、入力lock一致、成果物hashを確認済み。

[mask10_ema0999_50k.yaml](mask10_ema0999_50k.yaml) はEMA最良条件から
更新数だけ50,000に変更する。今回のユーザー指定による予算拡大であり、
既存の20,000更新runは保持する。optimizer状態を持たないfinal checkpointからの
継続ではなく、同じseedで初期化して50,000更新を実行する。
実行中runは `runs/study_043_c3_v3_ccnet_target_tuning/20260915T003602Z_684c4e4c659a_mask10_ema0999_50k/ccnet5d`。

[mask10_ema0999_20k.yaml](mask10_ema0999_20k.yaml) は現在のベスト
`mask10_20k.yaml` に `training.ema_decay: 0.999` だけを追加する。
最初のoptimizer更新後の重みでEMAを初期化し、以降の更新ごとに
`average = decay * average + (1 - decay) * weight` を適用する。
学習中はraw重みを使い、固定20,000 updatesの終了後にEMA重みで推論・保存する。
`final.pt` の `ema` とmetadataの `training_or_reconstruction` にdecayと最終EMA使用を記録する。
EMA未指定またはnullなら従来どおりraw重みを使用する。既存runは変更しない。

```bash
cd /workspace
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
unset CUDA_VISIBLE_DEVICES
qc=data/processed/c3_na/study_029_c3_amplitude_qc
window=data/processed/c3_na/study_041_c3_nersi_translated_window/shot18_ry18
run_root="runs/study_043_c3_v3_ccnet_target_tuning/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_mask10_ema0999_20k"
.venv/bin/python -m seis_interp.cli interpolate ccnet5d \
  --interim data/interim/c3_na/all_ffids \
  --processed "$qc/partition" \
  --mask "$qc/masks/c3_benchmark_test_random_trace_80_seed42" \
  --case "$window/case" --volume "$window/volume" \
  --config studies/study_043_c3_v3_ccnet_target_tuning/mask10_ema0999_20k.yaml \
  --output "$run_root/ccnet5d" --json
```
