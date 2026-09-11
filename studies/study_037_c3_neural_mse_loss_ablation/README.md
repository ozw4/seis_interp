# Stage-1b Neural MSE Loss Ablation

Status: `planned`。これは探索的Stage-1b loss ablationである。Stage-1 baselineを置換しない。

## 比較条件

固定target集合ではphysical-amplitude global SNRの最大化はtarget SSEの最小化と等価。
同じO-only Global RMSで正規化した振幅の通常MSEは物理振幅SSEと定数倍の関係にある。
trace-energy weightingを除くことでglobal SNRが改善するかを調べる。

変更はNeRSI・CCNet-5D・Proposed GNNの `training.loss` のみ：
`masked_trace_relative_mse` → `masked_trace_mse`。
formal YAMLはstandaloneで、[Stage-1 formal](../study_036_c3_random80_observed_only_poc/formal/)と
resolved mappingがloss以外完全一致する。全3手法の5000 updates、学習率0.001、report interval 100、
device cuda:1、architecture、全seed、optimizer、sampling、batch/patch、graph、inner mask率、
prediction、O-only Global RMS、target-only physical evaluator、full-target coverage、
observed traceのexact reinsertion、final checkpoint roleを固定する。
smokeは2 updates / report interval 1だけを変更し、そのmetricを性能比較・採否判断に使用しない。

(T) はすでにStage-1で参照済みなので、今回の結果だけで正式lossを採択しない。
正式採択は将来のStage Aでvalidation targetを使って行う。
改善・悪化に関係なく全3runを保存する。
test SNRを見たtraining延長やlearning-rate変更を行わない。
発散した場合は「同一learning-rate条件では不安定」と記録する。
HPO、validation selection、複数seed、自動retry、early stopping、best-checkpoint selectionは行わない。
POCS・DRRは対象外であり変更・再実行しない。

## 固定入力と比較元

nominal 80% random mask、crop内のrealized missing fractionは79.8874%

exact fraction: `104710 / 131072 = 0.7988739013671875`。
O=26362、T=104710、shape=`[384, 16, 32, 8, 32]`、
observed Global RMS=`9.257964353671534`。

```yaml
benchmark_id: c3_sl25_40_random80_observed_only_v1
dataset_id: seg_c3_na
case_id: c3_benchmark_test_random_trace_80_seed42
volume_id: c3_benchmark_test_random_trace_80_seed42_volume
selection:
  time: [0, 384]
  source_line: [25, 41]
  shot_in_line: [28, 60]
  relative_receiver_x: [0, 8]
  relative_receiver_y: [18, 50]
```

比較元はStudy 036の `20260911T081726Z_466b41fae15c_formal`。
[凍結lock](../study_036_c3_random80_observed_only_poc/stage_1_baseline.lock.json)を正本とする。
以下はrelative-loss baselineの記録（表示は小数4桁）。新runの合否閾値ではない。

| Method | SNR [dB] | RMSE | Relative L2 |
|---|---:|---:|---:|
| NeRSI | 10.2829 | 2.8272 | 0.3061 |
| CCNet-5D | 6.5988 | 4.3208 | 0.4678 |
| Proposed GNN | 9.2729 | 3.1758 | 0.3438 |

## 実行

リポジトリrootのBashで、依存関係をインストール済みのPython環境を使用する。
GPU 1の利用可能メモリを事前確認する。deviceの変更は単一要因条件を崩すため行わない。
`CUDA_VISIBLE_DEVICES`で番号を再割当てしない。各手法を別processで逐次実行し、同じGPUで並列実行しない。
`poc run-all`はPOCS・DRRまで再実行するため使用しない。
既存run directoryは再利用しない。

### 環境とpreflight

```bash
cd /workspace
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

study=studies/study_037_c3_neural_mse_loss_ablation
qc=data/processed/c3_na/study_029_c3_amplitude_qc
case_id=c3_benchmark_test_random_trace_80_seed42
poc_inputs=(
  --interim data/interim/c3_na/all_ffids
  --processed "$qc/partition"
  --mask "$qc/masks/$case_id"
  --case "$qc/cases/$case_id"
  --volume "$qc/volumes/$case_id"
)
python -m seis_interp.cli poc check "${poc_inputs[@]}" --json
```

preflight成功を確認してからsmokeへ進む。

### 実データsmoke

```bash
run_root="runs/study_037_c3_neural_mse_loss_ablation/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_smoke"

python -m seis_interp.cli interpolate nersi \
  "${poc_inputs[@]}" --config "$study/smoke/nersi.yaml" \
  --output "$run_root/nersi" --json

python -m seis_interp.cli interpolate ccnet5d \
  "${poc_inputs[@]}" --config "$study/smoke/ccnet5d.yaml" \
  --output "$run_root/ccnet5d" --json

python -m seis_interp.cli interpolate relational-trace-graph \
  "${poc_inputs[@]}" --config "$study/smoke/gnn.yaml" \
  --output "$run_root/relational_trace_graph" --json
```

3件のstatus・metadata・coverageを確認する。smoke metricから設定を選択しない。

### Formal

```bash
run_root="runs/study_037_c3_neural_mse_loss_ablation/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_formal"

python -m seis_interp.cli interpolate nersi \
  "${poc_inputs[@]}" --config "$study/formal/nersi.yaml" \
  --output "$run_root/nersi" --json

python -m seis_interp.cli interpolate ccnet5d \
  "${poc_inputs[@]}" --config "$study/formal/ccnet5d.yaml" \
  --output "$run_root/ccnet5d" --json

python -m seis_interp.cli interpolate relational-trace-graph \
  "${poc_inputs[@]}" --config "$study/formal/gnn.yaml" \
  --output "$run_root/relational_trace_graph" --json
```

失敗時も成功済みrunと失敗ログを保持し、残りの手法を実行する。別設定への切替・自動retryは行わない。

### 完了後の比較

```bash
python "$study/compare.py" "$run_root"
```

[compare.py](compare.py)は全3runのsuccess、full-target coverage、prediction/checkpoint hash、
全input lockの凍結baselineとの完全一致、loss metadata、5000 updates、
保存configのloss以外完全一致を検査してからCSVを標準出力へ表示する。
baseline artifactも凍結hashで検査する。振幅配列やtarget truthをロードせず再評価・再学習しない。

各手法についてbaseline/MSE/deltaのSNR・RMSE・mean_trace_relative_mse、
training time、prediction time、peak GPU memoryを表示する。deltaはMSE−baseline。
SNRが未定義の場合は空欄とstatusを表示する。表示丸め値から閾値判断を行わない。
学習lossをMSEに変更してもsecondary評価指標mean_trace_relative_mseは維持する。
