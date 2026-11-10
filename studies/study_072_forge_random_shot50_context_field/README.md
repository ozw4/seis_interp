# FORGE random whole-shot50 への条件付き implicit field の移植

Status: `completed` — 予測・採点を完了（失敗 0）。clean_target の relMSE は 1.0005 で、同一 mask の study_069 では ReGSI・CCNet5D・DRR・POCS の下、profile NeRSI の上。study_071 の同設定（random trace 70% で relMSE 0.7679）から大きく後退した。

> **後続による訂正（2026-11-10）**: 本 study の後退の原因を「gather 内近傍という情報の欠落」と帰属した判断は誤りだった。真の原因は学習と test の文脈不一致で、[study_075](../study_075_forge_random_shot50_gather_excluded/README.md) は手法設定の1行（`context.training_exclusion: source`）だけを変えて同一 mask で **1.6625 dB / relMSE 0.7490**（CCNet5D・DRR・POCS の上、6 手法中 2 位）を得た。本 README の測定値と幾何測定はそのまま有効だが、因果の帰属と「採用は random trace mask に限る」という記述は [decisions.md](decisions.md) の2026-11-10 節で撤回している。

## 目的

study_071 で選ばれた条件付き implicit field の設定を、各 source line 内で半数の shot をランダムに観測する
whole-shot 50% mask へそのまま移す固定設定比較。study_069 の5手法はこの mask の下で記録済みであり、
その中で profile NeRSI は相関中央値 approximately 0 の完全な縮退だった。

mask は study_069 と**byte 一致**させる。whole-shot mask の鍵には study ID が入るので、
`mask_study_id` で study_069 の名前空間を引き継ぐ（[`config.yaml`](config.yaml)）。
波形は study_069 と同じ study_067 FORGE run の sealed 入力を `physical_reference` で再利用し、
SEG-Y は読まない。

条件の正本は [`config.yaml`](config.yaml)、入力と選択元 hash は [`inputs.yaml`](inputs.yaml)。
手法設定 [`cinr_real.yaml`](cinr_real.yaml) は study_071 の
[base](../study_071_forge_c3_context_field/forge/cinr_real.yaml) に
[grid](../study_071_forge_c3_context_field/grid.yaml) の採用 cell `a05_b01`
（近傍 16 本、Fourier base 1.05）を機械適用したもので、手書きしない。
`selection_table` は study_071 の worker が出した候補別 summary CSV であり、
study_069 が参照した results 側の分析表にあたるものは study_071 ではまだ生成していない。

1 mask・1 model seed・5手法の記述的比較を1手法へ移すので、study_069 との対比は
**事後的・探索的**であり、両 study の事前登録対比群の外にある。

## 条件

study_069 から変えたのは `study_id`、`mask_study_id`、`methods`、`matched_training`、`workflow`、
`cuda_execution`、`boundary_handling`、`fixed_settings` だけで、候補領域・格子・時間軸・
`preprocessing`（`trace_oracle` と 500 m/s 整列）・`evaluation`・`visualization`・`selection`・
`training_loss`・`interpolation_mask`・`model_seed` は同一である。

config・inputs のいずれにも書けない値は次のとおり。

| 値 | |
|---|---:|
| 整列 canvas のサンプル数 | 2172 |
| パラメータ数 | 362,642（目標 363,269 比 −0.1726 %） |
| 近傍数 | 16 |
| admission（実測 8.578 GiB に C-0 規則） | 11 GiB |

## 結果

生成元 [run](../../runs/study_072_forge_random_shot50_context_field/20261109T232516173813Z_313fcc549ba5)、
[採用図表・CSV](../../results/study_072_forge_random_shot50_context_field/20261109T235235Z_313fcc549ba5_analysis)。
`dirty_worktree: false`、`total_runs: 1`、`test_scoring_calls: 1`、runtime 1460.3666 s、
peak CUDA reserved 7.1446 GiB（宣言 11 GiB）。入力9成果物は study_069 と byte 一致。

主評価 clean_target（24,397 trace）。表示は小数4桁。

| population | trace 数 | mean trace SNR (dB) | masked trace relative MSE |
| --- | ---: | ---: | ---: |
| clean_target | 24,397 | 0.3596 | 1.0005 |
| all_eligible | 24,576 | 0.3800 | 0.9984 |

shot 支持別（clean_target、shot 等重み）:

| shot_support | shot 数 | trace 数 | mean trace SNR (dB) | relMSE |
| --- | ---: | ---: | ---: | ---: |
| internal | 85 | 21,603 | 0.4196 | 0.9918 |
| edge | 11 | 2,794 | −0.1013 | 1.0665 |

### 同一 mask・同一 test 集合での対比（study_069 の記録）

| 手法 | mean trace SNR (dB) | relMSE |
| --- | ---: | ---: |
| ReGSI | 2.5702 | 0.6191 |
| CCNet5D | 1.2157 | 0.7898 |
| DRR | 1.0033 | 0.8192 |
| POCS | 0.7105 | 0.8901 |
| **cinr_real** | **0.3596** | **1.0005** |
| profile NeRSI | approximately 0 | 1.0000 |

### 近傍の測定（幾何のみ、波形を読まない）

study_071 の random trace 70% mask と本 study の random shot 50% mask で、test trace ごとに
採用設定と同じ 16 近傍を選び直した結果。

| | study_071 random trace 70% | study_072 random shot 50% |
| --- | ---: | ---: |
| 観測 trace 数 | 13,271 | 24,575 |
| 最近傍距離（中央値） | 0.03228 | 0.03229 |
| 第16近傍距離（中央値） | 0.14417 | 0.11636 |
| 同一 shot gather 内の近傍数（16 本中） | **2.63** | **0.00** |
| 同一 shot の近傍を1本以上持つ query の割合 | **0.9597** | **0.0000** |

本 study は観測 trace が約1.85倍あり最近傍距離も同じだが、同一 shot gather 内の観測 trace は
構造上ゼロである。

### 採否

本 mask では `cinr_real` を代表手法として採用しない。study_071 の random trace mask における採用
（[study_071 README](../study_071_forge_c3_context_field/README.md)）は維持する。
判断理由と帰結は [decisions.md](decisions.md) に置く。

## 実行

```bash
PY=.venv/bin/python
"$PY" scripts/forge_trace_experiment.py prepare --study studies/study_072_forge_random_shot50_context_field
"$PY" scripts/forge_trace_experiment.py launch  --experiment runs/study_072_forge_random_shot50_context_field/<run-id>
"$PY" scripts/forge_trace_experiment.py score   --experiment runs/study_072_forge_random_shot50_context_field/<run-id>
"$PY" scripts/forge_trace_experiment.py figures --experiment runs/study_072_forge_random_shot50_context_field/<run-id> \
  --output results/study_072_forge_random_shot50_context_field/<result-id>
```

prepare 後、launch の前に `input/partition.parquet` の sha256 が study_069 の
[採用 run](../../runs/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0) の記録と
一致することを確認する。

1手法なので `paired_shot_differences.csv` は空になる（`itertools.combinations` が空）。
study_069 の `per_shot.csv` との shot 別 paired 差は採点後の分析段で作る。
test は method の状態が確定した後にだけ採点する。図の shot は学習前に geometry だけで固定する。
