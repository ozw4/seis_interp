# FORGE random whole-shot50 への条件付き implicit field の移植

Status: `prepared` — 実行仕様を凍結した段階。本 study の結果は未確定。

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
