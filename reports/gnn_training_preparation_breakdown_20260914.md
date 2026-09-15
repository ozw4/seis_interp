# GNN training preparation breakdown and model-side revalidation — 2026-09-14

## Environment and fixed conditions

- Commit: `684c4e4c659a8b5435a3edb514962b2582645db8`（worktreeの実装変更を含む）。
- GPU: NVIDIA H100 NVL、`cuda:1`、UUID `GPU-5343d08b-0556-2774-6d44-7e8dc8455690`。PyTorch `2.5.0a0+b465a5843b.nv24.09`、CUDA `12.6`。
- Source: `studies/study_044_c3_v3_gnn_target_tuning/mask10_5k.yaml`の解決内容。既存のformal config・runは編集せず、resolvedをscratchへコピーした。
- 差分は`training.max_steps` 5000→200、`training.report_interval` 100→20、`prediction.query_batch_size` 16→64のみ。予測は本作業の対象外であり、対照として使うため実行時間だけを下げた。
- 各200 updates、query batch 64、`inner_mask_fraction` 0.1（42 step/episode、5 episode）、warm-upは最初の20 step。
- Model: width64、2 message-passing rounds、time downsample2、learned gate、101,701 parameters。初期化seed101、episode seed201。MSE、observed global RMS、AdamW、learning rate0.001、weight decay0、gradient clip1。
- `mixed_precision: off`、`edge_sampling`なし、K=2。`scripts/run_c3_v3_gnn_tuning.sh`と同じ環境（OMP/MKL/OPENBLAS各1 thread、`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`）で、同一processからA→Bを逐次実行した。
- 実行時、他processによるGPU負荷はなかった。これは一回の測定であり、区間ごとの追加CUDA同期は行っていない相対timingである。

Variantの差は1つだけである。

| Variant | `training.revalidate_model_inputs` |
|---|---|
| A | 省略（従来どおりmodel側でも再検証する） |
| B | `false` |

## Results

各値は全200 stepの集計。秒はstepあたり。機械可読JSONは`runs/gnn_training_speed_scratch/20260914T091302Z_684c4e4c659a_d1b1/comparison.json`にあり丸めていない。

| Variant | updates/s | Training s | Step s (warm-up除く) | Prep s | Optim s | Mean nodes / edges | Peak allocated GiB | Prediction s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2.3700 | 84.3886 | 0.4038 | 0.2938 | 0.1222 | 1616.4600 / 3158.4000 | 61.0606 | 201.6459 |
| B | 2.3660 | 84.5305 | 0.4053 | 0.2962 | 0.1218 | 1616.4600 / 3158.4000 | 61.0606 | 198.2879 |

### 準備時間の内訳（本測定で新規に取得）

| Variant | query geometry s | graph build s | input assembly s | label read s | Prep合計 s |
|---|---:|---:|---:|---:|---:|
| A | 0.0002436 | 0.2794879 | 0.0138833 | 0.0002070 | 0.2938218 |
| B | 0.0002220 | 0.2822640 | 0.0134560 | 0.0001670 | 0.2961590 |

Aでは準備時間がstepの72.8%、graph buildが準備時間の95.1%、step全体の69.2%である。
query geometryとlabel readは合わせてstepの0.11%であり、input assemblyは3.4%である。

### episode内位置別のgraph build（秒、平均）

| pos in episode | n | A graph build | B graph build | A optim | B optim |
|---|---:|---:|---:|---:|---:|
| 0 | 5 | 0.3512 | 0.3531 | 0.2507 | 0.2483 |
| 1-4 | 20 | 0.3444 | 0.3516 | 0.1157 | 0.1149 |
| 5-14 | 50 | 0.3144 | 0.3182 | 0.1470 | 0.1445 |
| 15-41 | 125 | 0.2523 | 0.2540 | 0.1082 | 0.1087 |

## 数値の実行間差

CUDAでは同一条件の再実行でもlossが一致しない。training-only controlを同一process・同一seedで3回実行した
（`determinism_control.py`、60 updates）。

| 比較対象 | baselineと一致したstep | 最大絶対差 | 最終stepの差 |
|---|---:|---:|---:|
| baseline再実行 | 2/60 | 4.061e-05 | 5.959e-06 |
| `revalidate_model_inputs: false` | 2/60 | 4.866e-05 | 6.870e-06 |

`revalidate_model_inputs: false`の差は、同一コードの再実行同士の差と同程度である。
したがってA/Bのloss差をこのoptionの効果として扱わない。CPUでの同一性は
`tests/unit/test_relational_trace_graph_poc.py`が固定している。

## Exact-index searchの最適化（同日、A→C→D）

準備時間の内訳が取れたので、graph buildをcProfileで分解した（`builder.build()`を30回、episode cacheを10 step分温めた後）。
`_box_rows`と`_relation_distances`が合計cumtimeの8割を占め、次の3点が無駄であった。

1. `_conservative_box_width`が72,472回呼ばれていた。戻り値は`radius`とscaleだけの関数であり、run中は定数である。
2. `_distances`が`_relation_distances`経由で4 relation分の距離を計算し、3列を捨てていた。
3. `np.errstate`がaxisごとに入り、`seterr`/`geterr`が869,664回発生していた。

これを次のように変更した。いずれも浮動小数の演算列を変えない。

- C: box widthをindex構築時に1回だけ計算する。`np.errstate`を`_box_rows`あたり1回に集約する。
  `_distances`はそのrelationの座標対だけを計算する。
- D: Cに加えて、最短範囲を与えたbound自身の再filterを省く（その範囲は定義上その boundを満たす）。
  axisごとの連続1-D列を保持してfilterのgatherを1-Dにする。長さ2のaxis和を`a*a + b*b`に置き換える。

### 結果（同一条件、200 updates）

| | A（変更前） | C | D | D vs A |
|---|---:|---:|---:|---:|
| Training s | 84.3886 | 66.6878 | 59.3125 | -29.7% |
| updates/s | 2.3700 | 2.9990 | 3.3720 | +42.3% |
| Step s（warm-up除く） | 0.4038 | 0.3131 | 0.2661 | -34.1% |
| Prep s | 0.2938 | 0.1920 | 0.1457 | -50.4% |
| graph build s | 0.2795 | 0.1769 | 0.1310 | -53.1% |
| input assembly s | 0.0139 | 0.0145 | 0.0142 | +2.5% |
| Optim s | 0.1222 | 0.1327 | 0.1297 | +6.1% |
| Mean nodes / edges | 1616.46 / 3158.40 | 1616.46 / 3158.40 | 1616.46 / 3158.40 | 一致 |
| Prediction s | 201.6459 | 157.8170 | 118.6710 | -41.1% |
| Training + prediction s | 286.0345 | 224.5049 | 177.9835 | -37.8% |

cProfileでのgraph build合計は10.479 s→4.329 s（-58.7%、30 build）。

予測にも同じ検索を使うため、予測時間も-41.1%である。
optimization時間が+6.1%になっているのは、trainerのdocstringが述べる計測上の性質による。
準備が速くなると準備区間中に完了するdevice処理が減り、その分が後続のoptimization区間に計上される。
stepの合計は-34.1%である。

### 同一性の検証

- 4 relationすべてでbox widthが従来式と一致する。
- 実C3 geometryの200通りのrow集合×4 relationで、距離が4列形式の該当列とbit一致する（不一致0）。
- 実C3 geometryで、exact indexのplanがbrute forceのplanとbit一致する（`trace_ids`、`edge_index`、
  `edge_type`、`edge_distances`、`degree`、`min_distance`、`depth`、`query_indices`、diagnostics）。
- subgraphのnode数・edge数はA/C/Dで完全に一致する。
- 既存のexact-index≡brute-force同値testと、`tests/unit/test_trace_graph_spatial_index.py`へ追加した
  不変条件testが上記2つの近道を固定する。

### 残っている項目

Dの時点でgraph buildはstepの49.2%であり、依然として最大項である。cProfileでは`_box_rows`が
残りのtottimeの37%を占める。これ以上はdestination方向のvectorize（`_select`の
`for destination × for relation`のPython二重loopの解体）が必要であり、本作業では行っていない。

## Interpretation

- 準備時間の内訳はgraph buildにほぼ集中する。以後の準備時間の最適化対象はgraph buildであり、
  input assembly、query geometry、label readは対象にしない。
- graph buildはepisode内で0.3512→0.2523まで下がるが、28.2%の低下にとどまる。
  observed-neighbor cacheが温まった後も残るコスト（毎stepのhop-0 query検索とclosure組み立て）が支配項である。
- `revalidate_model_inputs: false`のtraining時間差は+0.17%で、optimization時間差は-0.6%である。
  本条件では効果を検出していない。stepの69%がCPUのgraph buildで、GPUの投入待ちが常に解消している状態のため、
  device同期の除去が待ち時間を短縮しない状況と整合する。この結果を受けてoptionは実装ごと取り下げ、
  model forwardの入力検証は従来どおり常に実行する。
- 対照として置いた予測時間は201.6459 sと198.2879 sで、-1.7%である。学習側の差と同じ大きさであり、
  本測定の分解能はこの程度である。
- subgraphのnode数・edge数はA/Bで完全に一致し、peak allocatedも一致する。graph planはoptionに依存していない。
- FP32のpeak allocatedは61.0606 GiBであり、steady-state activationのmemory値としては解釈しない。
