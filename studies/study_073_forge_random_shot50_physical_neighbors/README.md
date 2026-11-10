# FORGE random whole-shot50 での物理スケール近傍

Status: `completed` — 予測・採点を完了（失敗 0）。物理スケールの順位付けは study_072 の index 正規化に対し **clean_target で −0.1684 dB / relMSE +0.0648 と劣化**した。改善しないという事前の予測は当たり、さらに小幅な劣化であることが確定した。`context.distance` の既定は `index` のまま維持する。

## 目的

study_072 からの変更は**1行だけ**である。近傍の順位付けを index 正規化座標から
取得幾何の物理座標（source / receiver の station 位置、m）へ変える
（`context.distance: physical`）。mask・入力・model・近傍数 16・Fourier base 1.05・
予算・seed はすべて study_072 と同一で、[`cinr_real.yaml`](cinr_real.yaml) は study_072 の
[手法設定](../study_072_forge_random_shot50_context_field/cinr_real.yaml) に
`context.distance: physical` を加えただけである。field が読む座標は変えない。順位付けだけを変える。

study_072 の近傍定義は4つの index 軸を `max(spatial_shape − 1, 1)` で割った等方距離であり、
「近い」の重みが station 間隔ではなく軸の index 本数で決まっていた。本 study はそれを物理間隔に置き換える。

## 事前に測定した予測

幾何だけの測定（波形を読まない）によれば、FORGE では両者がほぼ同じ近傍を選ぶ。

| 軸 | 物理ステップ | index 重み | 物理比（point 基準） | index 比 |
|---|---:|---:|---:|---:|
| source line | 372.31 m | 0.20000 | 7.45 | 6.20 |
| source point | 49.96 m | 0.03226 | 1.00 | 1.00 |
| receiver line | 200.01 m | 0.14286 | 4.00 | 4.43 |
| receiver point | 50.01 m | 0.03226 | 1.00 | 1.00 |

この mask の 24,576 test trace で 16 近傍を両方の順位付けで選び直すと:

| | 値 |
|---|---:|
| 16 近傍の一致数（平均） | 15.72 / 16 |
| 近傍集合が完全一致する query | 80.55 % |
| 最近傍が一致する query | 100.00 % |
| 同一 shot gather 内の近傍（index / physical） | 0.00 / 0.00 |
| 同一受振点の近傍（index / physical） | 3.31 / 3.29 |
| offset 差の中央値（index / physical） | 77.09 m / 76.68 m |

index 正規化が偶然物理異方性に近いため選択はほとんど動かず、study_072 の後退の構造的原因
（whole-shot mask では同一 gather 内の観測 trace が存在しない）は順位付けの重みでは変わらない。
したがって本 study は**改善を予想しない**。index 距離の異方性が格子の index 本数由来であるという
弱点に対して、物理スケールが FORGE でどれだけの効果量を持つかを確定させるための測定である。

## 結果

生成元 [run](../../runs/study_073_forge_random_shot50_physical_neighbors/20261110T001537787611Z_0c98af748471)、
[採用図表・CSV](../../results/study_073_forge_random_shot50_physical_neighbors/20261110T004234Z_0c98af748471_analysis)。
`dirty_worktree: false`、`total_runs: 1`、`test_scoring_calls: 1`、runtime 1450.9284 s。
run は `context_distance: physical` を記録している。入力10成果物は study_069 / study_072 と byte 一致。

### A/B（study_072 との差は近傍の順位付けだけ）

| | 順位付け | clean_target SNR (dB) | clean_target relMSE | all_eligible SNR (dB) | all_eligible relMSE | 学習 loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| study_072 | index | **0.3596** | **1.0005** | 0.3800 | 0.9984 | 0.2043 |
| study_073 | physical | 0.1912 | 1.0652 | 0.2125 | 1.0633 | 0.2038 |
| 差（physical − index） | | **−0.1684** | **+0.0648** | −0.1675 | +0.0649 | −0.0005 |

学習 loss はほぼ同一（0.2038 / 0.2043）であり、観測トレースへの適合は同等で、差は汎化だけに現れる。

### shot 支持別（clean_target）

| shot_support | shot 数 | index SNR (dB) | physical SNR (dB) | index relMSE | physical relMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| internal | 85 | +0.4193 | +0.3283 | 0.9919 | 1.0287 |
| edge | 11 | −0.1023 | **−0.8687** | 1.0667 | **1.3476** |

劣化は edge shot（source 軸の支持が片側だけの shot）に集中する。

### 両順位付けが選ぶ近傍の差（幾何のみ）

| query 群 | 件数 | 16 近傍の一致数 | 集合が完全一致 | 最近傍が一致 |
| --- | ---: | ---: | ---: | ---: |
| 学習 query（観測 → 観測） | 24,575 | 15.75 / 16 | 82.62 % | 98.27 % |
| test internal shot | 21,760 | 15.76 / 16 | 83.88 % | 100.00 % |
| test edge shot | 2,816 | 15.37 / 16 | **56.46 %** | 100.00 % |

不一致は edge shot で 43.5 % と、internal shot の 16.1 % に対し約 2.7 倍である。

### 採否

`context.distance` の既定は `index` のまま維持し、`physical` は採用しない。
判断理由は [decisions.md](decisions.md) に置く。

## 条件

条件の正本は [`config.yaml`](config.yaml)、入力と選択元 hash は [`inputs.yaml`](inputs.yaml)。
mask は `mask_study_id` で study_069 の名前空間を引き継ぎ、波形は study_067 FORGE run の
sealed 入力を `physical_reference` で再利用する。`inputs.yaml` は study_069 と study_072 の
記録（`experiment.json` / `summary.csv` / `per_shot.csv`）も対比対象として pin する。

1 mask・1 model seed・1手法の記述的比較であり、study_072 / study_069 との対比は事後的・探索的である。

## 実行

```bash
PY=.venv/bin/python
"$PY" scripts/forge_trace_experiment.py prepare --study studies/study_073_forge_random_shot50_physical_neighbors
"$PY" scripts/forge_trace_experiment.py launch  --experiment runs/study_073_forge_random_shot50_physical_neighbors/<run-id>
"$PY" scripts/forge_trace_experiment.py score   --experiment runs/study_073_forge_random_shot50_physical_neighbors/<run-id>
"$PY" scripts/forge_trace_experiment.py figures --experiment runs/study_073_forge_random_shot50_physical_neighbors/<run-id> \
  --output results/study_073_forge_random_shot50_physical_neighbors/<result-id>
```

prepare 後、launch の前に `input/partition.parquet` の sha256 が study_069 / study_072 と
一致することを確認する。
