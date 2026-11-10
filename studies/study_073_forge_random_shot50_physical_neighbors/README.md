# FORGE random whole-shot50 での物理スケール近傍

Status: `prepared` — 実行仕様を凍結した段階。本 study の結果は未確定。

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
