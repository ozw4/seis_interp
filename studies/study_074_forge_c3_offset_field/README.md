# study_074: C3 / FORGE 共通 25 候補による offset 座標付き条件付き field

Status: `stopped by user` — 2026-11-10 にユーザー判断で停止。S1 の数候補を走らせた段階で完了せず、結果は未確定のまま確定しない。幾何だけの測定（[decisions](decisions.md)）で、shot50 では offset 軸が近傍をほとんど動かさないこと（重み 4 でも 16 近傍のうち 13.68 本が不変）と、より大きな原因（学習と test の文脈不一致）が特定されたため、[study_075](../study_075_forge_random_shot50_gather_excluded/README.md) を優先した。仕様と事前予測は記録として保持する。

## 目的

study_071 で採用した条件付き field に **log-offset 座標**を加える。測定が指す支配項に直接当てる。

FORGE では、最短オフセット quintile だけが崩れており、そこが全エネルギーを持つ。

| FORGE test（study_071、seed 7321） | 相関中央値 | relMSE | NMSE |
|---|---:|---:|---:|
| 最短 quintile（平均 333 m） | **0.309** | **0.973** | **1.476** |
| 他の4 quintile | 0.546–0.563 | 0.698–0.733 | 0.668–0.745 |

| | trace 割合 | target エネルギー | 誤差エネルギー |
|---|---:|---:|---:|
| target エネルギー最上位 decile | 10.0 % | **99.3 %** | **99.6 %** |

原因も測定済みである。近傍を index 格子で選ぶと offset に盲目になり、選ばれた 16 近傍の offset 差は
絶対値でほぼ一定（94–98 m）なのに、query offset に対する相対値は短オフセット側で6倍になる
（0.384 対 0.064）。333 m のトレースに 94 m ずれた文脈は波形を大きく変え、1,539 m なら 6 % で無害である。

対数は**相対**差を一定に抑える軸なので、第5座標を
`log10(max(offset_m, 1 m)) / 4`（[`processing/trace_field_coordinates.py`](../../src/seis_interp/processing/trace_field_coordinates.py)）
とする。宣言した4 decade 幅で割るので、1 単位の意味が dataset に依らない（10 % の offset 変化 = 0.01035 単位）。
観測範囲で割る形は study_073 が見つけた「軸の重みをデータが決める」誤りの再発になるため採らない。

`context.offset_weight` はこの軸を**順位付けだけ**で scale する。したがって1つの grid で2つの変更を分離できる。

- axis_a = 0 の列: field は offset を読むが近傍は study_071 と同じ順位付け → **入力としての offset の効果**
- axis_a の 0 → 2 の傾向: その上に乗る **offset を考慮した近傍選択の効果**

## 事前に記録した差分予測（[`evaluation.yaml`](evaluation.yaml) の `prediction`）

| dataset | offset 範囲 | オフセット別 skill | 予測 |
|---|---|---|---|
| FORGE | 4.63–2266.63 m（2.69 decade） | 最短 quintile が崩れる | **改善する** |
| C3 | 720.28–1964.99 m（0.44 decade） | 平坦（相関 0.982–0.984、relMSE 0.053–0.062、最上位 decile は target エネルギーの 16.4 % のみ） | **中立** |

**反証条件**: 両 dataset で同程度に改善した場合、offset 機構という説明は誤りである。

## 条件

- 探索の正本（両 dataset 共通、1手法 5×5）: [`grid.yaml`](grid.yaml)
- dataset 別: [C3 config](c3/config.yaml) / [inputs](c3/inputs.yaml) / [method](c3/cinr_real.yaml)、
  [FORGE config](forge/config.yaml) / [inputs](forge/inputs.yaml) / [method](forge/cinr_real.yaml)
- 固定値の例外と理由: [`dataset_differences.yaml`](dataset_differences.yaml)
- 評価・事前登録規則: [`evaluation.yaml`](evaluation.yaml)
- 採用理由と未決事項: [`decisions.md`](decisions.md)

config・inputs のいずれにも書けない値は次のとおり。

| 値 | C3 | FORGE |
|---|---:|---:|
| 整列 canvas のサンプル数 | 384 | 2172 |
| width | 112 | 112 |
| パラメータ数（25 候補で共通） | **350,114** | **350,114** |
| 目標 363,269 比 | −3.6213 % | −3.6213 % |
| 近傍数（固定条件） | 16 | 16 |
| 実測 ms/update（k=16） | 5.011 | 23.880 |
| 実測 peak CUDA reserved | 1.393 GiB | 8.688 GiB |
| admission（C-0 規則） | 4 GiB | 11 GiB |

width は GroupNorm の 8 群制約により 8 の倍数に限られ、5座標では 112 が ±5 % 帯内で唯一の値である
（120 は +7.76 % で帯域外）。近傍数は study_071 の FORGE 採用値 16 を両 dataset の固定条件とする
（C3 自身の最適は 4 で、study_071 の測定では k=16 が relMSE 0.0613 → 0.0640 の約 4 % 劣化）。

S1 と 3 seed の合計は C3 約 2 GPU-h、FORGE 約 9.3 GPU-h。全 25 候補が k=16 なので study_071 より高い。

## 実行

```bash
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

C3=$(PYTHONPATH=src "$PY" scripts/run_c3_region_experiment.py prepare \
       --study studies/study_074_forge_c3_offset_field/c3)
FG=$(PYTHONPATH=src "$PY" scripts/run_forge_grid_experiment.py prepare \
       --study studies/study_074_forge_c3_offset_field/forge)
"$PY" scripts/launch_grid_experiment.py forge --experiment "$FG"
"$PY" scripts/launch_grid_experiment.py c3    --experiment "$C3"
```

prepare 後、launch の前に各 experiment の `input/seed_*/partition.parquet` の sha256 が study_067 の
記録と一致することを確認する（study_070 / study_071 と同じ8 partition）。
