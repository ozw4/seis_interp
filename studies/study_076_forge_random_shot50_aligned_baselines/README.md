# FORGE random whole-shot50 で baseline の学習文脈も test 文脈に揃える

Status: `prepared` — 実行仕様を凍結した段階。本 study の結果は未確定。

## 目的

[study_075](../study_075_forge_random_shot50_gather_excluded/README.md) は `cinr_real` の学習視野を
whole-shot test 視野へ揃えて **+1.3029 dB** を得た。その変更は mask family に合わせた適応であり、
[study_069](../study_069_forge_random_shot50/README.md) の手法群は同等の適応を受けていない。
観測を読む2つの baseline は同じ不一致を**未修正のまま、より強く**抱えているので、
study_075 の順位は公平な比較ではない。本 study はその不公平を取り除く。

変更は手法ごとに**1行だけ**である。

| 手法 | 宣言した1行 | 意味 |
| --- | --- | --- |
| `regsi_real` | `training.episode_kind: random_whole_ffid` | 学習 episode が trace 単位ではなく gather 単位で隠す |
| `ccnet5d_grid` | `patches.inner_mask_unit: source_station` | patch 内の pseudo-target を cell 単位ではなく source station 単位で抽選する |

mask・入力10成果物・model・予算・seed・選択元候補は study_069 と同一で、両手法の設定ファイルは
study_069 の [`regsi_real.yaml`](../study_069_forge_random_shot50/regsi_real.yaml) /
[`ccnet5d_grid.yaml`](../study_069_forge_random_shot50/ccnet5d_grid.yaml) にその1行を加えただけである。
**どちらも既定値を変えていない**ので、studies 052–069 の記録は無効化しない。

## 動機（幾何だけの測定）

`scripts/audit_shot_mask_training_context.py` で study_069 の実 run の mask・幾何・実コードから測定した
（波形を読まず、採点もしない）。

```bash
.venv/bin/python scripts/audit_shot_mask_training_context.py \
  --experiment runs/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0 \
  --method regsi_real --method ccnet5d_grid \
  [--settings studies/study_076_forge_random_shot50_aligned_baselines]
```

### ReGSI（query あたり、1,500 query 標本）

| 視野 | 全 edge | うち in-gather | in-gather を持つ query | `source` 関係の edge | うち in-gather |
| --- | ---: | ---: | ---: | ---: | ---: |
| 学習（study_069 のまま） | 19.999 | **10.042** | **100.0 %** | 5.000 | **4.751** |
| 学習（gather 単位 episode） | 19.606 | **0.000** | **0.0 %** | 4.620 | **0.000** |
| test（参照） | 19.973 | 0.000 | 0.0 % | 4.973 | 0.000 |

既定では**学習 query の graph の半分が自分の shot 由来**で、4 関係のうち `source` 関係は学習時 4.751/5 が
in-gather なのに test では構造的に空になる。gather 単位 episode はこれを 0 にし、edge 総数は test より
わずかに少ない（19.606 対 19.973）。少ない向きは良性である（学習が test より難しい側）。

### CCNet5D（cell あたり、400 patch 標本）

| 視野 | patch 内の可視 in-gather cell（48 本中） | 1本以上持つ割合 |
| --- | ---: | ---: |
| 学習（study_069 のまま） | **35.265** | **100.0 %** |
| 学習（station 単位抽選） | **0.000** | **0.0 %** |
| test（参照） | 0.000 | 0.0 % |

station 単位でも eligible な patch 配置数は 6,615 で変わらず、隠す cell 数も patch あたり
約 285 → 約 281（−1.2 %）でほぼ同じである。**情報量ではなく構造だけを揃えた。**

## 事前の予測

**両手法とも改善を予測する。** 機構は study_075 と同一（学習と test の文脈不一致の除去）で、
不一致の比率は `cinr_real`（近傍 16 本中 5.52 = 34.5 %）より両 baseline のほうが大きい
（ReGSI は edge の 50.2 %、CCNet5D は in-patch gather cell の 73.5 %）。

**学習 loss は両手法で上がると予測する。** study_075 で 0.2043 → 0.2444 と上がりながら test が改善したのと
同じ符号である。学習課題のほうが難しくなるためで、loss の上昇自体は失敗の兆候ではない。

**改善幅は予測しない。** study_069 の両手法はすでに機能しており（2.5702 / 1.2157 dB）、
`cinr_real` が study_072 で 0.3596 dB とほぼ縮退していたのとは出発点が違う。
すでに動いている手法のほうが同じ修正から得る余地は小さいことがあり、方向だけを予測する。

**反証条件**: どちらも改善しない（または劣化する）場合、学習と test の文脈不一致は観測を読む手法一般の
欠陥ではなく、study_075 の利得は `cinr_real` の k 近傍 softmax 条件付けに固有である。その場合は
study_069 の記録をそのまま公平な比較として扱い、study_075 の古典手法超えを機構の主張として使わない。

**順位への含意（先に書いておく）**: ReGSI が改善すれば `cinr_real` との差 0.9077 dB は広がり、
CCNet5D が 1.6625 dB を超えれば `cinr_real` は 3 位に下がる。**どちらでも
[study_075 の採用判断](../study_075_forge_random_shot50_gather_excluded/decisions.md) は変わらない。**
あれは手法内部の A/B（同一手法・同一 mask で学習視野だけを変えた +1.3029 dB）で成立しており、
順位に依存しない。本 study は順位を正しくするために走らせる。

## 予算の逸脱（1件、事前に開示する）

gather 単位 episode では隠す trace 数が gather サイズで量子化される。観測 96 gather のうち 95 個が
256 trace、1 個が 255 trace（`natural_missing_cell_count: 1`）なので、24 gather を隠すと
6,144 または 6,143 trace になる。6,143 のとき最後の query batch だけが 63 trace になる。
episode あたり 96 更新、全 50,000 更新で 63 trace の更新は期待値 **約 130 回（0.26 %）**、
loss trace 総数の不足は **0.004 %** である。`matched_training.loss_traces_per_update: 64` に対する
この逸脱を許容し、`inner_mask_fraction` は宣言値 0.25 のまま丸めない
（trace 単位での batch 丸めは gather 単位では宣言した gather 比率を動かしてしまう）。
CCNet5D 側に逸脱はない（`sample_batch` が hidden cell から厳密に 64 を選ぶ）。

## 条件

条件の正本は [`config.yaml`](config.yaml)、入力と選択元 hash は [`inputs.yaml`](inputs.yaml)。
mask は `mask_study_id` で study_069 の名前空間を引き継ぎ、波形は study_067 FORGE run の sealed 入力を
`physical_reference` で再利用する。`inputs.yaml` は study_069 の run 記録と両手法の設定ファイルも
対比対象として pin する。

1 mask・1 model seed・2手法の記述的比較であり、study_069 / study_075 との対比は事後的・探索的である。
CI も Holm 補正も付かない。

## 実行

```bash
PY=.venv/bin/python
"$PY" scripts/forge_trace_experiment.py prepare --study studies/study_076_forge_random_shot50_aligned_baselines
"$PY" scripts/forge_trace_experiment.py launch  --experiment runs/study_076_forge_random_shot50_aligned_baselines/<run-id>
"$PY" scripts/forge_trace_experiment.py score   --experiment runs/study_076_forge_random_shot50_aligned_baselines/<run-id>
"$PY" scripts/forge_trace_experiment.py figures --experiment runs/study_076_forge_random_shot50_aligned_baselines/<run-id> \
  --output results/study_076_forge_random_shot50_aligned_baselines/<result-id>
```

prepare 後、launch の前に `input/partition.parquet` の sha256 が study_069 / study_072 / study_075 と
一致することと、両手法のパラメータ数が study_069 の記録（regsi 346,885 / ccnet5d 363,292）と
一致することを確認する。所要見積りは study_069 の実測から ReGSI 約 6.6 h + CCNet5D 約 3.0 h。
