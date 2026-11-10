# study_074 decisions

## 2026-11-10 — 支配項である短オフセットに log-offset 座標で当てる

**Status:** 採用（条件の宣言）。結果が出る前の設計判断である。

**Decision:** 条件付き field に第5座標 `log10(max(offset_m, 1 m)) / 4` を加え、
`context.offset_weight` でその軸を順位付けだけ scale する。study_071 / study_067 / study_070 の
test を見た後に決めた追加実験であり、対比は3 study の事前登録対比群の外にある探索的な比較とする。

**Evidence:** study_071 の FORGE
[per-trace table](../../runs/study_071_forge_c3_context_field/20261109T042814452404Z_766f4f10a666_forge/method_runs/cinr_real/evaluation/seed_7321/test_per_trace.parquet)
から、誤差が1つの部分集団に集中していることが測定できた。

| オフセット 5 分位 | trace | 平均 offset | 相関中央値 | relMSE | SNR (dB) | NMSE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 6,831 | 331.67 m | **0.3089** | **0.9733** | 0.3940 | **1.4760** |
| 2 | 6,830 | 646.24 m | 0.5464 | 0.7326 | 1.6367 | 0.7454 |
| 3 | 6,831 | 899.45 m | 0.5550 | 0.7099 | 1.7450 | 0.6985 |
| 4 | 6,830 | 1160.66 m | 0.5461 | 0.7127 | 1.7072 | 0.7052 |
| 5 | 6,831 | 1533.84 m | 0.5631 | 0.6977 | 1.8634 | 0.6679 |

target エネルギー最上位 decile（10.0 % の trace）が target エネルギーの 99.3 %、
誤差エネルギーの 99.6 % を占め、その NMSE は 1.4786、relMSE 中央値 1.1006、相関中央値 0.1784 である。
残り 90 % は NMSE 0.8217、relMSE 中央値 0.7194。つまり **80 % のオフセット帯では機能しており、
最短 quintile だけが落ちていて、そこが全エネルギーを持つ**。

機構も同じデータから出る。index 格子の順位付けは offset に盲目なので、選ばれた 16 近傍の
offset 差は絶対値でほぼ一定である。

| query offset | 近傍 offset 差（絶対） | 同（query offset 比） |
|---|---:|---:|
| 333.31 m | 94.31 m | **0.3844** |
| 650.13 m | 96.92 m | 0.1509 |
| 902.36 m | 96.88 m | 0.1081 |
| 1164.73 m | 96.60 m | 0.0834 |
| 1539.09 m | 98.19 m | **0.0644** |

相対差が短オフセット側で6倍になり、skill が崩れる場所と一致する。

**上限の見積り:** 最短 quintile が他 quintile 並みになれば、FORGE の mean trace SNR は
1.4558 → 約 1.74 dB（CCNet5D 1.6010 を超え 7 手法中 2 位）、global NMSE は 1.4894 → 約 0.70
（global SNR −1.68 → +1.55 dB）。global NMSE は現在この比較で最悪（他手法 1.0444–1.1016）なので、
賞金が最も大きい箇所である。

**なぜ対数か:** 波形が従うのは offset の**相対**差である。対数軸に一定の重みを置くことが
相対差を抑えることに等しい。線形 offset 軸は絶対差を抑えるだけで、測定された勾配に当たらない。

**なぜ観測範囲で割らないか:** study_073 で、軸の重みがデータ（格子の index 本数）に決められていると
survey 端で損をすることが測定された。宣言した4 decade 幅で割れば 1 単位の意味が dataset に依らず
（10 % の offset 変化 = 0.01035 単位）、両 dataset の実測範囲は clip に触れない
（FORGE 0.1663–0.8388、C3 0.7144–0.8233）。

**なぜ順位付けだけを scale するか:** 入力としての offset と、offset を考慮した近傍選択は別の主張である。
field は常に第5座標を読み、`context.offset_weight` は順位付けのみに効く。したがって axis_a = 0 の列が
前者を、axis_a の傾向が後者を測る。

## 2026-11-10 — 固定条件とその代償

**Status:** 採用。

- **width 112 / 350,114 parameters（目標比 −3.6213 %）。** 第5座標で Fourier 幅が 320 → 400 になり、
  codec の GroupNorm 8 群制約により width は 8 の倍数に限られる。±5 % 帯内の 8 の倍数は 112 だけである
  （120 は +7.7554 %）。study_071 の −0.1726 % より帯の中心から離れることを代償として受け入れる。
- **近傍数 16 を両 dataset の固定条件とする。** study_071 の FORGE 採用値である。C3 自身の最適は 4 で、
  study_071 の測定では k=16 が relMSE 0.0613 → 0.0640 の約 4 % 劣化だった。本 study の主対象は FORGE であり、
  C3 は差分予測の対照として走らせるので、FORGE の採用値を共通条件とする。
- **axis_a = 0 は study_071 と厳密に同一ではない。** 入力が 4 → 5 座標、width が 120 → 112 に変わるため、
  study_071 との対比は「offset 入力」と「width 変更」を混同する。軸内の傾向（a01 → a05）は width と入力が
  一定なので交絡しない。この非対称性は対比の解釈時に明示する。

## 2026-11-10 — 差分予測と反証条件

**Status:** 採用（launch 前に `evaluation.yaml` の `prediction` へ記録済み）。

C3 のオフセット範囲は 720.28–1964.99 m（0.44 decade）で短オフセットが存在せず、skill はオフセットに対し
平坦（相関 0.982–0.984、relMSE 0.053–0.062、NMSE 0.052–0.060）、エネルギー集中もない
（最上位 decile が target エネルギーの 16.4 %）。したがって:

- **FORGE では改善を予測する**（2.69 decade、測定された短オフセットの失敗）。
- **C3 では中立を予測する**（0.44 decade、平坦）。
- **反証条件**: 両 dataset で同程度に改善した場合、offset 機構という説明は誤りであり、
  効いているのは別の何か（入力次元の増加、width の変更など）である。

## 2026-11-10 — study 内 YAML の sha256

launch 時点の正本。

| file | sha256 |
|---|---|
| [c3/cinr_real.yaml](c3/cinr_real.yaml) | `a3e44e7b0ffe4102b448790234c80783285dc63d4a8abf859016e71888881498` |
| [c3/config.yaml](c3/config.yaml) | `2d8dd4007966882775b226504d4647017d53669c805a3f492e5ce234f4414e9c` |
| [c3/inputs.yaml](c3/inputs.yaml) | `02db0cdeca2570fa8736495e2c8102e82076caf71de5413832475854d4f50ede` |
| [dataset_differences.yaml](dataset_differences.yaml) | `6572297f10ab05185ea01c585b75e483b86b8a0087cf555dc44f1f3bab2d2784` |
| [evaluation.yaml](evaluation.yaml) | `8d3bc2e82a9aa3839ecea8ef6c3780e828c94d2e7df682c770540c07650fe4f0` |
| [forge/cinr_real.yaml](forge/cinr_real.yaml) | `307dc54cdae573ad6d9e87d67793d498585cae371acf1cd3bb7118db49b4522a` |
| [forge/config.yaml](forge/config.yaml) | `6f6af0a08dad86881dcb947259ced0aa51633350f3e9ac070472bd9e7c016ed7` |
| [forge/inputs.yaml](forge/inputs.yaml) | `645f339c296102a6e9192a9ca2d515862ee425cd3c8e4a5b65187090ee375021` |
| [grid.yaml](grid.yaml) | `192a65bc667337380c022850a9b5e7635f3978a6fe9c39ff2cfe014b25db3db4` |
