# study_076 decisions

## 2026-11-10 — baseline 側の学習視野も mask family に揃える

**Status:** 採用（公平化の判断）。既定値は変えない。`episode_kind` と `inner_mask_unit` は
mask family ごとに宣言する値とする。

**Decision:** 観測を読む手法の学習視野は、その手法が test される mask family に揃える。
whole-shot mask では `regsi_real` に `training.episode_kind: random_whole_ffid`、
`ccnet5d_grid` に `patches.inner_mask_unit: source_station` を宣言する。
random-trace mask では既定（`random_trace` / `trace`）が正しいので変えない。

**なぜ公平化が必要か:** study_075 は `cinr_real` にこの整合を与えて +1.3029 dB を得たが、
study_069 の手法群は同等の適応を受けていない。mask family に適応した手法を未適応の baseline と
比べるのは典型的な不公平比較であり、[study_075 の README](../study_075_forge_random_shot50_gather_excluded/README.md)
にその開示を書いた。開示だけで済ませず測り直すのは、不公平な数字を根拠にした主張を論文に残さないためである。

**なぜこの2行が「同等の適応」なのか:** 3手法はいずれも観測集合を学習時に再 mask する。
再 mask の単位が test mask の単位と違うと、学習 query は test query が持てない文脈を持つ。
`cinr_real` では query の gather を文脈から除外し、ReGSI では episode を gather 単位にし、
CCNet5D では patch 内の pseudo-target を station 単位にする。いずれも
**「隠す単位を test mask の単位に合わせる」**という同一の操作で、手法ごとの実装語彙に翻訳したものである。

**実装上の決定:**

- どちらも既定値を変えない追加の宣言キーとして実装した。ReGSI 側の機構は repo に既にあった
  （`training/trace_graph_episodes.py` の `EPISODE_KINDS`、study_021 で使用）が、
  studies 067/069 の経路（`c3_poc_trace_graph_episodes.py`）は `random_trace` を固定していた。
  その固定を宣言値に変え、既定の RNG 呼び出し順序は1つも変えていない（テストで固定した）。
- `_validate_episode` は従来 `kind != "random_trace"` を拒否していた。両 kind が正当になったので、
  代わりに**ラベルと中身の整合**（`random_whole_ffid` なら hidden 集合が gather の和集合であること）を
  検査する invariant に置き換えた。既存のラベル付け替えテストはこの invariant で再び失敗する。
- `resolve_c3_candidate` の `inner_mask_fraction` 丸めは gather 単位では行わない。隠す数が gather サイズで
  量子化されるので、query batch への丸めは宣言した gather 比率を動かしてしまう。代わりに
  batch 1本が 63 trace になる期待 0.26 % の逸脱を README で開示する。
- CCNet5D 側は `_available_units` を導入して `_eligible_spatial_starts` と `_draw_pseudo_target_mask` の
  両方に同じ単位定義を使わせた。station 単位でも eligible な配置数（6,615）は変わらず、
  隠す cell 数は patch あたり約 285 → 約 281 で、情報量ではなく構造だけが変わる。

**留保:**

- `inner_mask_fraction` と `patches.inner_mask_fraction` は study_067 の random-trace grid が選んだ値を
  そのまま使う。**baseline を再調整はしない。** 再調整すれば「baseline に有利な探索をどこまでやるか」が
  恣意的になり、`cinr_real` 側は shot50 で grid 探索していないので対称性も崩れる。
  gather 単位 episode で学習時の可視密度が test より少し低くなる（edge 19.606 対 19.973）のは
  この選択の帰結であり、良性の向き（学習が test より難しい）である。
- 本 study は順位を正しくするために走らせる。結果が `cinr_real` の 2 位を崩しても
  [study_075 の採用判断](../study_075_forge_random_shot50_gather_excluded/decisions.md)は変わらない。
  あれは同一手法・同一 mask で学習視野だけを変えた内部 A/B で成立している。
- `nersi_profile` と `inr_real` は観測を読まない重みのみの手法なので、この整合は適用対象外である。
  POCS / DRR は学習しないので学習視野を持たない。したがって study_069 の5手法のうち
  整合が必要なのは本 study の2手法だけである。
- 1 mask・1 model seed の記述的比較である。事前登録した cluster bootstrap の対比
  （`evaluation/cross_study_method_contrast.py`）はまだ pipeline として走らせていない。
