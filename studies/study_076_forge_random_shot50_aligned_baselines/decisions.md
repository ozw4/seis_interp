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

## 2026-11-10 — 結果: 不一致は観測を読む手法一般の欠陥だった

**Status:** 公平化は完了。以後 whole-shot mask の比較は本 study の順位を正本とする。

**Decision:** 学習と test の文脈不一致は `cinr_real` 固有ではなく、**観測を読んで学習する手法一般の欠陥**
だった。3手法すべてが同一の修正から改善し、3手法すべてで学習 loss が上がりながら test が改善した。

| 手法 | 修正前 | 修正後 | 差 | 学習 loss |
|---|---:|---:|---:|---|
| `cinr_real`（study_072 → 075） | 0.3596 | **1.6625** | **+1.3029** | 0.2043 → 0.2444 |
| CCNet5D（study_069 → 076） | 1.2157 | **1.9120** | **+0.6963** | 0.2051 → 0.2724 |
| ReGSI（study_069 → 076） | 2.5702 | **2.9124** | **+0.3422** | 0.1568 → 0.1990 |

改善幅は出発点の逆順である。不一致に壊されていた度合いが大きい手法ほど利得が大きく、
「test に存在しない文脈への適合を取り除く」という機構の説明と整合する。
事前に記録した2つの予測（両手法改善・学習 loss 上昇）はどちらも的中し、反証条件は成立しなかった。

**順位の訂正:** 公平化した順位は
ReGSI 2.9124 > CCNet5D 1.9120 > `cinr_real` 1.6625 > DRR 1.0033 > POCS 0.7105 > profile NeRSI approximately 0。
**`cinr_real` は study_075 時点の2位から3位へ下がった**（CCNet5D が 0.2495 dB 上回る）。
これは不公平を取り除いた結果であり、study_075 の順位は 2026-11-10 の開示どおり使わない。

**論文に書くこと:**

- neural-implicit の代表手法（`cinr_real`）は whole-shot 欠損で**古典手法2つを上回るが、
  学習型 baseline 2つには届かない**。random-trace mask（study_071）での 7 手法中3位と整合する。
- 提案法 ReGSI の優位は公平化後も残る。本 study 内の paired 対比では 96 shot のうち
  **91 shot** で CCNet5D を上回り（平均 +1.0006 dB、中央値 +0.9138）、internal 83/85、edge 8/11。
- **比較手法の学習視野を mask family に揃えることは、比較の前提条件として報告する。**
  揃えないと CCNet5D を 0.70 dB、ReGSI を 0.34 dB 不当に低く見せる。提案法の優位を
  過大に見せる方向の誤りなので、これを開示せずに study_069 の数字を使うことはしない。
- 経緯（072 の誤診 → 075 の修正 → 075 の不公平の自己開示 → 076 の公平化で自分の順位が下がる）を
  そのまま書く。順位が下がる結果を出す測定を自分で走らせたことが、他の数字の信頼性の根拠になる。

**残る非対称性（開示する）:**

- `inner_mask_fraction` は study_067 の random-trace grid が選んだ値のままで、**3手法いずれも
  shot50 で再調整していない**。この点は対称である。
- 学習時の可視密度は手法ごとに少し違う。`cinr_real` は query の gather 1 個だけを除くので
  観測 96 gather 中 95 個が見える。ReGSI は 24 gather を隠すので 72 個（edge 19.606 対 test 19.973）、
  CCNet5D は patch 内の約 25 % の station を隠す。いずれも test より**少ない**向き（良性）だが同一ではない。
  完全に揃えるには leave-one-shot-out episode が必要で、それは batch 内の shot 多様性を失う
  （1 episode が 1 gather の 256 query だけになる）ため別 study の対象とする。
- ReGSI の edge shot はほぼ改善しない（+0.0129、11 本中 5 本）。edge では片側支持しかなく、
  gather 単位 episode が奪う情報の代替が得にくい。edge に限れば trace 単位 episode のほうが
  良い可能性があり、mask family ではなく shot 支持に応じた学習視野は未検証である。
- 1 mask・1 model seed・記述的比較である。事前登録した cluster bootstrap の対比
  （`evaluation/cross_study_method_contrast.py`）は pipeline 未実装で、CI も Holm 補正も付いていない。
