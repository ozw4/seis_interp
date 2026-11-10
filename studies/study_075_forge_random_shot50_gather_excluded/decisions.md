# study_075 decisions

## 2026-11-10 — 学習時の gather 除外を whole-shot mask の宣言条件として採用する

**Status:** 採用。whole-shot mask では `context.training_exclusion: source` を `cinr_real` の宣言条件とする。
random-trace mask では既定の `query` を維持する（study_071 の採用設定を変更しない）。

**Decision:** study_072 の後退の支配的な原因は**情報量ではなく学習と test の文脈不一致**だった。
学習時に query の source gather 全体を文脈から除外するだけで clean_target は **+1.3029 dB /
relMSE −0.2514** 改善し、同一 mask・同一 test 集合での順位は 6 位（profile NeRSI の上のみ）から
**2 位**（ReGSI の下、CCNet5D・DRR・POCS の上）へ変わる。

**Evidence:** study_072 との差は手法設定の1行だけで、mask・入力10成果物・model・近傍数 16・
Fourier base 1.05・予算・seed・model_seed は同一である。

| | 学習時の除外 | clean_target SNR (dB) | clean_target relMSE | 学習 loss |
|---|---|---:|---:|---:|
| study_072 | query | 0.3596 | 1.0005 | 0.2043 |
| study_075 | source | **1.6625** | **0.7490** | 0.2444 |

**学習 loss が上がって test が改善した**ことが機構の証拠である。gather を除外すると学習課題は
難しくなる（観測 query も兄弟トレースを使えない）。それでも test が 1.3 dB 改善したのは、
除外前の model が test に存在しない情報へ適合していたからである。loss と test が同方向に動く
過剰適合の形ではなく、train/test 分布不一致の除去の形をしている。

改善は層別しても一様である。edge shot は符号が負から正へ変わり（−0.1023 → +1.1832 dB）、
5 つの offset 五分位すべてで相関と relMSE が改善し（最短 0.2077 → 0.3559 / 1.1208 → 0.8968、
最長 0.4581 → 0.6208 / 0.9254 → 0.6619）、相関中央値は 0.3818 → 0.5431 である。
特定の層だけが得をして平均を動かしたのではない。

**事前の予測との関係:** launch 前に「改善するが study_071 の水準（1.4558 dB）には届かず、
到達目標は study_069 の古典手法帯（POCS 0.7105、DRR 1.0033 dB）」と記録していた。
**実測 1.6625 dB はこの予測を上回り、study_071 の random-trace 水準すら超えた。**
反証条件（改善しないこと）は成立しなかった。予測が外れた向きは楽観側ではなく悲観側である。

予測を外した理由として整合的な説明は、gather 除外が不一致の除去と同時に**暗黙の正則化**として
働くことである。学習 query から最も情報量の多い近傍（同一 gather の兄弟）を奪うと、model は
gather 間の関係だけで補間する関数を学ばざるを得ない。これは test 条件そのものなので、
「文脈を揃える」以上の利得が出た。ただしこれは事後の説明であり、本 study は検証していない。

**帰結と未決事項:**

- whole-shot 欠損に対して neural-implicit の代表手法は **機能する**。study_072 の記録だけから
  「条件付き field は whole-shot 欠損に向かない」と結論するのは誤りであり、
  後退は学習側の視野設定の誤りだった。論文ではこの経緯（072 → 075）を開示して書く。
- `training_exclusion` は mask family に依存する宣言値であり、既定にはしない。random-trace mask で
  gather 除外を行うと学習文脈が test 文脈より**狭く**なる（test は in-gather 近傍 2.63/16 を持つ）ので、
  同じ変更が逆向きの不一致を作る。mask family ごとに test 文脈へ揃えるのが規則である。
- 最短 offset 帯が依然として最も弱い（relMSE 0.8968、NMSE 1.1867）。offset 座標軸の study_074 は
  別の理由（shot50 では近傍をほとんど動かさない）で停止したが、**条件付けではなく出力側**の
  近接 offset 表現は未検証のまま残る。
- ReGSI との差 0.9077 dB は残る。提案法は trace 間の関係を明示的な graph で読み、`cinr_real` は
  k 近傍の softmax 加重で読む。この差の要因分解は本 study では行っていない。
- 1 mask・1 model seed の記述的比較である。study_072 / study_069 との対比は事後的・探索的であり、
  CI も Holm 補正も付いていない。事前登録した cluster bootstrap の対比
  （`evaluation/cross_study_method_contrast.py`）はまだ pipeline として走らせていない。
- 関係別の近傍定義（common receiver / common offset）と格子座標の軸別重みは、study_073 の
  decisions.md に記録したとおり別 study の対象として残る。本 study はそれらを不要にしていない。

## 2026-11-10 — 本 study の順位は公平な比較ではない（開示）

**Status:** 開示。上の採用判断は維持するが、**study_069 の手法群に対する優位の主張には使えない**。

**Decision:** study_075 の変更は shot50 という mask family に合わせた学習視野の適応であり、
study_069 の5手法は同等の適応を受けていない。さらに観測を読む2つの baseline は、本 study が
取り除いたのと同じ train/test 文脈不一致を**未修正のまま、より強く**抱えている。
公平な比較には両者へ同等の mask family 整合を与えた再測定が必要である。

**Evidence:** study_069 の実 run の mask・幾何・実コードで測定した（波形を読まず、採点もしない）。
再現コマンドは [`scripts/audit_shot_mask_training_context.py`](../../scripts/audit_shot_mask_training_context.py):

```bash
.venv/bin/python scripts/audit_shot_mask_training_context.py \
  --experiment runs/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0 \
  --method regsi_real --method ccnet5d_grid
```

| 学習時の in-gather 文脈 | 学習 query | test query |
|---|---:|---:|
| cinr_real（study_072、修正前） | 5.52 / 16 近傍、100 % | 0.00、0 % |
| ReGSI（study_069） | **10.04 / 20 edge、100 %** | 0.00、0 % |
| CCNet5D（study_069） | **35.27 / 48 in-patch gather cell、100 %** | 0.00、0 % |

ReGSI は 4 関係のうち `source` 関係が学習時 4.751/5 が in-gather、test では 0 である。すなわち
**1 関係が学習時は満杯で test では構造的に空**になる。原因は
`training/c3_poc_trace_graph_episodes.py` の episode が `kind="random_trace"` 固定であること。
CCNet5D は patch 内の可視観測 cell 総数が 867 対 841 とほぼ同じなので、情報量ではなく構造だけの
不一致であり、cinr_real study_072 と同じ signature である。

**対称である次元（ここは公平である）:** mask（`partition.parquet` は byte 一致）、test 集合、
入力10成果物、前処理 `trace_oracle`、指標と採点コード、予算（50,000 更新・lr 1e-3・
64 loss trace/更新・パラメータ数 341,485–363,292 がすべて 363,269 ±5% 帯内）、
そして設定の選択元（5手法すべて study_067 の random-trace 70% grid、cinr_real は study_071 の
random-trace grid。**shot50 で grid 探索した手法は1つもない**）。

**帰結:**

- 論文では study_075 の順位を単独で示さない。示すなら「cinr_real だけが mask family に合わせた
  学習視野を持つ」ことを同じ表の中で明記する。そうでなければ、mask family に適応した手法を
  未適応の baseline と比べるという典型的な不公平比較になる。
- 公平化には別 study（whole-shot 単位の学習視野を ReGSI と CCNet5D にも与える）が必要である。
  ReGSI 側の機構は repo に既にある（`training/trace_graph_episodes.py` の `random_whole_ffid`、
  study_021 で使用）が、studies 067/069 の経路（`c3_poc_trace_graph_episodes.py`）は使っていない。
  CCNet5D 側は `_draw_pseudo_target_mask` に shot 単位の抽選を追加する必要がある。
  どちらも既定を変えず宣言値で選ぶ形にすれば、studies 067/069 の記録は無効化しない。
  所要は study_069 の実測から ReGSI 約 6.6 h + CCNet5D 約 3.0 h の約 10 GPU-h である。
- この再測定で ReGSI と CCNet5D が改善する場合、`cinr_real` の 2 位は維持されない可能性がある。
  その結果も記録する。**本 study の採用判断（学習文脈を test 文脈へ揃える）は手法内部の A/B で
  成立しており、順位がどう動いても変わらない。**
