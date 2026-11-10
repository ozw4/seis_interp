# study_072 decisions

## 2026-11-09 — 本 mask では条件付き field を代表手法として採用しない

**Status:** 採用（不採用の判断）。study_071 の random trace mask における採用は維持する。

**Decision:** random whole-shot 50% mask では `cinr_real` を neural-implicit の代表手法として採用しない。
study_069 の5手法の中で ReGSI・CCNet5D・DRR・POCS の下、profile NeRSI の上に位置する。

**Evidence:** study_071 で選ばれた設定（近傍 16 本、Fourier base 1.05）を一切変えずに移した結果、
clean_target の relMSE は 1.0005、mean trace SNR は 0.3596 dB だった。同じ手法・同じ設定が
random trace 70% mask では relMSE 0.7679 / 1.4558 dB だったので、大きく後退している。
internal shot（85 本）でも relMSE 0.9918 で、ゼロ予測との差はわずかである。

後退の原因は近傍の測定で特定できる。幾何だけから、採用設定と同じ 16 近傍を両 mask で選び直した。

| | study_071 random trace 70% | study_072 random shot 50% |
|---|---:|---:|
| 観測 trace 数 | 13,271 | 24,575 |
| 最近傍距離（中央値） | 0.03228 | 0.03229 |
| 同一 shot gather 内の近傍数（16 本中、平均） | **2.63** | **0.00** |
| 同一 shot の近傍を1本以上持つ query | **95.97 %** | **0.00 %** |

本 study は観測 trace が約1.85倍あり、最近傍距離の中央値も同一である。にもかかわらず成績は落ちた。
構造上の違いは一つで、whole-shot mask では shot 単位で観測／欠損が決まるため、test trace と**同一の
shot gather に属する観測 trace が存在しない**。random trace mask では test trace の 95.97 % が
同一 gather 内に観測近傍を持っていた。

**理由:** study_071 で機構が効いたのは gather 内の近傍（同じ震源・隣接受振点）を読めたからであり、
`cinr_real` の条件付けはその情報に依存している。whole-shot mask はその情報を構造的に取り除くので、
残るのは別 shot の近傍だけになり、条件付けの利得が失われる。観測量を増やしても代替にならないことを
上表が示している。

**比較のための注記:** 同一 mask で study_069 の他手法も study_067 の random trace mask から低下するが、
低下幅は小さい（ReGSI 3.0342 → 2.5702、CCNet5D 1.6010 → 1.2157、DRR 1.0910 → 1.0033、
POCS 1.0510 → 0.7105 dB）。`cinr_real` は 1.4558 → 0.3596 dB で最大の低下である。
gather 内近傍以外の構造（関係グラフ、5D 畳み込み、低ランク／スパース性）を使う手法は
whole-shot 欠損でも機能するのに対し、近傍参照だけに依存する条件付けは機能しない。

**帰結と未決事項:**

- 論文では、条件付き field の採用は random trace mask に限ることを明示する。whole-shot 欠損では
  `nersi_profile` を僅かに上回るだけで、古典手法にも届かない。
- この否定的結果は mask 族の性質を切り分ける測定として記録する。条件付けが「どの観測を読めるか」に
  依存することは、近傍参照型の手法一般に当てはまる制約である。
- gather 内近傍が無い場合に機能させるには、近傍の定義を座標距離から別の構造（関係、offset 整合、
  低ランク部分空間）へ変える必要があり、それは本 study の救済ではなく別 study の対象とする。
- 1 mask・1 model seed の記述的比較であり、study_069 との対比は事後的・探索的である。

## 2026-11-10 — 上の因果帰属を訂正する（study_075 による）

**Status:** 上の「不採用」判断は**測定としては有効**だが、**因果の帰属は誤り**だった。
この節は上の節を削除せず、後から分かったことを開示して上書きする。

**訂正の内容:** 上の節は後退の原因を「gather 内近傍という情報が構造的に無いこと」に帰し、
「近傍参照だけに依存する条件付けは whole-shot 欠損では機能しない」「論文では条件付き field の採用を
random trace mask に限る」と書いた。**この結論は誤りである。**
[study_075](../study_075_forge_random_shot50_gather_excluded/README.md) は手法設定の1行
（`context.training_exclusion: source`、学習時に query の source gather 全体を文脈から除外）だけを
変えて同一 mask・同一入力・同一 seed で測り直し、clean_target **1.6625 dB / relMSE 0.7490** を得た。
同一 test 集合で CCNet5D 1.2157・DRR 1.0033・POCS 0.7105 dB を上回り、ReGSI 2.5702 dB の下の **2 位**である。

**何が本当の原因だったか:** 学習と test の**文脈不一致**である。上の表は test query の in-gather 近傍が
0.00/16 であることを正しく測っていたが、**学習 query 側を測っていなかった**。学習 query は自分の shot が
観測されているので in-gather 近傍を 5.52/16 持つ（100 % が1本以上）。study_072 は test に存在しない
情報へ適合するよう学習していた。random trace mask では両者が一致する（2.61 対 2.63）ため study_071 では
この不一致が生じなかった。「study_071 が効いて study_072 が効かなかった差」は上の節が書いた
「gather 内近傍を読めるかどうか」ではなく、「学習と test で読めるものが揃っているかどうか」だった。

**上の節のうち有効なまま残るもの:** 幾何測定（観測数 24,575、最近傍距離の中央値 0.03229、
test の in-gather 近傍 0.00/16、同一 gather 近傍を持つ query 0.00 %）はすべて正しく、
study_075 の設計はこの測定の上に立っている。study_072 の設定での不採用判断も有効である。

**上の節のうち撤回するもの:**

- 「`cinr_real` の条件付けは gather 内近傍の情報に依存している」— 依存していない。
  学習時にその情報を**与えない**ほうが whole-shot mask では良い。
- 「近傍参照だけに依存する条件付けは whole-shot 欠損では機能しない」— 機能する。古典手法3つを上回る。
- 「論文では条件付き field の採用を random trace mask に限ることを明示する」— 限らない。
  mask family ごとに `training_exclusion` を test 文脈へ揃えて宣言する、が正しい記述である。
- 「観測量を増やしても代替にならない」— この主張の根拠は不一致のある学習から得た値なので、
  本 study は判定していない。

**手続き上の含意:** 否定的結果の原因を「与えられない情報」に帰する前に、**学習側とtest側の両方で**
同じ統計を測る。本件では test 側だけを測って原因を確定したと書いたことが誤りの発生源であり、
学習側を測るのは追加の run を要しない幾何計算だけだった。
