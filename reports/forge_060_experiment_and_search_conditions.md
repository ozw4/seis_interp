# study_060_forge_fair_grid：実験条件と各手法のパラメータ探索条件

作成日：2026-10-08。共通設定、事前登録した探索グリッド、各段階の実行記録、最終評価の記録を照合してまとめた。主比較の探索・選択・最終再学習・評価は完了している。指標は小数4桁に丸め、設定値は実行時の精度で記載する。

本実験はFORGE 2017の同一領域に対し、**トレース欠損率70%、観測率30%**のマスクBを適用し、POCS、DRR、CCNet5D、NeRSI、ReGSI-real、ReGSI-gridを比較した。探索には観測集合内の共通検証集合Vを使い、選択後に観測集合全体から再学習して、Bのtestを一括評価した。

条件の正本は[共通設定](../studies/study_060_forge_fair_grid/config.yaml)、[入力契約](../studies/study_060_forge_fair_grid/inputs.yaml)、[探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml)である。本書はこれらと実行記録を説明する文書であり、実行条件を変更するものではない。

## 1. データと外部マスク

| 項目 | 条件 |
|---|---|
| データ | FORGE 2017 |
| 対象領域ID | `medium_source_b06_u3400.0_receiver_b13_u1900.0` |
| 空間軸の順序 | source line、source point、receiver line、receiver point |
| 空間形状 | `[6, 32, 8, 32]` |
| 震源セル数 | 192（6 × 32） |
| 受振セル数 | 256（8 × 32） |
| 全格子セル数 | 49,152 |
| 利用可能トレース | 49,151 |
| 自然欠損セル | 1。人工欠損の採点対象に含めない |
| 外部マスクID | `m1_random70_fair_v1` |
| 外部マスクの種類 | トレース単位のランダム欠損 |
| 外部マスクseed | 600701 |
| 人工欠損（B test） | 34,406本（利用可能トレースの約70%） |
| 観測 | 14,745本（約30%） |
| 主評価のclean test | 34,145本 |
| 元の時間サンプリング | 4,001点、1 ms、0–4 s |
| 共通処理後 | 2,001点、2 ms、0–4 s |

マスクはSHA256順位によって固定し、全手法が同じ観測セルとtestセルを使う。欠損率70%は領域全体での割合であり、個々のギャザーの欠損率は異なる。

FORGE全体の中での解析位置は[全体・解析範囲図（PNG）](../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_analysis_extent.png)、[PDF](../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_analysis_extent.pdf)、[SVG](../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_analysis_extent.svg)で示す。全体配置を灰色、解析の震源192点を橙色三角、受振点256点を青い丸で強調した。破線は解析点のXY外接矩形で、M1は全体の南東寄りに位置する。背景はnavigationの震源1,114点と座標既知の受振点1,740点であり、座標不明の受振点1点を描画対象から外している。

震源と受振点の実座標を同じ縮尺で示す[配置・マスク図（PNG）](../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/geometry_mask_source112.png)、[PDF](../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/geometry_mask_source112.pdf)、[SVG](../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/geometry_mask_source112.svg)を作成した。source cell 112を固定し、その震源に対する受振点73点の観測と183点の人工欠損を色・記号で区別している。他の震源は配置の背景として示し、投影格子座標は表示していない。

`clean_target` は固定QCの `dc_review`、`amplitude_review`、`near_constant_review` のいずれも付いていないトレースである。主評価ではclean集合を使い、副次評価ではQCのreviewフラグを持つトレースも含めた全利用可能test集合を使う。

参照：[共通設定](../studies/study_060_forge_fair_grid/config.yaml)、[マスク・QCの実装](../src/seis_interp/processing/forge_mvp_contract.py)。

## 2. 検証集合Vと最終再学習

Bの観測14,745本から、SHA256順位で10%の1,474本を検証集合Vとして取り分けた。全手法・全候補で同じVを使い、検証マスクseedは600702とした。

| 段階 | 学習／入力観測 | 採点対象 | B testの扱い |
|---|---:|---:|---|
| 探索S1–E、再seed確認R | 13,271本 | V：1,474本、うちclean 1,471本 | 34,406本を除外 |
| 選択後の最終再学習 | 14,745本 | 学習中にB testで設定選択しない | 最終予測を保存 |
| 最終評価 | 学習済み予測を使用 | B test：34,406本、うちclean 34,145本 | 全手法を一括採点 |

最終再学習ではVを観測集合に戻す。したがって、探索時と最終再学習時では利用可能な観測密度が異なる。global RMSも各段階の観測集合から計算し直しており、探索時は約0.0122552、最終再学習時は約0.0118674である。

探索用準備runではB test波形を読んでいない。Vの実RMSは評価側の振幅復元に使い、V波形やRMSをモデル入力として与えない。

参照：[探索用準備記録](../runs/study_060_forge_fair_grid/20260927T133848730415Z_25c5193588b2_validation/preparation_manifest.json)、[最終用準備記録](../runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2/preparation_manifest.json)。

## 3. 全手法に共通する前処理

### 3.1 時間方向の間引き

元の1 ms波形に反射paddingを適用し、41 tapのHamming窓付きsinc FIRでanti-alias処理した後、2倍間引きする。カットオフは元の1サンプル当たり `0.45 / 2 = 0.225` cyclesで、時刻0の位置を保つ。出力はfloat32、2 ms間隔の2,001点である。

AGC、手法固有のband-pass、mute、事前の空間補間は適用しない。

### 3.2 振幅正規化

主条件は `trace_oracle`。観測トレースを各トレースのRMSで正規化し、推定した波形形状を評価対象トレースの実RMSで復元する。

V／testの実RMSを使う箇所は評価境界の正規化・振幅復元に限る。モデルの特徴量や学習ラベルとしてV／testの振幅を渡さない。ただし、欠損トレースの実RMSが分かることを仮定するため、主結果は振幅尺度にoracle情報を使う比較である。実運用で欠損トレースの振幅尺度も未知の場合は、副次評価の条件も併記する必要がある。

### 3.3 幾何に基づく時間整列

全手法が速度500 m/sの共通幾何整列を使う。最大遅延は2,342サンプル、整列後のキャンバスは4,343サンプルである。

ニューラル手法の学習lossは整列キャンバス全体で計算する。CCNet5Dの探索表の時間長 `2001` は設定上の元波形長であり、実際の整列処理では共通キャンバスに合わせる。NeRSIのdecoder出力長も4,343点に合わせる。推論後は時間シフトを逆変換し、元の2,001点へ戻してから振幅を復元し、採点する。

参照：[間引き処理](../src/seis_interp/processing/forge_condition_inputs.py)、[共通の整列・振幅復元](../src/seis_interp/training/forge_mvp_methods.py)、[最終用準備記録](../runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2/preparation_manifest.json)。

## 4. ニューラル手法の共通学習条件

| 項目 | 条件 |
|---|---|
| 対象 | ReGSI-real/grid、CCNet5D、NeRSI |
| optimizer更新数 | 各run 20,000回 |
| 学習率 | 0.001、一定 |
| 1更新のloss対象 | 64本の全時間トレース |
| loss対象トレースの延べ回数 | 1 run当たり1,280,000本分 |
| パラメータ数の基準 | 363,269 ± 5% |
| 学習loss | `masked_trace_relative_mse` |
| lossの集計精度 | デフォルトfloat64 |
| mixed precision | off |
| checkpoint選択 | 最終20,000更新時点のEMA |
| 探索時のEMA decay | 0.999、0.9995、0.9998 |
| 補足用の重み保存 | 10,000更新時点の各EMAも保存 |
| 探索seed | 7301 |
| Rで追加するseed | 7302、7303 |
| 最終再学習seed | 7311、7312、7313 |

ReGSIはAdamW、CCNet5DとNeRSIはAdamを使う。weight decayの数値は手法ごとに探索するため、optimizerの違いも含めて完全に同一の正則化ではない。

64本はloss対象の本数である。ReGSIの近傍入力、CCNet5Dの可視パッチなど、各手法が参照する入力トレース数まで64本に揃えているわけではない。また、パラメータ数・更新数・loss本数を揃えても、FLOPsやGPU実行時間は同一にならない。

### 4.1 選択に使う指標

各トレースの正解を `y_i`、予測を `yhat_i` とすると、正解エネルギーが正のトレースについては次式を使う。

```text
relative_mse_i = sum_t (yhat_i(t) - y_i(t))^2 / sum_t y_i(t)^2
score = mean_i relative_mse_i
```

トレースごとの相対誤差を平均するため、エネルギーを全トレースで合算するglobal NMSEとは異なる。正解のRMSが0の場合は除数を1にする実装で、通常のMSEを使う。

探索スコアはVのclean 1,471本の平均値で、小さいほど良い。探索runごとにEMAの3水準を別構成として採点する。20,000更新以前の最良checkpointを選ぶ方式ではない。

参照：[探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml)、[loss実装](../src/seis_interp/training/trace_relative_loss.py)、各手法の中心設定：[ReGSI-real](../studies/study_060_forge_fair_grid/regsi_real.yaml)、[ReGSI-grid](../studies/study_060_forge_fair_grid/regsi_grid.yaml)、[CCNet5D](../studies/study_060_forge_fair_grid/ccnet5d_grid.yaml)、[NeRSI](../studies/study_060_forge_fair_grid/nersi_real.yaml)。

## 5. 探索・選択の共通手順

### 5.1 S1–E：16学習run、48 EMA構成

| 段階 | 新規run数 | 探索内容 |
|---|---:|---|
| S1 | 9 | 主軸1の3水準 × 主軸2の3水準 |
| S2 | 3 | S1の勝者を土台に、副次設定を3候補試す |
| S3 | 3 | S1–S2の最良設定を土台に、weight decayを3水準追加 |
| E | 1 | 事前登録した規則で、延長または中点を1候補選ぶ |
| 合計 | 16 | 各runのEMA 3水準を採点するため48構成 |

各段階で追加候補が改善しなければ、それ以前の勝者を保持する。S3では新規3水準に土台のweight decayを加えた4水準を比較する。POCSとDRRは後述する16構成の全組合せをS1で実行し、S2–Eは行わない。

### 5.2 Eで実行する1候補の決め方

次の順に判定し、最初に当てはまった候補を実行する。

1. weight decayの最大水準の最終train lossが、最小水準より5%未満しか増えていなければ、weight decayを1段上げる。
2. 勝者が探索軸の端にあり、その先の値が定義されていれば延長する。軸1、軸2、weight decayの順に判定する。
3. 勝者と良い方の隣接候補の中点を試す。軸1を優先し、利用できる中点がなければ軸2を使う。
4. いずれも当てはまらなければweight decayを1段上げる。

「最終train loss」は最後の1,000更新の平均である。軸1・軸2上の位置と隣接候補の比較はS1の勝者に基づき、Eの土台はS3までの最良runとする。S2で変更された軸にはEを適用しない。weight decayの最小または最大水準が発散した場合は、分岐1を適用しない。

延長値・中点として複数の値を登録していても、Eでそれらをすべて実行するわけではない。各手法で実行するEは1 runである。Eの生成にはS3までの結果が必要になる。

### 5.3 R：上位2設定のseed確認

S1–Eから、各runの最良EMAで順位を付け、異なる上位2 runを選ぶ。それぞれをseed 7302、7303で再学習する。元のseed 7301と合わせ、各設定・各EMAの3 seed平均Vスコアが最小の組合せを採用する。

ReGSIはrealとgridで同じハイパーパラメータを使うため、次の追加規則を使う。

- S1–Eの16 runはrealだけで実行する。
- Rでrealの2設定 × 追加2 seedの4 runを実行する。
- 同じ2設定についてgridを3 seedで実行し、6 runを追加する。
- realの3 seed平均とgridの3 seed平均をさらに平均し、設定とEMAを共通で選ぶ。

したがって、ReGSIのRは計10 runである。realとgridを別々の最良設定に調整して比較する方式ではない。

### 5.4 同点・失敗・再実行

同点は段階順、run名、EMAの宣言順で先に来る構成を選ぶ。非有限loss・勾配ノルム・予測で失敗したrunは発散として予算に数え、再実行しない。選択では全EMAを `+inf` として最下位に置き、探索結果の表・曲線ではスコア1.0として残す。

メモリ不足や中断は同じ設定で再実行する。有限値で性能が悪いseedを、性能を理由に除外しない。NeRSIではRのseed 7302のスコアが約1だったが、その結果も3 seed平均に含めた。

参照：[選択規則の実装](../src/seis_interp/processing/fair_grid_rules.py)、[事前登録時の判断・補完](../studies/study_060_forge_fair_grid/decisions.md)。

## 6. ReGSI-real / ReGSI-gridの探索条件

realは実測の震源・受振座標を使い、gridは投影した規則格子座標を使う。モデル構造、学習予算、最終ハイパーパラメータは共通である。

### 6.1 S1：グラフトポロジー × 内部マスク率

| トポロジーID | 関係ごとの近傍数k | message passing回数r | 幅W | 時間dilation | パラメータ数 |
|---|---:|---:|---:|---|---:|
| `k2r2w128` | 2 | 2 | 128 | `[1, 2]` | 363,269 |
| `k1r3w112` | 1 | 3 | 112 | `[1, 2, 4]` | 370,295 |
| `k8r1w152` | 8 | 1 | 152 | `[1]` | 352,347 |

この3種類と内部マスク名目率5%、10%、25%を交差させ、9 runとする。

外部欠損70%とは別に、内部マスクは学習時に観測集合の一部を隠して教師信号にする。1更新64本を維持するため、各episodeで隠す本数を64の倍数に揃える。その結果、探索時と最終再学習時の実装上の率は次の値になる。

| 名目率 | 探索時の率 | 探索時の隠し本数 | 最終再学習時の率 | 最終時の隠し本数 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.04823 | 640 | 0.05209 | 768 |
| 0.10 | 0.10127 | 1,344 | 0.09983 | 1,472 |
| 0.25 | 0.25077 | 3,328 | 0.25175 | 3,712 |

S1の固定条件は、weight decay 0、node Fourier components 16、edge time shift 0である。source、receiver、CMP、offset/azimuthの4関係を使い、距離尺度はtight 160 m、loose 640 m、異方性比4とする。

| 関係 | tight 160 mを使う成分 | loose 640 mを使う成分 |
|---|---|---|
| source | 震源位置 | 受振位置 |
| receiver | 受振位置 | 震源位置 |
| CMP | midpoint | offset vector |
| offset/azimuth | offset vector | midpoint |

### 6.2 S2、S3、E

| 段階 | 候補 |
|---|---|
| S2 | node Fourierを0へ変更、8へ変更、関係異方性を2へ変更の3候補 |
| S2の異方性2 | tight 226 m、loose 453 m。幾何平均を約320 mに保つ |
| S3 | weight decay 0.01、0.05、0.2。土台の0も保持 |
| Eのwd上側延長 | 0.5 |
| Eの内部マスク下側／上側延長 | 名目0.025／0.30 |
| Eの内部マスク中点 | 名目0.075／0.175 |
| topology軸 | 順序なし。延長・中点を定義しない |

内部マスクの追加候補は、探索時→最終時で、名目0.025が0.02411→0.02604、0.30が0.299→0.29949、0.075が0.07716→0.07379、0.175が0.17361→0.17362となる。

S1の勝者が `k8r1w152` の場合、S2のFourier 0/8では幅を160に補正して、パラメータ数を許容範囲に保つ。今回は `k1r3w112` が勝者なので、この補正は使っていない。

### 6.3 実際の選択結果

| 段階 | 新規候補の最良設定 | 最良EMA | Vスコア |
|---|---|---:|---:|
| S1 | `k1r3w112`、内部マスク名目0.25 | 0.999 | 0.6685 |
| S2 | 上記からnode Fourierを0へ変更 | 0.999 | 0.5891 |
| S3 | 上記にweight decay 0.01 | 0.999 | 0.5869 |
| E | 内部マスクを名目0.30へ延長 | 0.999 | 0.5924 |

Eは改善しなかった。RではS3のwd 0.01とS2のFourier 0を比較し、**S3のwd 0.01、EMA 0.999**を共通で採用した。採用構成の3 seed平均Vはreal 0.5900、grid 0.6070で、その平均は0.5985だった。

最終設定は近傍数1、message passing 3回、幅112、時間dilation `[1,2,4]`、node Fourier 0、内部マスク0.25175、weight decay 0.01、EMA 0.999、パラメータ数355,959である。

その他の固定設定は、time downsample factor 2、stem kernel 7、temporal kernel 5、attention幅32、relation embedding次元8、learned gateによる関係融合、gradient clip norm 1.0、edge time shift 0。学習query batchは64、推論query batchは16である。

参照：[探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml)、[ReGSI最終選択記録](../runs/study_060_forge_fair_grid/fair_grid/regsi/final/stage.json)。

## 7. CCNet5Dの探索条件

格子上の5Dパッチを使う。パッチ形状の順序は `[time, source line, source point, receiver line, receiver point]` である。各更新で1パッチを処理し、内部マスクで隠した観測から正確に64本をloss対象として選ぶ。gradient accumulationは使わない。

### 7.1 S1：パッチ文脈 × 時間dilation

| パッチ文脈ID | 設定上の形状 | 内部マスク率 |
|---|---|---:|
| `p4848_f40` | `[2001, 4, 8, 4, 8]` | 0.40 |
| `p4848_f30` | `[2001, 4, 8, 4, 8]` | 0.30 |
| `p6686_f20` | `[2001, 6, 6, 8, 6]` | 0.20 |

この3種類を時間dilation 1、2、4と交差させ、9 runとする。固定条件は空間kernel 3、時間kernel 9、hidden/intermediate channels各36、weight decay 0。S1のパラメータ数は353,413である。

パッチ軸は可視観測文脈の小さい順に並べる。パッチが複数の震源測線・受振測線を含むため、測線方向のkernel成分も学習対象になる。

### 7.2 S2、S3、E

| 段階 | 候補 |
|---|---|
| S2候補1 | 時間kernel 3、時間dilation 1、channels各58（365,807パラメータ） |
| S2候補2 | 空間kernel 5、channels各22（368,655パラメータ） |
| S2候補3 | `[2001,6,6,8,6]`、内部マスク0.30（353,413パラメータ） |
| S3 | weight decay `1e-5`、`5e-5`、`2e-4`。土台の0も保持 |
| Eのパッチ上側延長 | `[2001,6,8,8,8]`、内部マスク0.10 |
| Eのパッチ中点 | `[2001,4,8,4,8]`、内部マスク0.35 |
| Eの時間dilation上側延長／中点 | 6／3 |
| Eのwd上側延長 | `5e-4` |

パッチ軸とdilation軸の下側延長、および登録されていない中点は試さない。

### 7.3 実際の選択結果

S1では `[2001,6,6,8,6]`、内部マスク0.20、dilation 4、EMA 0.9998が最良で、Vスコアは0.6871だった。S2・S3は改善せず、この設定を保持した。

Eではパッチを `[2001,6,8,8,8]`、内部マスク0.10へ延長し、Vスコア0.6823へ改善した。RでEとS1の勝者を比較し、**Eの設定、EMA 0.9998**を採用した。採用設定の3 seed平均Vは0.6846である。

最終設定は空間kernel 3、時間kernel 9、時間dilation 4、channels各36、内部マスク0.10、weight decay 0、EMA 0.9998、パラメータ数353,413。最終マスクでのパッチ配置は625通り、1パッチ当たりの隠し観測は最小84本であり、64本のloss対象を確保できる。推論core形状は `[256,6,32,8,32]` である。

参照：[CCNet5D中心設定](../studies/study_060_forge_fair_grid/ccnet5d_grid.yaml)、[探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml)、[CCNet5D最終選択記録](../runs/study_060_forge_fair_grid/fair_grid/ccnet5d_grid/final/stage.json)。

## 8. NeRSIの探索条件

実測震源・受振座標に基づく連続座標を使い、固定の幾何変換とpolar offset表現によって、midpoint x/y、offset radius、azimuthに対応する4成分をFourier符号化する。周波数は各成分のbaseに対し `pi * base**i`、`i=1,...,16` とする。

固定の幾何変換行列は[中心設定の `coordinate_transform`](../studies/study_060_forge_fair_grid/nersi_real.yaml)に保存され、波形から学習していない。

### 8.1 S1：midpoint周波数base × azimuth周波数base

| 項目 | 水準／固定値 |
|---|---|
| midpoint x/yのbase | 両成分を同時に1.05、1.07、1.10へ変更 |
| azimuthのbase | 1.05、1.10、1.15 |
| radiusのbase | 1.10に固定 |
| Fourier components | 16 |
| latent channels | 4 |
| encoder width | 150 |
| decoder channels | `[16,8,4]` |
| weight decay | `5e-5` |
| パラメータ数 | 369,839 |

midpointの3水準 × azimuthの3水準で9 runとする。その他はGELU、linear出力、upsample scales `[2,2,2]`、decoder kernel 5に固定する。

### 8.2 S2、S3、E

| 段階 | 候補 |
|---|---|
| S2候補1 | latent 8、encoder width 74、decoder `[16,8,4]`（364,263パラメータ） |
| S2候補2 | latent 4、encoder width 123、decoder `[32,16,8]`（362,228パラメータ） |
| S2候補3 | radius baseを1.05へ変更（369,839パラメータ） |
| S3 | weight decay `3e-5`、`7e-5`、`1e-4`。土台の`5e-5`も保持 |
| Eのmidpoint下側／上側延長 | 1.03／1.13 |
| Eのmidpoint中点 | 1.06／1.085 |
| Eのazimuth下側／上側延長 | 1.00001／1.20 |
| Eのazimuth中点 | 1.075／1.125 |
| Eのwd下側／上側延長 | `2e-5`／`1.5e-4` |

### 8.3 実際の選択結果

S1ではmidpoint base 1.10、azimuth base 1.05、EMA 0.9998が最良で、Vスコアは0.8813だった。S2は改善せず、S3のweight decay `7e-5`で0.8771に改善した。Eでmidpoint baseを1.13へ延長したが、Vスコアは0.8832となり改善しなかった。

Rではweight decay `7e-5`と`1e-4`を比較し、**`7e-5`、EMA 0.9998**を採用した。seedのばらつきを含めた採用構成の3 seed平均Vは0.9223である。

最終設定はfrequency bases `[1.1,1.1,1.1,1.05]`、latent 4、encoder width 150、decoder `[16,8,4]`、weight decay `7e-5`、EMA 0.9998、パラメータ数369,839。学習batchは64、推論batchは256である。

参照：[NeRSI中心設定](../studies/study_060_forge_fair_grid/nersi_real.yaml)、[探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml)、[NeRSI最終選択記録](../runs/study_060_forge_fair_grid/fair_grid/nersi_real/final/stage.json)。

## 9. POCSの探索条件

ニューラル学習やEMAはなく、次の4因子を全組合せで評価する。計 `2 × 2 × 2 × 2 = 16` 構成である。

| 探索因子 | 水準 |
|---|---|
| 反復回数 | 8、16 |
| threshold end | 0.10、0.20 |
| 時間窓長 | 512、1,024サンプル |
| 空間窓 | `[1,8,1,8]`、`[2,8,2,8]` |

threshold startは1.0に固定する。時間窓のoverlapは半窓、すなわち256または512。空間overlapは `[1,8,1,8]` に対して `[0,6,0,6]`、`[2,8,2,8]` に対して `[1,6,1,6]` とする。CPU上のbatched NumPy実装を使う。

採用設定は**反復16、threshold end 0.20、時間窓512、空間窓 `[2,8,2,8]`**。全窓形状は `[512,2,8,2,8]`、全overlapは `[256,1,6,1,6]`、Vスコアは0.7452だった。

決定論的な手法なのでRのseed確認は行わず、最終マスクで1 runを実行した。

参照：[POCS中心設定](../studies/study_060_forge_fair_grid/pocs_grid.yaml)、[探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml)、[POCS最終選択記録](../runs/study_060_forge_fair_grid/fair_grid/pocs_grid/final/stage.json)。

## 10. DRRの探索条件

次の3因子を全組合せで評価する。計 `4 × 2 × 2 = 16` 構成である。

| 探索因子 | 水準 |
|---|---|
| rank | 1、2、3、4 |
| 反復回数 | 6、12 |
| damping power | 2、3 |

空間窓 `[1,8,1,8]`、overlap `[0,6,0,6]` に固定する。周波数下限は0 Hz、上限は設定しない。手法固有のband-passを追加しない。CPU上のbatched NumPy実装を使い、workersは32とする。

採用設定は**rank 3、反復12、damping power 2**で、Vスコアは0.7952だった。POCSと同様にRのseed確認は行わず、最終マスクで1 runを実行した。

参照：[DRR中心設定](../studies/study_060_forge_fair_grid/drr_grid.yaml)、[探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml)、[DRR最終選択記録](../runs/study_060_forge_fair_grid/fair_grid/drr_grid/final/stage.json)。

## 11. 実行予算と最終設定の一覧

### 11.1 正式なrun数

| 探索単位 | S1–E／古典S1 | Rの追加run | 最終run | 合計 |
|---|---:|---:|---:|---:|
| ReGSI（real/gridの共有探索） | 16 | 10 | 6 | 32 |
| CCNet5D | 16 | 4 | 3 | 23 |
| NeRSI | 16 | 4 | 3 | 23 |
| POCS | 16 | 0 | 1 | 17 |
| DRR | 16 | 0 | 1 | 17 |
| 合計 | 80 | 18 | 14 | 112 |

この112 runは正式な候補・再seed・最終実行の数で、メモリ不足や中断による再試行回数を含めない。探索量の比較に使う構成数はニューラル手法が48（16 run × 3 EMA）、古典手法が16である。Rと最終再学習はS1–Eの探索予算とは別枠である。

### 11.2 最終採用設定

| 手法 | 主な採用設定 | EMA | パラメータ数 | 最終seed |
|---|---|---:|---:|---|
| ReGSI-real | k1/r3/W112、Fourier 0、内部mask 0.25175、wd 0.01 | 0.999 | 355,959 | 7311/7312/7313 |
| ReGSI-grid | realと同一設定、投影格子座標 | 0.999 | 355,959 | 7311/7312/7313 |
| CCNet5D | 空間patch `[6,8,8,8]`、内部mask 0.10、時間dilation 4、wd 0 | 0.9998 | 353,413 | 7311/7312/7313 |
| NeRSI | bases `[1.1,1.1,1.1,1.05]`、latent 4、encoder 150、wd `7e-5` | 0.9998 | 369,839 | 7311/7312/7313 |
| POCS | 反復16、end 0.20、窓 `[512,2,8,2,8]` | — | — | なし |
| DRR | rank 3、反復12、damping 2 | — | — | なし |

## 12. 最終評価と統計比較

選択後の最終runは採用EMAだけを主checkpointとして使い、`purpose: final_evaluation`、`evaluate: false` で予測を保存する。B testを見た後に別のEMAや設定を選ばない。

最終評価では設定・準備run・各手法の選択記録・予測ファイルのhashを照合し、全14 runをまとめて採点した。2026-10-08 03:41 UTCに完了し、B test波形34,406本の読み込みはこの評価処理内で1回だった。

| 項目 | 条件 |
|---|---|
| 主指標 | clean test 34,145本の平均 `masked_trace_relative_mse`、`trace_oracle` |
| ニューラル手法の報告 | seed 7311/7312/7313の平均と標本標準偏差 |
| 古典手法の報告 | 決定論的な1 run |
| 再標本化単位 | source cell、192群 |
| bootstrap回数 | 10,000 |
| bootstrap seed | 600703 |
| 信頼区間 | 95% |
| 多重比較補正 | 事前登録した6対比内のHolm補正 |

事前登録した対比は次の6つである。

1. ReGSI-grid − CCNet5D：格子幾何を揃えた主比較。
2. ReGSI-real − NeRSI：実測幾何を揃えた主比較。
3. ReGSI-real − ReGSI-grid：ReGSIにおける幾何の効果。
4. ReGSI-grid − POCS：古典手法との比較。
5. ReGSI-grid − DRR：古典手法との比較。
6. ReGSI-real − CCNet5D：幾何の違いも含む比較。

副次評価として、全利用可能test集合、観測log-RMSのIDWによるnon-oracle振幅復元、スケール不変の相対MSEも報告した。non-oracleのIDWは振幅尺度の推定に使うもので、独立した波形補間手法としてのIDW比較ではない。

参照：[最終評価設定](../runs/study_060_forge_fair_grid/final_evaluation.yaml)、[最終評価記録](../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/manifest.json)、[主結果・図表一覧](../studies/study_060_forge_fair_grid/README.md)。

## 13. 比較の解釈に必要な制約と未実施項目

### 13.1 testの独立性

今回の探索ではB testを選択に使っていない。一方、中心設定にはstudy_058/059で従来のマスクA testを使って決めた条件が含まれる。B testの34,406本中24,036本、約70%はA testと同じトレースである。

そのため、今回のVによる選択とBによる評価は分離されているが、同じsurvey上の過去の調整の影響まで除いた、完全に新しいデータによる独立評価とは扱えない。

### 13.2 揃えた予算と揃えていない予算

ニューラル手法では更新数、学習率、loss対象本数、パラメータ数の範囲を揃えた。optimizer、入力文脈、FLOPs、実行時間は異なる。また、ニューラル48 EMA構成と古典16構成には構成数の差があり、ReGSIはreal/gridを共有選択する追加確認を行った。これらは比較条件として明示する。

### 13.3 指標と振幅尺度

主指標はoracle RMSを用いるトレース平均の相対MSEである。全手法で最終結果のglobal NMSEは1を上回っているため、主指標の改善を、エネルギーを合算した誤差でもゼロ予測を上回るという意味に拡張できない。non-oracle復元の結果とともに評価する。

### 13.4 補足実験

次の項目は主比較に含めていない。

- 独立したIDW波形補間baseline：未実施。
- マスクAでの副次評価：実施しない判断を事前記録済み。
- 探索時の観測13,271本と最終時14,745本の密度差の感度：未実施。
- 10,000更新時点と20,000更新時点の比較：重みは保存済み、補足評価は未実施。

参照：[判断・逸脱の記録](../studies/study_060_forge_fair_grid/decisions.md)、[現在の完了範囲と結果](../studies/study_060_forge_fair_grid/README.md)。

## 14. 実行環境と再現に使う記録

最終評価manifestに記録された環境はLinux x86_64、NVIDIA H100 NVL × 2、Python 3.10.12、NumPy 1.26.4、pandas 2.3.3、PyArrow 25.0.1、PyTorch `2.5.0a0+b465a5843b.nv24.9`、segyio 1.9.14、PyYAML 6.0.3である。

ニューラル学習はGPU、POCS・DRRはCPUで実行した。CCNet5Dは後半をGPU 0/1へ振り分け、メモリ不足・中断時には同じ設定で再実行した。実行先の変更は段階記録と各tuning manifestに残している。

再現の参照先は以下のとおり。

| 記録 | 用途 |
|---|---|
| [共通設定](../studies/study_060_forge_fair_grid/config.yaml) | 対象領域、マスク、前処理、学習予算 |
| [入力契約](../studies/study_060_forge_fair_grid/inputs.yaml) | raw入力lock、領域metadataとhash |
| [探索グリッド](../studies/study_060_forge_fair_grid/grid.yaml) | 探索水準、seed、EMA、bootstrap |
| [判断記録](../studies/study_060_forge_fair_grid/decisions.md) | 規則の補完、逸脱、事前登録hash |
| [ReGSI最終段階](../runs/study_060_forge_fair_grid/fair_grid/regsi/final/stage.json) | real/gridの共有選択と最終設定 |
| [CCNet5D最終段階](../runs/study_060_forge_fair_grid/fair_grid/ccnet5d_grid/final/stage.json) | seed平均による選択と最終設定 |
| [NeRSI最終段階](../runs/study_060_forge_fair_grid/fair_grid/nersi_real/final/stage.json) | seed平均による選択と最終設定 |
| [POCS最終段階](../runs/study_060_forge_fair_grid/fair_grid/pocs_grid/final/stage.json) | 検証スコア最小の選択と最終設定 |
| [DRR最終段階](../runs/study_060_forge_fair_grid/fair_grid/drr_grid/final/stage.json) | 検証スコア最小の選択と最終設定 |
| [最終評価設定](../runs/study_060_forge_fair_grid/final_evaluation.yaml) | 最終14 runとbootstrapの対応 |
| [最終評価manifest](../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/manifest.json) | 評価完了、予測・成果物hash、環境、読込回数 |

各最終段階の `rule_inputs` にはS1–Eの候補ごとのスコアと入力runが記録される。個別runの完全な設定はその段階のYAMLとtuning run内の解決済み設定を参照する。
