# study_060_forge_fair_grid

Status: `completed`（主比較） — 全手法の探索・選択・最終再学習と14本の最終予測を評価済み。密度差・10,000更新時点の感度は補足として未実施。

評価設定は [`final_evaluation.yaml`](../../runs/study_060_forge_fair_grid/final_evaluation.yaml)、実行記録は [`20261008T030804955521Z_05c92e79ce80_final_evaluation`](../../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/manifest.json) に保存した。2026-10-08 03:41 UTCに正常終了。test波形34,406本は1回読み込み、全78成果物のsha256を照合済み。再実行は行っていない。

## 目的

- FORGE M1（欠損70%）で、GeRSI（ReGSI real/grid）・NeRSI・CCNet5D・POCS・DRRを、事前に登録した同じ探索手順で調整し、新しいマスクBのtest集合で1回だけ評価して比較する。
- 手法の選択には、マスクBの観測トレースから分けた検証集合Vだけを使う。

## 主比較の結果

主指標はclean test 34,145本の平均 `masked_trace_relative_mse`（`trace_oracle`、小さいほど良い）。ニューラル手法はseed 7311/7312/7313の平均±標本標準偏差で、POCS・DRRは決定論的な1 run。数値は小数4桁に丸めた。

| 手法 | 主指標（seed平均±SD） | global NMSE（seed平均） | 相関中央値（seed平均） |
|---|---:|---:|---:|
| ReGSI-real | 0.5779 ± 0.0040 | 1.0995 | 0.6870 |
| ReGSI-grid | 0.5884 ± 0.0051 | 1.0777 | 0.6812 |
| CCNet5D | 0.6526 ± 0.0043 | 1.0621 | 0.6220 |
| POCS | 0.7310 | 1.0800 | 0.5450 |
| DRR | 0.7754 | 1.0820 | 0.4871 |
| NeRSI | 0.8865 ± 0.0088 | 1.0187 | 0.3193 |

事前登録した6対比を、source cell 192群のcluster bootstrap（10,000回、seed 600703）で評価した。差は左−右で、負なら左の誤差が小さい。Holm補正はこの6対比内で行う。

| 対比 | 主指標の差 | 95%信頼区間 | Holm補正p値 |
|---|---:|---:|---:|
| ReGSI-grid − CCNet5D | -0.0642 | [-0.0674, -0.0611] | 0.0006 |
| ReGSI-real − NeRSI | -0.3086 | [-0.3173, -0.3000] | 0.0006 |
| ReGSI-real − ReGSI-grid | -0.0105 | [-0.0123, -0.0088] | 0.0006 |
| ReGSI-grid − POCS | -0.1427 | [-0.1483, -0.1374] | 0.0006 |
| ReGSI-grid − DRR | -0.1870 | [-0.1927, -0.1815] | 0.0006 |
| ReGSI-real − CCNet5D | -0.0747 | [-0.0781, -0.0714] | 0.0006 |

主指標ではReGSI-realが最小で、6対比すべての95%信頼区間が0をまたがない。全手法でglobal NMSEは1を上回るため、エネルギーを合算した誤差がゼロ予測より小さいという結論は得られない。マスクAとBのtestにはトレースの重複がある（[decisions.md](decisions.md)）。

副次評価のclean testでのseed平均:

| 手法 | 観測log-RMS IDWによる振幅復元 | スケール不変の相対MSE |
|---|---:|---:|
| ReGSI-real | 0.6648 | 0.5579 |
| ReGSI-grid | 0.6768 | 0.5671 |
| CCNet5D | 0.7225 | 0.6280 |
| POCS | 0.7692 | 0.6937 |
| DRR | 0.8003 | 0.7358 |
| NeRSI | 0.9147 | 0.8520 |

主指標の順序は副次評価でも同じ。all-eligible、観測振幅からの復元、スケール不変の各対比群でも6対比すべての95%信頼区間が0をまたがず、Holm補正p値は0.05未満だった。

正本: [手法別集計](../../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/aggregate/method_summary.csv)、[seed別指標](../../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/aggregate/seed_metrics.csv)、[全対比](../../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/aggregate/contrasts.csv)、[探索量の曲線](../../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/aggregate/expected_best_curves.csv)。[結果図](../../reports/forge_060_fair_grid_final_evaluation.png)はこれらのCSVから作成した。

## 全体配置と解析範囲

FORGE全体の実座標に、今回のM1解析に使った震源192点と受振点256点を重ねた。全体の震源はnavigationの1,114点（うちローカル記録あり1,106点）、受振点は座標既知の1,740点を表示した。座標不明の受振点1点は描画していない。

全体配置は灰色、解析対象の震源は橙色三角、受振点は青い丸で示す。M1は全体配置の南東寄りに位置する。破線は選択した震源・受振点のXY外接矩形で、解析に使用した点は色で示している。

| 解析点の範囲 | 値 |
|---|---|
| 震源測線 | 507–512の6測線、各32点 |
| 受振測線 | 153、157、161、165、169、173、177、181の8測線、各32点 |
| 合わせたXY範囲 | UTM 12N easting 335,116.7–337,081.4 m、northing 4,260,848.8–4,262,911.1 m |

[全体・解析範囲図 PNG（600 dpi）](../../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_analysis_extent.png) / [PDF（ベクトル）](../../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_analysis_extent.pdf) / [SVG（ベクトル）](../../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_analysis_extent.svg)。幅178 mm・高さ160 mm。[震源座標と解析採用フラグ](../../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_sources.csv)、[受振点座標と解析採用フラグ](../../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/survey_receivers.csv)、[生成記録](../../runs/study_060_forge_fair_grid/20261008T065508000142Z_05c92e79ce80_survey_analysis_extent/manifest.json)も保存した。

全体座標は[全取得配置の監査run](../../runs/study_049_forge_geometry_eda/20260918T053449083425Z_5d660f87b9b4/summary.json)のhash付き幾何表とnavigationから取得し、解析点は最終準備runの対応表から取得した。波形の読み込み・推論・test採点は行っていない。

再生成:

```bash
.venv/bin/python scripts/forge_survey_map.py \
  --preparation runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2 \
  --geometry-run runs/study_049_forge_geometry_eda/20260918T053449083425Z_5d660f87b9b4 \
  --output-dir runs/study_060_forge_fair_grid/<new-survey-map-run> --dpi 600
```

## 実座標の配置・マスク図

震源192点と受振点256点を同じUTM平面図に表示した。実座標のみを使い、横軸と縦軸の縮尺を揃えている。マスクは震源・受振点の組合せ単位なので、既存ギャザー図と同じsource cell 112を固定し、その震源に対する受振点73点の観測と183点の人工欠損を区別した。マスクBの領域全体の欠損率は70%で、この震源の欠損率は約71.5%。

灰色三角は他の震源、星印はsource 112、青い丸は観測受振点、橙色の白抜き丸は欠損受振点を示す。

[PNG（600 dpi）](../../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/geometry_mask_source112.png) / [PDF（ベクトル）](../../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/geometry_mask_source112.pdf) / [SVG（ベクトル）](../../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/geometry_mask_source112.svg)。幅・高さ178 mm。表示した[震源座標](../../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/source_stations.csv)、[受振点座標とマスク状態](../../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/receiver_states.csv)、[入力hash・生成記録](../../runs/study_060_forge_fair_grid/20261008T064324639746Z_05c92e79ce80_geometry_mask_source112/manifest.json)も保存した。波形の読み込み・推論・test採点は行っていない。

再生成:

```bash
.venv/bin/python scripts/forge_geometry_map.py \
  --preparation runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2 \
  --output-dir runs/study_060_forge_fair_grid/<new-geometry-run> \
  --source-cell 112 --dpi 600
```

## 論文用ギャザー図

c3_naのStudy 042と同じ構成を6手法へ拡張した。上段は正解と欠損入力、中段はPOCS・DRR・NeRSI・CCNet5D・ReGSI-real・ReGSI-grid、下段は各手法の正解−内挿結果の残差。全14パネルは同じ寸法・時間軸・振幅クリップ値を使う。

ニューラル手法はすべてseed 7311の保存済み最終予測を使用し、観測列は元波形をそのまま残す。位置は波形やtest指標を見て選ばず、4次元グリッドの中央インデックス `[3, 16, 4, 16]` に固定した。RMS版は全パネルを各正解トレースの全時間RMSで割り、物理振幅版は保存時の物理振幅を使用する。各表示のクリップは4断面の正解振幅だけから計算した99パーセンタイルで、手法ごとに再正規化しない。

| 変化させる方向 | 観測／欠損本数 | RMS版 | 物理振幅版 |
|---|---:|---|---|
| 受振点（共通source cell 112、receiver line 4） | 9／23 | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_point_trace_rms.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_point_trace_rms.pdf) | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_point_physical.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_point_physical.pdf) |
| 震源点（共通receiver cell 144、source line 3） | 8／24 | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_point_trace_rms.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_point_trace_rms.pdf) | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_point_physical.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_point_physical.pdf) |
| 震源測線（source point 16、receiver cell 144） | 1／5 | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_line_trace_rms.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_line_trace_rms.pdf) | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_line_physical.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_source_grid_line_physical.pdf) |
| 受振測線（source cell 112、receiver point 16） | 3／5 | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_line_trace_rms.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_line_trace_rms.pdf) | [PNG](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_line_physical.png)・[PDF](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/gather_receiver_grid_line_physical.pdf) |

8図のPNG/PDF、表示用波形、入力セル一覧、コードとhash記録を[ZIP](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers.zip)にまとめた。幅178 mm・高さ142 mm、PNGは1200 dpi、PDFは文字を埋め込み、波形画像を1200 dpiで保存した。フォントはDejaVu Sans。[生成記録](../../runs/study_060_forge_fair_grid/20261008T045038107403Z_05c92e79ce80_publication_gathers/manifest.json)で予測run・seed・クリップ値・座標を固定している。可視化用に75本の正解トレースを読んだが、新たな推論・test採点・設定選択は行っていない。

再生成:

```bash
.venv/bin/python scripts/forge_fair_grid_gathers.py \
  --evaluation runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation \
  --output-dir runs/study_060_forge_fair_grid/<new-figure-run> --seed 7311 --dpi 1200
```

## 条件

共通の実験条件、各手法の探索範囲、選択規則、実際の採用設定の詳細は[実験・パラメータ探索条件の説明](../../reports/forge_060_experiment_and_search_conditions.md)にまとめた。

実行条件の正本は次のファイルとする。

- 共通条件: [`config.yaml`](config.yaml)
- 入力: [`inputs.yaml`](inputs.yaml)
- 探索グリッドとseed: [`grid.yaml`](grid.yaml)
- 各手法の中心設定: [POCS](pocs_grid.yaml)、[DRR](drr_grid.yaml)、[CCNet5D](ccnet5d_grid.yaml)、[NeRSI](nersi_real.yaml)、[ReGSI-real](regsi_real.yaml)、[ReGSI-grid](regsi_grid.yaml)

判断理由・計画書の解釈・事前登録したファイルのsha256は [decisions.md](decisions.md) に置く。

準備run（どちらも `scripts/forge_m1_mvp.py prepare` で作成）:

| 準備run | 役割 | 観測 | 採点対象 | 除外 |
|---|---|---:|---:|---:|
| [`20260927T133829534335Z_25c5193588b2`](../../runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2/preparation_manifest.json) | 最終（マスクB） | 14,745 | test 34,406（clean 34,145） | 0 |
| [`20260927T133848730415Z_25c5193588b2_validation`](../../runs/study_060_forge_fair_grid/20260927T133848730415Z_25c5193588b2_validation/preparation_manifest.json) | 探索（`--validation-holdout`） | 13,271 | V 1,474（clean 1,471） | 34,406 |

探索用の準備runはマスクBのtest波形を読んでいない（`test_waveforms_read: 0`）。

configに書けない選択規則は次のとおり。すべて `pipelines/fair_grid_stage.py` が機械的に適用し、
各段階の判断と入力を `runs/study_060_forge_fair_grid/fair_grid/<method>/<stage>/stage.json` に記録する。

- スコアはVのcleanトレースでの平均 `masked_trace_relative_mse`。1 runのEMA 3水準をそれぞれ1構成として採点する。
- ニューラル手法は1手法16 run（S1 9、S2 3、S3 3、E 1）で、48構成になる。POCSとDRRはS1の16構成だけとする。
- S1は軸1×軸2の全組合せ。S2はS1の勝者に副次設定を3つ、S3はS2までの最良にweight decayを3水準加える。
- Eは次の順に、最初に当てはまる1 runを行う。
  1. wdの最大水準の最終学習lossが最小水準より5%未満しか高くない場合、wdを1段上げる。
  2. 勝者が軸1・軸2・wdの端にあり、その先の値が定義されている場合、その値へ延ばす。
  3. 軸1（未定義なら軸2）で、勝者とその良い方の隣の中点を試す。
  4. どれも当てはまらなければwdを1段上げる。
- Rでは、各runの最良EMAで上位2 runをseed 7302/7303で再学習する。2 run×EMA 3水準から、3 seedの平均Vが最小のものを選ぶ。
- ReGSIのS1〜Eは `regsi_real` だけで行う。Rでは `regsi_grid` を3 seedで追加し、realとgridそれぞれの3 seed平均をさらに平均した値で選ぶ。
- 同点は特別扱いしない。段階順、run名、EMAの宣言順で先に来たものを選ぶ。
- 発散run（非有限のloss・勾配ノルム・予測で失敗したrun）は予算に数え、再実行しない。選択では最下位、表と探索量の曲線ではスコア1.0とする。メモリ不足などそれ以外の失敗は同じyamlで再実行する。
- 選んだ設定をseed 7311/7312/7313でマスクBの観測14,745本から学習し直し、testを1回だけ評価する。
  最終runは選んだEMAだけを `evaluate: false` で予測・保存し、`scripts/forge_fair_grid_final_evaluation.py` でまとめて採点する。
  評価は、grid.yamlのbootstrap（10,000回、seed 600703）と各手法の `final/stage.json` に一致しない入力を、test波形を読む前に拒否する。
- IDWの下限、マスクAの副次評価は行わない。探索時の勝者によるB testの予測（密度差）と10,000更新時点の重みによる感度は、重みを保存しておき、主結果の後に補足として扱う（[decisions.md](decisions.md)）。

生成と記録の手順:

```bash
P=runs/study_060_forge_fair_grid
V=$P/20260927T133848730415Z_25c5193588b2_validation
F=$P/20260927T133829534335Z_25c5193588b2
.venv/bin/python scripts/forge_fair_grid.py plan --method regsi --stage S1 \
  --validation-preparation $V --final-preparation $F
# stage.json の commands に並ぶ scripts/forge_matched_tuning.py を実行してから次の段階を生成する。
.venv/bin/python scripts/forge_fair_grid.py status
.venv/bin/python scripts/forge_fair_grid.py select --method regsi \
  --validation-preparation $V --final-preparation $F
```

`--method` には `regsi`、`ccnet5d_grid`、`nersi_real`、`pocs_grid`、`drr_grid` を指定する。
前の段階のrunが1本でも終わっていなければ（完了も発散もしていなければ）、次の段階は生成しない。

最終評価の設定ファイル（全手法の `select` と最終runの後に作る）:

```yaml
preparation: runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2
bootstrap: {resamples: 10000, seed: 600703}
final_runs:
  regsi_real:
    - {seed: 7311, tuning_run: runs/study_060_forge_fair_grid/<tuning run>, candidate: <final run name>}
  # pocs_grid・drr_grid（seed: null）、ccnet5d_grid、nersi_real、regsi_grid も同様
```

S1で生成したrun数とパラメータ数（`model_parameter_count`、4,343サンプルのキャンバス）:

| 手法 | S1 run数 | パラメータ数 |
|---|---:|---|
| ReGSI（real） | 9 | 363,269（k2 r2 W128）、370,295（k1 r3 W112）、352,347（k8 r1 W152） |
| CCNet5D | 9 | 353,413 |
| NeRSI | 9 | 369,839 |
| POCS | 16 | — |
| DRR | 16 | — |
