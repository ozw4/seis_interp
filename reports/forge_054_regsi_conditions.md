Study 054：ReGSI-real の成功条件の探索

**トレース毎RMS正規化の条件では、欠測10%・20%・50%で探索の成功基準を満たし、元の1 msでも改善を確認できた。** 8 msへのダウンサンプリングでは学習時間が約2.1〜2.2倍短くなった。一方、短いオフセットの大振幅波形には大きな誤差が残った。

各トレースを自身のRMSで正規化し、テストの正解RMSも振幅復元に使う条件で、欠測率と時間サンプリングを比較した。テスト波形そのものはモデル入力・学習教師に入れていない。テスト情報を利用する探索結果であり、通常の未知データに対する性能とは区別する。

[条件設定](../studies/study_054_forge_regsi_conditions/config.yaml) / [入力](../studies/study_054_forge_regsi_conditions/inputs.yaml) / [集計CSV](../runs/study_054_forge_regsi_conditions/20260921T040650094735Z_51acdb640a29_analysis/common_probe_metrics.csv) / [オフセット別CSV](../runs/study_054_forge_regsi_conditions/20260921T040650094735Z_51acdb640a29_analysis/offset_metrics.csv) / [検証記録](../runs/study_054_forge_regsi_conditions/20260921T040650094735Z_51acdb640a29_analysis/verification.json)。

モデル構造・seed 7301・学習率0.001・AdamW・relative MSE・inner欠測率10%を維持した。各条件は同じ初期重みから5,000 optimizer step、query batch 32本で学習した。元のStudy 053のbatch 6本とは異なるため、同studyとの差をダウンサンプリングだけの効果とは解釈しない。各runの設定と実装snapshotを保存した。

時間の間引き前に、反射padding付きHamming窓sinc低域フィルタ（20*q+1 taps、cutoff 0.45/q cycles/sample）を適用した。q=8では4,001点が501点になる。単純な間引きではない。波形を100本ごとに抽出した診断では、q=8の平均二乗振幅保持率の中央値は約99.5%だった。低レート化では畳み込みの物理時間幅も広がるため、計算短縮以外の影響もある。

**共通の未観測トレースでの結果**

元のhash順位を保持した入れ子の欠測集合を使い、全条件で同じ512本を未観測に保った。以下はそのうちcleanな504本、固定最終EMAの結果。相対MSEはトレースごとの誤差を等重みで平均し、ゼロ予測の値は1。表の数値は小数4桁へ丸めた。8 msで相対MSE≤0.8かつ相関中央値≥0.3を満たす条件を1 msで確認する方針とした。raw重みの結果も集計CSVに残した。

| 欠測率 | 8 ms 相対MSE | 8 ms 相関中央値 | 1 ms 相対MSE | 1 ms 相関中央値 |
|---|---:|---:|---:|---:|
| 10% | 0.3999 | 0.8080 | 0.4773 | 0.7609 |
| 20% | 0.4474 | 0.7708 | 0.5326 | 0.7297 |
| 50% | 0.5662 | 0.6968 | 0.6758 | 0.5999 |
| 80% | 0.8079 | 0.3959 | — | — |

![欠測率・サンプリング間隔による比較](../runs/study_054_forge_regsi_conditions/20260921T040650094735Z_51acdb640a29_analysis/condition_metrics.png)

1 ms・欠測10%については、共通probe以外も含めた全欠測トレースを採点した。cleanな4,879本で相対MSE 0.4907、相関中央値 0.7566、global NMSE 0.9811だった。

8 ms・欠測10%の予測を線形補間で1 msへ戻し、元の未フィルタ波形に対して採点すると相対MSEは 0.4018 だった。この診断では、モデルのunit-RMS空間の出力を補間し、正解の1 ms波形のRMSで復元した。追加学習はしていない。上表の1 msで最初から学習した実験とは別である。

**計算時間**

学習historyのstep時間を合計した実測値。データ準備・最終予測・ファイル保存を除く。H100 NVLを2台使い、各GPUでは原則1条件ずつ実行した。元レートの実行中に短い追加推論を行った時間も含まれるため、厳密な速度benchmarkではない。

| 欠測率 | 8 ms 学習時間（分） | 1 ms 学習時間（分） | 時間比（1 ms / 8 ms） |
|---|---:|---:|---:|
| 10% | 15.6195 | 33.9305 | 2.1723 |
| 20% | 15.9095 | 32.8934 | 2.0675 |
| 50% | 15.5944 | 34.5332 | 2.2145 |

**正解RMSを利用する効果と残る誤差**

同じ学習済み予測について、正解RMSでの復元と、観測トレースだけから推定したIDW RMSでの復元を比較した。IDWはStudy 053と同じ投影格子、各軸±2セル、物理距離の逆二乗重み。モデルの再学習はしていない。比較として、送信点・受信点の実座標4次元のユークリッド距離を用い、近い観測16本のRMSを逆二乗距離で平均する復元も試した。いずれもIDWの入力には観測RMSだけを渡し、正解RMSは保存済みのoracle復元を取り除く際にのみ使った。

| サンプリング | 欠測率 | Oracle相対MSE | 格子IDW相対MSE | 実座標16近傍IDW相対MSE | Oracle global NMSE |
|---|---:|---:|---:|---:|---:|
| 1 ms | 10% | 0.4773 | 0.8916 | 0.5979 | 0.9786 |
| 1 ms | 20% | 0.5326 | 0.9467 | 0.6574 | 0.9759 |
| 1 ms | 50% | 0.6758 | 0.9022 | 0.7299 | 0.9977 |
| 8 ms | 10% | 0.3999 | 0.8314 | 0.5189 | 0.9650 |
| 8 ms | 20% | 0.4474 | 0.8590 | 0.5734 | 0.9778 |
| 8 ms | 50% | 0.5662 | 0.8466 | 0.6518 | 0.9956 |
| 8 ms | 80% | 0.8079 | 0.9982 | 0.8601 | 1.0063 |

さらに8 ms・欠測10%では、保存したEMAモデルをロードし、観測波形と観測RMSだけを読み込む独立した推論を実行した。テスト波形は予測とRMS推定が完了した後に初めて読み込み、評価した。相対MSEは 0.5189。oracle評価と同じunit-RMS空間の予測であることも検証した（最大差 3.71e-07）。[推論実装と検証記録](../runs/study_054_forge_regsi_conditions/20260921T033726277764Z_51acdb640a29_observed_real_idw16/manifest.json)。元の1 ms・欠測10%でも同じ観測のみの独立推論を行い、全clean欠測 4,879 本では相対MSE 0.7987、相関中央値 0.7566 だった（[記録](../runs/study_054_forge_regsi_conditions/20260921T034359032552Z_51acdb640a29_observed_real_idw16_native/manifest.json)）。共通probeよりも全欠測集合で誤差が大きく、少数の振幅推定の外れ値が残る。この追加のRMS推定方式は探索後の診断であり、独立した検証集合で選択したものではない。

トレース等重みの指標が改善しても、物理振幅でのglobal NMSEは高い。8 msの共通clean集合では、エネルギー上位5本が全体の約82%を占めた。これらは短いオフセットに集中していた。オフセット別集計は事後の原因診断であり、全体評価から失敗例を取り除いてはいない。

![波形の比較](../runs/study_054_forge_regsi_conditions/20260921T040650094735Z_51acdb640a29_analysis/waveform_examples.png)

波形図の上段は8 ms・欠測10%におけるcleanトレースの誤差中央値の例、下段は共通clean集合で最もエネルギーが大きい例。選択規則を固定して典型例と失敗例の両方を示した。

![学習曲線](../runs/study_054_forge_regsi_conditions/20260921T040650094735Z_51acdb640a29_analysis/learning_curves.png)

学習lossは直近100 stepの平均。評価には最良途中stepを選ばず、固定5,000回終了時のraw/EMAを使用した。各条件は1つのmask順位・model seedに限られる。今回の比較はすべてトレース毎正規化で、正規化しない学習との対照実験は実行していない。欠測率ごとに観測本数が変わるため、同じupdate数でも各トレースの教師提示回数は同じではない。

学習曲線は終了時にも低下傾向があり、収束後の性能上限は調べていない。欠測80%の基準未達も、この学習予算での結果であり、学習時間を増やしても改善しないという意味ではない。

**現時点で支持される条件と原因の切り分け**

- まずトレース毎RMS正規化・8 ms・欠測10〜20%を、ReGSI-realの動作確認条件として採用する。欠測50%まで広げた結果と元レートの結果は上表のとおり。これは設定した探索基準に対する採用であり、全波形・全振幅で十分な精度という判定ではない。
- 欠測率の比較ではモデル、学習step、seedと評価トレースを揃えた。観測を増やすと誤差が下がるため、観測密度は今回の失敗を左右する条件である。一方、マスク間では学習に使えるトレースも変わるので、近傍密度だけの因果効果とは分離していない。共通512本から最も近い観測点までの実座標4次元距離の90パーセンタイルは、欠測10%で49.9374 m、欠測80%で100.2311 mだった（格納値は集計CSV）。
- 同じモデル出力でも、RMSの復元方法で誤差が大きく変わる。従来の投影格子IDWによる振幅推定は、波形推定と別に改善すべき箇所である。実座標での推定は改善したが、少数の外れ値が残る。
- 正解RMSを与えても短いオフセットの大振幅トレースは難しい。ここには振幅スケールだけで説明できない波形形状の誤差が残っている。
- 8 msの出力を元波形に戻しても改善が残る。時間畳み込みのカーネルサイズを固定したままサンプリング間隔を広げるため、物理時間での畳み込み範囲が広がる効果も考えられる。ただし、この仮説を独立に検証する受容野対照実験は今回行っていない。

再実行例：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python scripts/forge_regsi_conditions.py run \
  --cache runs/study_054_forge_regsi_conditions/20260921T023914084646Z_51acdb640a29_data \
  --factor 8 --missing-fraction 0.1 --steps 5000 --batch-size 32 --device cuda:0
```

元レートは `--factor 1`、全欠測トレースの追加評価は `--all-test` を指定する。実行したrun：

- [8 ms / 欠測10%](../runs/study_054_forge_regsi_conditions/20260921T023952131404Z_51acdb640a29_q8_missing0.1_trace_oracle/manifest.json)
- [8 ms / 欠測80%](../runs/study_054_forge_regsi_conditions/20260921T023953036258Z_51acdb640a29_q8_missing0.8_trace_oracle/manifest.json)
- [8 ms / 欠測50%](../runs/study_054_forge_regsi_conditions/20260921T025503069300Z_51acdb640a29_q8_missing0.5_trace_oracle/manifest.json)
- [1 ms / 欠測10%](../runs/study_054_forge_regsi_conditions/20260921T025610384771Z_51acdb640a29_q1_missing0.1_trace_oracle/manifest.json)
- [8 ms / 欠測20%](../runs/study_054_forge_regsi_conditions/20260921T031107277636Z_51acdb640a29_q8_missing0.2_trace_oracle/manifest.json)
- [1 ms / 欠測50%](../runs/study_054_forge_regsi_conditions/20260921T032842239760Z_51acdb640a29_q1_missing0.5_trace_oracle/manifest.json)
- [1 ms / 欠測20%](../runs/study_054_forge_regsi_conditions/20260921T033302063863Z_51acdb640a29_q1_missing0.2_trace_oracle/manifest.json)

実際の欠測集合と振幅条件は各runのmask.parquetおよびmanifest.jsonに記録されている。探索初期のrunのconfig.yamlには準備元のmask/RMSに関する継承metadataが残るが、探索の入力にはmanifest記載の条件を使用した。後続runではこの継承metadataをsource_preparation_metadataへ分離した。この整理は学習計算を変更していない。
