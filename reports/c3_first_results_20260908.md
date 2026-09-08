# 固定C3 validationの初回pilot結果（2026-09-08）

固定Study 027の同一validation入力で、POCSの20反復はSNR **12.8654 dB**、
DRRの5反復は **7.2542 dB**、独自実装SIREN-5Dの2000stepは **−0.0003 dB**、
小型CCNet-5Dの2epoch後は **−0.0027 dB**、小型GNNの200step後の独立final予測は
**−567.8483 dB** だった。研究5手法とzero-fillの全volume成果物と再採点がそろった。
ただし、**GNNの学習時評価と独立予測の厳密一致検査は不合格で、E10の一致条件は未達**である。
集計の`first_results_complete`は5手法の全volume成果物がそろった意味に限り、
E01–E11の全検証合格を意味しない。

[Study 028の条件](../studies/study_028_c3_first_results/README.md)に従い、
対象は`c3_benchmark_validation_random_trace_80_seed142`、shapeは`[384,9,32,8,32]`、
timeは0–3.064秒（8 ms刻み）。source line `[41,50)`、shot `[32,64)`、
receiver x `[0,8)`、receiver y `[18,50)`を維持した。
[完全検証記録](../runs/study_028_c3_first_results/20260908T135858775312Z_edda0ae6aa06_check/native/metrics.json)で
suiteの144ファイル・20caseとexpected SHA-256の一致を確認した。
ここで20caseを検証したことは、20caseのモデル実験を行ったことを意味しない。
データ・partition・mask・volumeは再生成していない。

評価対象は58,999 trace、22,655,616 sample、観測は14,729 trace。
実現欠損率は80.0225%である。同じdense evaluatorで再採点した全行のreference energyは
厳密に同じ`2237830414.4522877`だった。GNNのnative query集計は`2237830414.4522996`で、
加算順序に伴う相対差は約5.3270×10^−15である。物理振幅のreference/error energyを
全target・全384sampleにわたりfloat64で合計したSNRであり、trace別SNRの平均ではない。

| 方法 | 現在の到達点 | 全target SNR (dB) | RMSE | zero-fillとの差 (dB) |
|---|---|---:|---:|---:|
| zero-fill | sanity基準・全volume採点済み | 0.0000 | 9.9386 | 0.0000 |
| POCS | 宣言した20反復・全volume採点済み | 12.8654 | 2.2597 | 12.8654 |
| DRR | 宣言した5反復・全volume採点済み | 7.2542 | 4.3114 | 7.2542 |
| SIREN-5D（独自実装） | 2000stepのfinal・全volume採点済み | −0.0003 | 9.9390 | −0.0003 |
| CCNet-5D | 2epoch・128更新のfinal・全volume採点済み | −0.0027 | 9.9417 | −0.0027 |
| RelationalTraceGraphInterpolator | 200stepのfinal・全volume採点済み、学習時との厳密一致は未達 | −567.8483 | 2.4532×10^29 | −567.8483 |

POCS、DRR、SIREN、CCNetのuncovered sample数はいずれも0、observed再挿入後の最大絶対誤差は0。
GNNも全58,999 target IDを重複・欠落なく覆い、observed誤差は0、contextなしqueryは0だった。
graphのcontextなしは観測依存情報の有無を表し、窓合成のuncovered sampleとは別指標である。
zero-fillは研究比較の第6手法ではなく、採点のsanity基準である。
数表は小数4桁へ丸めており、詳細値の正本は各native記録である。

POCSは窓`[128,4,8,4,16]`、overlap`[64,2,4,2,8]`、threshold 0.9→0.01で実行した。
DRRはrank4・damping4、空間窓`[4,8,4,8]`、overlap`[2,4,2,4]`の588窓を使い、
0–62.5000 Hzの全257 FFT binを処理した。最大Hankel行列は225×64、
FFT/SVD内部はfloat64/complex128で、周波数帯を短縮していない。
この入力と予算ではPOCSとDRRがzero-fillよりtarget誤差を小さくした。
SIRENは`cmp_offset_azimuth`の6特徴、width256・4 hidden layers、199,425 parameterを使い、
同じcropのobserved 5,655,936 sampleだけで正規化・学習した。
batch16,384×2000stepの32,768,000点抽出は復元抽出を含み、異なるsampleの個数ではない。
観測fit lossは約1のままで、今回の予算ではtargetの改善を確認できなかった。
この低いSNRを失敗として削除せず、事前予算のfinal結果として残す。

CCNetはhidden/intermediate各8、kernel3、linear出力の7,257 parameterモデルである。
fit領域はsource line `[0,8)`、shot `[32,64)`、receiver x `[0,8)`、
receiver y `[17,49)`の65,536 trace、selection領域はsource line `[8,10)`、
shot `[32,64)`、receiver x `[0,8)`、receiver y `[18,50)`の16,384 traceとした。
両者はcanonical train pool（1,146,803 trace）内で物理trace集合が重ならず、timeは共に`[0,384)`。
fit領域だけから得たRMSは8.2957である。
固定patch`[32,2,4,2,8]`をfit64・selection16、random_trace80%、seed20260908で作り、
2epochで128更新を完了した。学習のcomplete-patch sample提示数は524,288であり、
領域全体のsample数や異なる学習trace数と同一ではない。
凍結推論はcore`[32,4,8,4,16]`、halo4、576tileで全volumeを覆った。
最大入力shapeは`[40,9,16,8,20]`。この短い予算ではzero-fillからの改善は確認できなかった。

CCNetの最初の外側requestでは`training_inputs.fit_trace_count=8`、
`selection_trace_count=2`と誤記された。これは4次元row配列の第1軸長であり、
trace総数ではない。本報告の65,536／16,384は、native `source_inputs_lock`に固定された
shapeの空間4軸の積とregion解決記録に基づく。元runの記録は書き換えていない。
[checkpoint監査](../runs/study_028_c3_first_results/20260908T141258548561Z_edda0ae6aa06_ccnet_checkpoint_audit/audit.json)では
このsource binding、final role、128step、有限weightと初期weightからの更新を確認した。

GNNはcanonical train pool全1,146,803 trace・time`[0,384)`から80%を隠す
1episodeを作り、seed20260908でhidden IDをシャッフルした順序の最初のbatch
（4query）についてforward/backwardを測った。validationは固定crop内のID先頭queryを
別に測っており、trainのquery順とは区別する。宣言した200step×4queryは800本分の
教師波形提示（307,200 sample）に相当し、全train poolの波形を毎step教師として提示する
意味ではない。全poolは固定RMSのfitと観測support候補に使われる。
[初期preflight](../runs/study_028_c3_first_results/20260908T141536716257Z_edda0ae6aa06_gnn-preflight/native/preflight.json)では
各relation4近傍・validation batch8で、1queryと8queryを計測した。
学習200stepとnative内部のvalidation・best予測を含むtrain actionは8135.5484秒、
別actionのfinal凍結予測は3792.0788秒と外挿され、各action上限3600秒を超えた。

資源に基づく改訂を**1案だけ**行い、各relationの近傍数を4→2、validationと
凍結予測のquery batchを8→32へ変更した。width32、2round、attention16、embedding8、
train query batch4、200step、candidate chunk4096、関係距離は維持した。
[改訂preflight](../runs/study_028_c3_first_results/20260908T142005777781Z_edda0ae6aa06_gnn-preflight/native/preflight.json)では
validation 1・8・32queryと別のtrain1batchを計測し、train action2803.8469秒、
final凍結予測1197.0107秒という各上限内の概算を得た。これを採用して14:24:19 UTCに
新processの本学習を開始し、15:13:24 UTCに外側actionが正常完了した。
200step・800query・307,200 sampleを処理し、
完了episodeは0、`final_episode_interrupted=true`、学習queryのcontextなしは0だった。
27,333 parameterのfinal checkpointが保存され、学習中の全validation指標は
SNR −567.8474 dB（native値`-567.8474194519732`）、RMSE 2.4530×10^29となった。
[final checkpoint監査](../runs/study_028_c3_first_results/20260908T145621168423Z_edda0ae6aa06_gnn_checkpoint_audit/audit.json)で
200step・設定・前処理・provenance・有限weightと72 state tensor中63個の初期値からの更新を
CPU上で確認し、checkpointのhashは不変だった。この監査は学習action内の補助予測中に
行ったcheckpoint限定の確認で、actionの完了根拠はその後の
[native完了記録](../runs/study_028_c3_first_results/20260908T142418647043Z_edda0ae6aa06_gnn-train/native/run.json)である。
native学習actionは全validationと`best_validation` checkpointによる補助予測まで完了した。
今回best stepも200だが、主表は別actionで実行したfinal checkpointの凍結予測を使った。
そのnative値はSNR `-567.8483100732062`、error energy `1.363509535270468e66`である。

[学習時／凍結予測の一致監査](../runs/study_028_c3_first_results/20260908T153234862216Z_edda0ae6aa06_gnn_final_metric_audit/result.json)は、
error energyの相対差2.0509×10^−4、SNR差−0.0009 dB（丸め前`-0.000890621232997546`）により、
元の条件`rtol=1e-6, atol=1e-12`で不合格となった。失敗記録を保持し、許容差を緩めていない。
[保存成果物のCPU監査](../runs/study_028_c3_first_results/20260908T153531961228Z_edda0ae6aa06_gnn_saved_prediction_audit/audit.json)では
best/finalの全72 state tensorと全target ID順が厳密一致し、両dense予測はそれぞれ自身の
native指標と約1e−14の相対誤差で整合した。したがって、個々の成果物の再採点整合と、
異なるprocess間の予測一致を区別する。後者は未達のままである。

固定した先頭・中央・末尾の32／32／23 queryによる数値モード監査では、
[`cudnn_benchmark=false`](../runs/study_028_c3_first_results/20260908T153527990916Z_edda0ae6aa06_gnn_numerical_modes/benchmark_off/audit.json)が
全87 queryで凍結予測と厳密一致した。
[`true`](../runs/study_028_c3_first_results/20260908T153527990916Z_edda0ae6aa06_gnn_numerical_modes/benchmark_on/audit.json)は
中央・末尾で学習actionの補助予測と一致したが、先頭は凍結予測側と一致した。
[`TF32無効`](../runs/study_028_c3_first_results/20260908T153527990916Z_edda0ae6aa06_gnn_numerical_modes/tf32_off/audit.json)は
両方と異なった。フラグを合わせるだけでは全一致を回復できず、差の原因は確定していない。
この限定監査を全volume再現の確認とはせず、全volumeの再実行や主成果物の置換は行わなかった。

| GNN資源設定 | train action概算 (秒) | final予測action概算 (秒) | 全validation batch数 |
|---|---:|---:|---:|
| 初期：各relation4近傍、validation/prediction batch8 | 8135.5484 | 3792.0788 | 7,375 |
| 採用：各relation2近傍、validation/prediction batch32 | 2803.8469 | 1197.0107 | 1,844 |

上表は測定batchからの線形外挿であり、完了runの所要時間ではない。入力検証・
前処理・保存・target採点、および先頭query以外の幾何差の追加費用を含まない。
2つのpreflightのtrain側は同じ全canonical poolを使い、validation側は固定crop外の
supportを加えていない。SNRによる設定選択は行わず、smoke model・optimizer・RNG状態は
本学習へ持ち越していない。近傍数の削減はモデルの参照情報を変えるため、単なる高速化と
同一視しない。train batchとstep数は同じだが、得られる観測supportは初期案と異なる。
preflightのcontextなしqueryはsample内で0であり、全targetのcoverageは別の本予測で確認した。

GNNの全train RMSが極端に大きかったため、固定入力を変更せず
[train振幅監査](../runs/study_028_c3_first_results/20260908T143208413758Z_edda0ae6aa06_train_amplitude_audit/audit.json)を
CPUで行った。対象は許可canonical trainの1,146,803 trace・440,372,352 sampleに限定し、
validation/test波形は読んでいない。native記録、native前処理の再計算、独立したfloat64
stream集計の3経路でRMSが厳密に一致し、値は4.1921×10^34（表示丸め）だった。
絶対振幅1e30超の710 sample・235 traceがほぼ全train振幅エネルギーを占め、
最大絶対振幅は3.3961×10^38、非有限sample数は0だった。

この有限な極端値の原因は監査では特定していない。CCNetの限定fit領域のRMS8.2957と
GNNの全train RMSは大きく異なり、同じ許可poolを参照できることを同じ学習情報・
同じ正規化条件とみなせない。GNN結果の解釈にはこの振幅分布を含める必要がある。
監査は元振幅・mask・pool・モデル・学習設定を変更せず、filterやclipも加えていない。

| 方法・処理 | native段階時間 (秒) | native process最大RSS (KiB) | GPU最大allocated (bytes) |
|---|---:|---:|---:|
| zero-fill採点 | 0.2184 | 808,292 | 対象外 |
| POCS再構成 | 226.8755 | 1,111,788 | 対象外 |
| DRR再構成 | 1444.9943 | 933,984 | 対象外 |
| SIREN対象volumeのobserved fit | 37.4256 | 1,895,352 | 271,598,592 |
| SIREN学習後の凍結予測 | 2.9611 | 同じprocessの値 | 同じprocessの値 |
| CCNet事前学習＋train内selection | 5.4844 | 3,228,044 | 12,998,656 |
| CCNet凍結予測 | 6.6955 | 1,602,496 | 184,357,376 |
| GNN学習・全validation・補助best予測を含むnative全体 | 2925.1236 | 4,081,176 | 7,437,720,576 |
| GNN独立final凍結予測のnative全体 | 966.5042 | 2,195,880 | 148,726,272 |

時間欄は異なる処理範囲を持つ。POCSの入力検証は4.1248秒、DRRは4.3618秒、
SIRENは4.1099秒、
CCNet訓練は16.6777秒、CCNet凍結推論は4.2619秒で、上表の段階時間には含めていない。
DRRの外側action全体は1453.9785秒、native再構成後の採点は0.2235秒だった。
GNNの外側action全体は2945.3956秒で、上表はnative processの全期間を含む。
その内訳として補助best予測はgraph構築492.1041秒、forward371.2899秒、
入力組立306.7747秒、観測読込0.6176秒を記録した。
独立final凍結予測は外側action全体973.2474秒、graph構築482.0640秒、
forward257.3257秒、入力組立212.3292秒、観測読込0.6638秒だった。
RSSとGPU peakはnative process全体の高水位値であり、SIRENのfitと予測に分離した値ではない。
未計測の段階別資源は0とみなさない。CPU本処理は別processで並行する時間帯があり、
GPU手法は`cuda:1`で順次実行したため、専有環境の速度順位は主張しない。

環境はAMD EPYC 9374F、NVIDIA H100 NVL、PyTorchのCUDA 12.6環境。
nativeニューラルrunはfloat32、`float32_matmul_precision=high`、TF32許可、
`cudnn_deterministic=false`で記録されている。`cudnn_benchmark`はSIREN・CCNet訓練・GNN訓練でtrue、
CCNet・GNN凍結推論でfalseであり、native経路ごとの差を保存している。
SIRENのCPU再読込監査では、GPU保存予測との差の最大値が0.0004（表示丸め）だった。
デバイス・数値モード・元のprediction batch shapeをそろえた
[別のGPU監査](../runs/study_028_c3_first_results/20260908T141053861132Z_edda0ae6aa06_siren_checkpoint_audit_same_batch/audit.json)では、
事前固定した先頭4target trace・1,536 sampleの差は厳密に0となった。
これはcheckpoint再読込と数値条件の確認であり、全volumeの追加実験や品質によるrun選択ではない。

主結果の追跡先は、[zero-fill](../runs/study_028_c3_first_results/20260908T140105265573Z_edda0ae6aa06_zero-fill/native/metrics.json)、
[POCS](../runs/study_028_c3_first_results/20260908T140343770055Z_edda0ae6aa06_pocs/native/metrics.json)、
[DRR](../runs/study_028_c3_first_results/20260908T140521669511Z_edda0ae6aa06_drr/native/metrics.json)、
[SIREN](../runs/study_028_c3_first_results/20260908T140522881249Z_edda0ae6aa06_siren/native/metrics.json)、
[CCNet訓練](../runs/study_028_c3_first_results/20260908T141141288531Z_edda0ae6aa06_ccnet-train/native/metrics.json)、
[CCNet凍結推論](../runs/study_028_c3_first_results/20260908T141318498508Z_edda0ae6aa06_ccnet-predict/native/metrics.json)、
[GNN凍結推論](../runs/study_028_c3_first_results/20260908T151434795132Z_edda0ae6aa06_gnn-predict/native/metrics.json)である。
CCNetのtrain内selectionの指標を、固定validationのSNRとして表へ転記していない。
GNNの完了した学習・学習時validationの追跡先は
[GNN訓練指標](../runs/study_028_c3_first_results/20260908T142418647043Z_edda0ae6aa06_gnn-train/native/metrics.json)であり、
独立final凍結予測の主結果とは区別している。

[最終集計JSON](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.json)と
[CSV](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.csv)は
明示したrun pathのみを採用し、case・suite・volume・checkpoint・prediction hash、全target、
各native指標との照合を記録する。未計測資源・段階時間はnullと理由を残した。
[採用manifest](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/manifest.json)には生成元・各図表のhashと、
GNNの実行間照合が不合格だったことを機械可読で記録した。
公開用JSON・CSVでは、環境固有の参照pathをリポジトリ相対に変換し、NumPyのbuild情報内の
ローカルpathだけを省略した。指標・数値設定・図は変更せず、元のnative集計を保持している。
変換前後のhashと変換内容は同manifestに記録する。`runs/`への参照はローカルの実行記録を指し、
Git管理対象の図表は`results/`に置く。
固定断面は[time–receiver y](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/section_time_relative_receiver_y.png)、
[time–shot](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/section_time_shot_in_line.png)、
[time–source line](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/section_time_source_line.png)で、
中央の整数位置と物理座標をJSONに残した。各断面のtargetは24／25／9 traceである。
全方法のpredictionとresidualへ共通のreference由来99百分位clip（±52.3129）を適用し、
source staggerを保った軸で表示した。GNNの極端な予測はこの表示範囲を大きく超える。

[個別波形](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/target_traces.png)は
ID昇順の先頭3本`1898954,1898955,1898956`を固定した。左列はreferenceから決めた
共通範囲±48.7886、右列は全物理振幅の表示で、波形値を正規化していない。
左列の範囲外はGNNが各384 sample・合計1,152 sample、他4手法は0である。
左列では範囲外の点を隠し、その前後を線でつながない。右列には全点を保持する。
表示clipは評価値を変更しない。学習曲線は
[SIRENのobserved fit](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/learning_curve_siren5d.png)、
[CCNetのcomplete-patch loss](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/learning_curve_ccnet5d.png)、
[GNNのmasked-query loss](../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/learning_curve_relational_trace_graph.png)を
別図とした。GNNはnativeの累積`train_loss`（最終1.5147×10^−9）であり、各batchのlossではない。
手法ごとに正規化と教師情報が異なるため、同じloss尺度として比較しない。

このpilot条件ではPOCS・DRRの改善を観測し、ニューラル3手法の改善を確認できなかった。
制約は短い学習予算、小型CCNet/GNN、教師情報・sample提示数の違い、実行範囲の異なる
資源測定、全trainの極端な有限振幅、およびGNNのprocess間厳密一致未達である。
次段階の候補は、SIRENのfit診断、train RMSを支配する極端値の由来確認、
GNNの決定的な数値計算経路の診断とする。いずれも今回の実測に基づく未実行の調査候補である。
このpilotでは追加学習、別checkpointの自動探索、test評価、追加case実験を開始しない。
