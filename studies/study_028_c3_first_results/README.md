# study_028_c3_first_results

固定Study 027のvalidation・random trace欠損80%で、有限予算のPOCS、DRR、
SIREN-5D、CCNet-5D、RelationalTraceGraphInterpolatorの全volume予測・採点・
計算資源・比較図をそろえるpilot実験。学習不足と小型モデルを含む初期観測を扱い、
論文の最終順位や原著性能の再現を目的にしない。

入力は[inputs.yaml](inputs.yaml)のsuiteと単一case
`c3_benchmark_validation_random_trace_80_seed142`。suiteのexpected SHA-256は
`6447a35cc7d4de43532ee8e2e0de3245ac0e93d855efa7ff59c98a85e1631cab`。
volume・mask seed・実現欠損率はmanifestから取得する。
固定されたvalidation shapeは`[384,9,32,8,32]`、time `[0,384)`、
source line `[41,50)`、shot `[32,64)`、receiver x `[0,8)`、receiver y `[18,50)`。
時刻は8 ms間隔、0–3.064秒。これは既存validationであり、Study 027の主test
shape `[384,16,32,8,32]`、sail line25–40を変更するものではない。
データ・partition・mask・case・volume・QCは再生成しない。

[config.yaml](config.yaml)は実験計画と上限、`methods/*.yaml`はnativeキーの
モデル・計算設定である。実験計画をnative pipelineへ直接渡さず、対象actionの
fragmentに検証済みsuiteのbindingだけを合成する。過去Studyのselectionは継承しない。

実行入口は[run_c3_first_results.py](../../scripts/run_c3_first_results.py)。
`--execute`なしでは対象actionの検証済み計画をJSONで返し、runを作らない。
1回の呼び出しにつき1actionを実行する。例えばPOCSの計画確認とpreflightは次のとおり。

```bash
python scripts/run_c3_first_results.py --action pocs
python scripts/run_c3_first_results.py --action pocs --preflight --execute
```

本処理は`--preflight`を外して`--execute`を付ける。`ccnet-predict`と`gnn-predict`には
`--checkpoint <明示したfinal.ptのpath>`が必要で、`summarize`には
`--method-run <method>=<明示run path>`を方法ごとに指定する。
主結果のrunは採用成果のmanifestと集計requestに固定する。

| 方法 | 実行した有限予算 | 小規模preflight |
|---|---|---|
| POCS | 20反復、threshold 0.9→0.01、窓 `[128,4,8,4,16]`、overlap `[64,2,4,2,8]` | 固定1窓・2反復 |
| DRR | rank4、damping4、5反復、空間窓 `[4,8,4,8]`、overlap `[2,4,2,4]`、0〜Nyquist | 固定1窓・全周波数・1反復 |
| SIREN-5D | 6特徴、width256、4 hidden layers、omega30、lr0.0001、2000step、batch16384、prediction batch65536 | observedのみ10step・一部座標 |
| CCNet-5D | hidden/intermediate各8、kernel3、linear、patch `[32,2,4,2,8]`、fit64/selection16、batch1、2epoch、lr0.0001 | 2step・haloを含むtile |
| GNN（資源改訂を採用） | multi_relation/learned_gate、2round、width32、attention16、embedding8、各relation2近傍、candidate chunk4096、200step、train query batch4、validation/prediction batch32 | validation1/8/32query・train1batch |

CCNetは **pilot reduced-width/kernel configuration** であり、原著相当の
64channel/kernel5と同じサイズではない。GNNもwidth・近傍数を縮小している。
関係距離とgeometry featuresはStudy 026の現行設定を使う。method_variantの
意味は既存実装を維持し、小型pilotという情報は実験設定へ記録する。

GNNの採用設定は各relation2近傍・validation/prediction batch32。
4近傍・batch8の初期preflightではtrain action8135.5484秒、final予測3792.0788秒と
各action上限3600秒を超える概算になったため、資源に基づく1案のみを改訂・再計測した。
採用案の概算はtrain action2803.8469秒、final予測1197.0107秒である。
width32、train batch4、200step、関係距離、全canonical train poolと固定validationを維持し、
SNRによる選択は行っていない。予測値は測定batchの線形外挿であり、本学習・全volume予測の
完了時間や品質ではない。本学習actionは実測2945.3956秒、独立したfinal推論actionは
973.2474秒で完了し、ともに3600秒上限内だった。
採用fragmentは[gnn_train_resource_revision.yaml](methods/gnn_train_resource_revision.yaml)と
[gnn_predict_resource_revision.yaml](methods/gnn_predict_resource_revision.yaml)、
根拠の実測と制約は[初回報告](../../reports/c3_first_results_20260908.md)に記録する。

SIRENは対象cropのobservedだけでRMSと学習点を作る独自実装で、
`cmp_offset_azimuth`の6入力を維持する。CCNet/GNNに許可された教師情報は
canonical train rowsとtime `[0,384)`。CCNetのfit/selection領域はtrain内で
幾何確認後に確定し、GNNは`all_train_traces`を使う。許可poolが共通でも、
実際のトレース・sample・mask露出量は異なる。純粋な構造比較とは結論しない。
CCNetはcomplete-patch loss、GNNはepisodeのmasked lossで学習し、両者とも
初回training maskはrandom_trace80%だけを使う。

model seedは20260908。CCNet patch seedも20260908、SIREN sampler/GNN episodeは
既存APIの`training.random_seed`に従う20260908で、個別の役割として記録する。
評価mask seed142はsuiteから読む。native `project.random_seed`はdense補間に
mask seed、CCNet訓練/GNN訓練・推論にpartition seed42を渡す。
CCNetはvalidate_every_steps64、report_every_steps16、decay_after_epochs2、
decay_factor0.1、GNNはvalidation_interval200とする。

各actionの上限は60分、各小規模preflightは10分。実行前に資源と概算所要時間を
確認し、許された計算設定の改訂は理由・変更前後を新runへ保存する。
GPU actionは`cuda:1`の新プロセスで順次実行し、CPU長時間学習へ自動fallbackしない。
初期予算は測定済み推奨値ではなく、良いSNRまで繰り返す条件でもない。
モデルのsmoke状態を本pilotへ持ち越さない。

推論入力は同じcropの観測振幅と利用可能な座標。crop外supportやtarget波形を
モデル・正規化へ追加しない。全target・全384sampleの物理energyをfloat64で
合計して`physical_amplitude_global_snr_db`を求める。contextなし・uncoveredも
採点し、null-SNRは既存statusと保存する。zero-fillは採点のsanity baseline。
validation誤差による診断・選択は許容するが、主表は宣言予算のfinal checkpointを
用いる。bestは補助欄に分ける。testによる設定選択・test予測は実行しない。

成功条件は5方法の全volume予測・全target採点・資源計測・比較図の実生成。
SNR閾値、他手法に勝つこと、loss単調減少は要求しない。失敗・OOM・timeoutは
比較から消さず、不足があれば`partial_results`、そろえば`first_results_complete`。
比較用runは明示pathで選択し、元native出力を編集しない。全20case、追加モデル、
アブレーション、任意のwhole-FFID確認は自動実行の対象外。

固定suiteの完全検証後、5方法の全volume予測・全target採点・資源記録・固定比較図を生成した。
各保存予測を既存dense採点器で再採点し、それぞれのnative指標との一致を確認した。
したがって成果物の状態は`first_results_complete`である。ただし、**GNNの学習時と独立推論間の
指標照合は、当初の許容差を満たさなかった**。この状態名は全監査の合格を意味しない。

| 方法 | 全target SNR (dB) | RMSE |
|---|---:|---:|
| zero-fill（sanity基準） | 0.0000 | 9.9386 |
| POCS | 12.8654 | 2.2597 |
| DRR | 7.2542 | 4.3114 |
| SIREN-5D（独自実装） | −0.0003 | 9.9390 |
| CCNet-5D | −0.0027 | 9.9417 |
| RelationalTraceGraphInterpolator | −567.8483 | 2.4532×10^29 |

全方法が58,999 target trace・22,655,616 sampleを採点した。GNNのquery indexに欠落・重複はなく、
contextなしqueryも0だった。GNNは200step、800教師query、307,200 sampleの提示で終了し、
全train poolの1episodeは完了していない。低いSNRもそのまま採用し、品質による追加学習は行っていない。

GNNの全train poolのRMSは4.1921×10^34（表示丸め）で、極端な有限値710 sampleが
振幅エネルギーのほぼ全量を占めた。発生原因は未確定であり、固定入力や正規化は変更していない。
CCNetの限定fit領域のRMS8.2957との差を、学習条件の制約として扱う。

GNNの学習時finalと独立推論のerror energy差は相対2.0509×10^-4、SNR差は0.0009 dBだった。
元の監査条件`rtol=1e-6, atol=1e-12`は不合格のまま保持した。weight・対象IDは一致し、
固定した87queryの独立推論値は再読込で完全再現できたが、学習時と同じcuDNN設定でも
全batchは学習時予測と一致しなかった。原因は完全には切り分けられておらず、
許容差の緩和や、良い指標が出るまでの再実行はしていない。

[初回報告](../../reports/c3_first_results_20260908.md)に、主結果、監査未達、
小型モデル・短い学習予算・情報量の差・計測範囲の制約を記録する。

採用成果： [CSV](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.csv)、
[JSON](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/first_results.json)、
[固定断面](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/section_time_relative_receiver_y.png)、
[波形比較](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/target_traces.png)、
[生成元・hash・監査未達のmanifest](../../results/study_028_c3_first_results/20260908T154124633001Z_edda0ae6aa06_summarize/manifest.json)。
