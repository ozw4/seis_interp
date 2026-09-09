# 判断理由

## QC後の物理振幅評価を固定する

異常437本の除外前にfitしたglobal RMSは極端値に支配されており、そのGNN指標を
学習量比較の基準には使えない。Study029のQC suiteとcanonical trainを固定し、
global RMSを28.6279へfitし直した200更新モデルを基準とする。
評価対象は全58,999欠測トレースの384 samplesで、無近傍やゼロ振幅も除かない。
過去のunit-RMS評価や訓練4本へのfitは、今回の物理振幅での10 dB達成とは区別する。

## 正確な近傍検索を保ったまま学習量を増やす

固定訓練4本では学習率1e-3・1,000更新で20.0908 dBまでfitできたが、
QC後200更新の全target基準値は0.0144 dBだった。この差だけで過学習とは判定せず、
学習量と振幅応答を調べる。実C3で幾何検索の同値性とGPUの単発資源を確認したうえで、
幅64・query batch128・学習率1e-3・5,000更新を宣言した。
複数条件を同時に変更するため、単一要因の効果とは主張しない。

`exact_index` とepisodeの可視sender近傍cacheは、近傍距離・ID順・依存graphを保つ。
バッチごとのラベル読み出しも同じ値と順序を保つ。実行コードを凍結し、
学習中に別の候補を実装しても条件が混ざらないようにする。
GPU予備測定では小さいバッチの方がpeakが大きい例もあったため、使用量の単調性や
全学習の上限は推定しない。各実行の予算と計測値をそのrunへ保存する。

## 可視波形の振幅をモデル内で明示する候補

固定訓練4本へfitした従来モデルは、入力を10倍にしても出力RMSが約1.1698倍で、
その入力範囲では振幅の比例応答を持たなかった。これは正規化経路を見直す根拠であり、
GroupNormだけを原因と確定する証拠やvalidationの改善証拠ではない。

候補`observed_trace_rms`は可視波形をtraceごとに正規化し、queryの直接可視近傍の
RMSをD0-IDWで内挿してdecoder出力に掛ける。訓練教師のtrace RMSは用いず、
同じ物理振幅MSEを保つ。従来のglobal RMS条件に対する比較では、このmodeだけを変更し、
幾何・seed・optimizer・学習予算を固定する。新modeの採用には全targetの実測が必要である。

## 採否と参照記録

各条件のprimaryは事前に宣言した更新数のfinal checkpointによる独立予測とし、
best checkpointは補助記録として残す。保存配列の全target再採点と、事前固定の
部分batchによるCPU復元監査を併用する。監査の閾値を出力を見て緩めない。
条件を変更するときはその根拠と予算を実行前に宣言し、過去runを上書きしない。

ユーザーの指定により、SIRENの11.3422 dBは固定shear 0.0006 s/mを明記した
採用参照へ戻した。GNNへ同じ補助変換は導入せず、両モデルの学習情報量の差も明記する。


## TF32を無効にした実行条件へ改訂する

global RMS・5,000更新の独立予測は7.4002 dBだった。全target保存配列の再採点は
通過したが、学習中評価との誤差エネルギー相対差3.1848e-5は固定閾値1e-6を超え、
CPU復元の正規化RMSE・最大差も不合格だった。これらの判定を保持する。

同じcheckpoint・観測入力・事前固定した1,143 queryで、TF32の有無とcuDNN benchmarkの
有無を組み合わせた4条件を調べた。TF32有効の2条件はCPU差の2基準を超え、無効の
2条件は既存3基準を満たした。これは部分queryの数値診断であり、全target監査の代用ではない。
記録は`20260909T074804795328Z_e39df16d563c_tf32_environment_diagnostic`に保存した。

次の学習・学習中評価・best補助予測・final独立予測では
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0`を明示する。
後者はTF32を全kernelで無効にする環境設定である
（[PyTorch公式仕様](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)）。
既存seed helper後のbenchmark設定や乱数列は変えない。新FP32条件のバッチ128・1更新は
成功し、peak allocatedは8.2587 GBだった。単発値を全学習の上限とは扱わない。

振幅処理と実行精度の両方を変えるため、以前のTF32付きglobal RMS条件との差を
振幅処理だけの効果とは主張しない。新しい比較を
`config_observed_trace_rms_fp32_5k.yaml`として実行前に宣言し、元の未実行条件も残す。

## 学習可能なedge時間ずれは本比較に含めない

可視senderのlatent valueへ、幾何差から求める4個の学習係数による時間ずれを与える
opt-inを実装した。固定4本・同じ初期重み・1,000更新のCPU診断では、lag無効の
21.1003 dBに対し20.3853 dBであり、今回の診断では改善しなかった。最大lagは
0.2574 raw samplesで飽和はなかった。これは訓練適合の診断であり、validationでの
優劣を示すものではない。次の全C3比較はlagを含まない振幅処理版の凍結sourceを使う。


## 訓練のtrace RMSをlossの重みに使う候補

全canonical train 1,146,366本の真値エネルギーを調べると、RMS上位1%が79.2234%を
占めた。宣言済み5,000更新が訪れる640,000 queryでも79.1988%だった。
元SEG-Yとの追加照合では高RMSの8本が625 samplesすべて一致した。全訓練energyの
92.6812%が冒頭64 samplesに集中し、validationの同区間は0.0066%だった。
[源照合とoffset・時間帯の報告](../../reports/c3_proposed_gnn_training_energy_20260909.md)を参照する。
これらはゼロ出力時や相対誤差一定時のMSEの重みを表すものであり、学習済みモデルの
残差やgradientの寄与を直接測った結果ではない。記録は
`20260909T081013427411Z_e39df16d563c_training_energy_distribution`に保存した。

一方、global5kの全validation targetでは、真値を参照した単一倍率の最適化でも
7.4002から7.5932 dBへの改善にとどまった。これは教師参照の残差診断であり、
その倍率を実予測や学習へ採用しない。
`20260909T080905003548Z_e39df16d563c_global5k_residual_structure_audit`を参照する。

これらを根拠に、訓練教師traceのRMSを損失の重みにだけ使う候補を準備する。
モデル入力は可視波形だけ、queryの出力scaleは直接可視近傍RMSの内挿のままとする。
教師RMSをモデルのforwardや予測時scaleへ渡さず、validation/testから重みをfitしない。
最終採点は引き続き全targetの物理振幅SNRで行う。実行中のFP32・物理MSE条件は
宣言した5,000更新を完了させ、候補の実行条件とsourceは別に固定する。


ゼロ予測時の実データによる損失配分も確認した。冒頭64 samplesの寄与率は通常MSEの
92.6812%から相対MSEの23.2944%へ、offset40 m未満は69.1804%から0.4153%へ下がった。
1,195本のゼロtraceも分母へ保持した。これは初期目的関数の配分の診断であり、
学習済みの残差やgradientの寄与・validation改善を測った値ではない。
[診断記録](../../runs/study_032_c3_proposed_gnn_10db/20260909T083835772948Z_e39df16d563c_zero_prediction_loss_distribution/result.json)に
定義・float32丸めの扱い・全区分の検算を保存した。
正RMSの最小値は2.7057で、今回の訓練データに極小の正RMSはなかった。

## FP32・観測trace RMS条件を基準に損失だけを比較する

FP32・観測trace RMS・物理MSEの5,000更新は完了し、finalの独立予測は
7.2797 dB・RMSE 4.2987だった。全58,999 targetの再採点、学習中評価との
誤差energy照合、CPU復元の既定3基準をすべて満たした。
[固定監査](../../runs/study_032_c3_proposed_gnn_10db/20260909T081425547636Z_e39df16d563c_observed_rms_fp32_completion_handoff/verification/result.json)を参照する。
数値再現性を確認できたが10 dB未満であり、これを相対MSE比較の基準とする。

次は`config_observed_trace_rms_relative_mse_fp32_5k.yaml`の条件で、損失だけを
`masked_trace_relative_mse`へ変える。モデル・近傍・入力と出力の振幅処理・seed・
optimizer・batch128・5,000更新・FP32設定を保ち、新規初期化から学習する。
固定4本の訓練適合確認は21.1003→21.3990 dBだったが、この小規模値を
全target性能の代わりには使わない。実バッチ1更新の資源確認を経て本学習へ進む。
最終採否は、同じ全targetの物理SNRと変更しない独立監査により判定する。

## 共有GPUの資源変動に合わせてcuDNNの自動探索を無効にする

相対MSE・benchmark有効の実バッチ128・1更新は成功した。ただし本学習の起動前に
共有GPUの外部使用量が増え、空き46,878 MiBに対して既存本学習のpeak reservedは
45,948 MiBと余裕が小さくなった。空きはその後も変動しており、外部jobは操作しない。
この本学習peakは定期validationとbest補助予測も含み、最大値の発生段階は特定していない。

既存の同一checkpoint・観測入力によるFP32診断C/Dでは、benchmark有効／無効で
peak reservedが35,824→2,236 MiBだった。物理出力の相対L2差は1.5004e-7で
固定の数値基準内だがbitwise一致ではない。これは1,143 queryの推論時の比較で、
相対MSEでのbackwardや全5,000更新の最大メモリ・学習軌道を保証するものではない。

これを根拠に`config_observed_trace_rms_relative_mse_fp32_no_benchmark_5k.yaml`で
`training.cudnn_benchmark: false`を明示する。
[PyTorchの仕様](https://docs.pytorch.org/docs/stable/backends)では、このフラグは複数の
畳み込みアルゴリズムを実測して選ぶ処理を制御する。モデル・入力・lossの定義やbatch128を
保ち、自動探索を無効にした実バッチ確認を経て本学習へ進む。物理MSE基準との比較には
lossに加えてこの実行設定の差があることを記録し、lossだけの因果効果とは主張しない。
新条件にも同じ保存予測再採点・CPU復元基準を適用する。

## 2026-09-09 — Adopt the final 5,000-step GNN above 10 dB without fixed shear

The declared relative-MSE, observed-trace-RMS, FP32 condition with cuDNN
benchmark disabled completed all 5,000 updates. The final checkpoint from
`20260909T094605478732Z_e39df16d563c_gnn-train` produced physical target SNR
11.9354 dB and RMSE2.5151 in the separate frozen prediction
`20260909T104419208387Z_e39df16d563c_gnn-predict`. All58,999 targets and
22,655,616 samples were retained. Adopt this final checkpoint; the better
3,000-step validation result,12.1987 dB, remains an auxiliary record and
does not replace the predeclared final-checkpoint selection rule.

The independent saved-output rescore, training/frozen metric comparison,
checkpoint integrity checks, and predeclared CPU restoration on1,143
queries all passed their unchanged tolerances. Training/frozen error-energy
relative difference was1.4331e-10 against rtol1e-6. CPU normalized difference
RMSE was1.0269e-7, maximum6.1962e-6, and physical relative L2 3.3955e-7,
below their respective1e-4,1e-3,1e-3 thresholds. Source and fixed-input
hashes remained unchanged. No quality retries, budget extension, teacher
gain substitution, fixed shear, or learned-edge lag was used.

The audited physical-MSE reference was7.2797 dB. The new condition improves
the full-target result by4.6557 dB while retaining the same declared width,
batch, seed, geometry, and5,000 updates. It changes both the training loss
and the cuDNN benchmark setting, so this is not an isolated causal ablation
of the loss. High-amplitude source-verified train traces were retained;
only the established437 QC exclusions were applied. Training teacher RMS
weights the loss after forward; only visible traces determine model inputs
and interpolated prediction scales.

Set current `config.yaml` status to `final_5000_adopted_10db_achieved` and
retain the pre-execution named configuration unchanged. Stop training for
this goal now that its fixed validation criterion and audits are met.
Keep the SIREN11.3422 dB model as the separately declared fixed-shear
reference. These are one-seed results on one validation case with different
training information: GNN uses the disjoint canonical train pool, while
SIREN fits validation observations. No test-partition performance or
matched-information superiority is established. Preserve every prior
failed condition and audit. See the
[final report](../../reports/c3_proposed_gnn_relative_mse_5k_20260909.md)
for the adopted record, saved reconstruction figures, and run provenance.
