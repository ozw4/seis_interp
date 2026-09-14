# NeRSI target-informed parameter tuning

Status: `target_unmet`. Target: physical-amplitude mean trace SNR **> 15 dB**.
Primary metric: arithmetic mean of per-target-trace SNR in dB, over the full fixed T.
Global SNR is supplementary; existing run metrics and comparison artifacts retain their original definitions.
Reference: frozen Study 037 NeRSI, mean trace SNR 10.9811 dB / global SNR 10.1017 dB.

各traceの真値energyをE、誤差energyをDとして、主指標はmean(10 log10(E/D))。
epsilon、clipping、非有限traceの暗黙除外は行わず、E=0 / D=0の件数と非有限状態を別途報告する。
既存comparisonの`snr_db`はglobal SNRであり、主指標として読み替えない。
保存予測を再評価した5条件では、`aligned_ema0999`のtrace SNR平均13.7364 dBが最高。
他の全候補についてtrace SNR平均の順位は未集計。

### QC閾値の感度分析

元の625 samples全体のmax(abs(amplitude))が閾値を超えるtraceを除く既存QC規則を使用する。
固定crop内の最大値は153.8800で、閾値10,000 / 3,000 / 1,000 / 300では除外なし。
`aligned_ema0999`の保存済み物理予測について、Tを仮に絞った場合は次のとおり。

| QC threshold | Would exclude O | Would exclude T | Remaining T | Mean trace SNR [dB] |
|---|---:|---:|---:|---:|
| 10,000–300 | 0 | 0 | 104,710 | 13.7364 |
| 100 | 480 | 1,651 | 103,059 | 13.7615 |
| 50 | 21,724 | 85,841 | 18,869 | 13.4184 |

これは評価集合だけの感度分析であり、O除外後の再学習結果ではない。
振幅の大きさだけで異常とは断定できない。閾値の採択、QC artifact・mask再生成は行っていない。
QC変更でTを減らした値は、固定Tの15 dB達成とは扱わない。

## 保存済みglobal SNR集計（副指標）

単一モデル64学習runの検証済み最高は`aligned_ema0999`の12.3879 dB（RMSE 2.2188）。
reference比+2.2862 dB、15 dBまでの差は2.6121 dB。目標は未達。
最高候補はGlobal RMS、時間整列あり、データ拡張なし、constant LR=0.001、EMA decay=0.999。
`aligned_fractional30625`の12.1843 dBに対してEMAだけを変更し、+0.2036 dBとなった。
全候補の正本は[summary.json](../../runs/study_040_c3_nersi_target_tuning/20260912T101025Z_03c2fc83df18_comparison/summary.json)、
目視用は[summary.csv](../../runs/study_040_c3_nersi_target_tuning/20260912T101025Z_03c2fc83df18_comparison/summary.csv)。
全runのsuccess、full input lock、resolved config、実update数、full target coverage、
observed exact reinsertion、prediction/checkpoint hashを再検証している。
単一モデル累計2,640,000 updates。独立局所モデル6条件・28モデルの200,000 updatesは別集計。
referenceと等探索予算ではない。

座標・最適化の比較は各50,000 updatesで、全8条件を保持する。

| Candidate | T SNR [dB] |
|---|---:|
| aligned_cosine | 12.0920 |
| aligned_ema0999 | 12.3879 |
| aligned_nyquist | 11.2504 |
| aligned_cartesian | 12.1386 |
| aligned_cartesian_nyquist_cosine_ema | 11.1302 |
| aligned_ema09999 | 12.0835 |
| aligned_nyquist_dense | 11.4503 |
| aligned_cartesian_nyquist_dense_cosine_ema | 11.3987 |

この条件範囲ではEMA=0.999のみが比較元を改善した。
EMA=0.999と0.9999のraw training lossログは500回のreportすべて一致した。
Cartesian profile座標や帯域制限が一般に不利であるとの結論ではなく、
同一Tを繰り返し参照した探索結果として扱う。

主な条件を抜粋する。

| Candidate | Updates | T SNR [dB] |
|---|---:|---:|
| large_frequency125_steps50000（時間整列なし） | 50,000 | 11.6646 |
| siren_batch256_lr0001 | 10,000 | 9.0543 |
| siren_batch256_lr001 | 10,000 | 9.5637 |
| aligned_base | 5,000 | 10.7949 |
| aligned_idw | 5,000 | 10.4837 |
| aligned_large | 50,000 | 11.8145 |
| aligned_idw_large | 50,000 | 11.1624 |
| aligned_low_frequency | 50,000 | 11.5367 |
| aligned_fourier8 | 50,000 | 9.7724 |
| aligned_fourier16 | 50,000 | 11.6963 |
| aligned_fourier80 | 50,000 | 12.0722 |
| aligned_fourier160 | 50,000 | 12.1025 |
| aligned_decoder_bottleneck | 50,000 | 11.1475 |
| aligned_mixup020 | 50,000 | 11.7244 |
| aligned_mixup050 | 50,000 | 11.7990 |
| aligned_encoder4096 | 50,000 | 11.4516 |
| aligned_latent384 | 50,000 | 11.8864 |
| aligned_kernel9 | 50,000 | approximately 0 |
| aligned_zero_pad | 50,000 | 11.8373 |
| aligned_lr0003 | 50,000 | 11.8371 |
| aligned_kernel9_lr0001 | 50,000 | 11.3818 |
| aligned_steps200000 | 200,000 | 12.1755 |
| aligned_fractional30625 | 50,000 | 12.1843 |
| aligned_fractional29375 | 50,000 | 12.1257 |
| aligned_fourier16_jitter020 | 50,000 | 11.6450 |
| aligned_fourier16_jitter005 | 50,000 | 11.6431 |
| aligned_kernel3 | 50,000 | 11.6302 |
| aligned_kernel1 | 50,000 | 9.6466 |

`aligned_steps200000`はupdate数だけを200,000へ変更した条件。
`aligned_kernel1/3`はkernel=5の条件からkernelだけを変更し、局所的なdecoderを調べた。
全64条件はglobal SNRで評価済み。主指標の到達判定には全Tのtrace SNR平均を使う。

### 勾配蓄積による大型バッチ

micro-batchは16 profilesで固定し、O trace数に比例したloss重みで勾配を蓄積する。
optimizer・LR schedule・EMAは実効バッチごとに1回だけ更新する。
以下は同じGlobal RMS、model、seed、LR=0.001、EMA=0.999、時間整列を使う。

| Candidate | Accumulation | Effective profiles | Updates | O trace presentations | Physical SNR [dB] |
|---|---:|---:|---:|---:|---:|
| aligned_ema0999 | 1 | 16 | 50,000 | 5,151,573 | 12.3879 |
| aligned_ema_accumulate8 | 8 | 128 | 10,000 | 8,240,639 | 12.3124 |
| aligned_ema_accumulate16 | 16 | 256 | 10,000 | 16,482,238 | 12.1844 |

実効128→256は蓄積回数だけを変更した比較で、0.1280 dB悪化した。
実効16の比較元とはoptimizer更新数・提示profile数が異なるため、純粋な同一計算量比較ではない。
peak GPU allocatedは128/256とも13,804,825,600 bytes（推論を含むrun全体）。
学習時間は順に280.9 / 334.6 / 4,066.9秒だが、GPU共有負荷が異なり速度比として解釈しない。
この条件範囲では大型バッチ化は物理SNRを改善せず、15 dB未達。
[実効128](../../runs/study_040_c3_nersi_target_tuning/20260912T101028Z_03c2fc83df18_global_accumulation_gpu_audit/aligned_ema_accumulate8/result.json)と
[実効256](../../runs/study_040_c3_nersi_target_tuning/20260912T101028Z_03c2fc83df18_global_accumulation_gpu_audit/aligned_ema_accumulate16/result.json)の
strict GPU checkpoint再推論はいずれも最大絶対誤差0、保存出力の再評価も一致した。

## 入力・探索契約

### 正規化振幅の参照実験

`normalized_ema`は`aligned_ema0999`の学習振幅のみを各Oトレース自身のRMSで正規化する。
`normalized_ema_mixup`はO-only隣接profile mixup（最大比率0.5、seed=501）だけを追加、
`normalized_ema_steps200000`はupdate数だけを200,000へ変更する。
MSE、model/seed、時間整列、EMA=0.999、LR=0.001、固定O/Tを保持する。

参照SNRはモデルの正規化出力と`T / RMS(T)`を全Tで比較する。
予測自身のRMSによる再正規化はしない。TのRMSは評価時にのみ使用する。
保存済みphysical predictionをcheckpointのO-only IDW scaleで割り、正規化出力を復元するため、
float32保存丸めの誤差を含む。通常の物理振幅SNRは別途保存し、その15 dB達成とは扱わない。
参照結果は別directoryの`summary.json`へ保存し、元runのmetricsを上書きしない。

全Tの[正規化参照結果](../../runs/study_040_c3_nersi_target_tuning/20260912T025759Z_03c2fc83df18_normalized_reference_final/summary.json)は次のとおり。目標は未達。

| Candidate | Updates | Normalized SNR [dB] | Physical SNR [dB] |
|---|---:|---:|---:|
| normalized_ema | 50,000 | 12.3212 | 11.7028 |
| normalized_ema_mixup | 50,000 | 12.2566 | 11.6521 |
| normalized_ema_steps200000 | 200,000 | 12.3167 | 11.7092 |
| [normalized_ema_accumulate8](../../runs/study_040_c3_nersi_target_tuning/20260912T084712Z_03c2fc83df18_normalized_accumulation_reference/summary.json) | 10,000 | 12.4088 | 11.7601 |

この条件ではmixupも4倍の学習予算も正規化SNRを改善しなかった。
勾配蓄積8回・実効128 profilesは正規化SNRを0.0876 dB改善したが、15 dBには達していない。
比較元50,000更新に対して10,000 optimizer更新・提示profile数1.6倍で、同一計算量比較ではない。
この条件の[GPU復元監査](../../runs/study_040_c3_nersi_target_tuning/20260912T085636Z_03c2fc83df18_accumulation_gpu_audit/result.json)も最大絶対誤差0で一致した。
`normalized_ema`の[GPU checkpoint復元監査](../../runs/study_040_c3_nersi_target_tuning/20260912T025926Z_03c2fc83df18_normalized_gpu_audit/result.json)は
保存済みpredictionと完全一致（RMSE、最大絶対誤差とも0）、再評価も一致した。
GPU共有による競合があるため、実時間を条件間の厳密な速度比較には使用しない。
独立した局所NeRSIモデルの使用は承認済み。単一モデルと同じ比較条件とは扱わない。

### 独立局所モデル

`local_lines8.yaml`はsource-lineを8本ずつの2領域に分け、各5,000更新する。
各領域の座標を独立に[0,1]へ正規化し、モデル・optimizer・sampling RNGを新規作成する。
各モデルはseed=101/201、Fourier=40、encoder=384、latent=96、kernel=3、
MSE、LR=0.001、EMA=0.999、3.0625サンプルの時間整列を使用する。
RMS-IDW場だけは全Oから共通に推定し、T振幅は最終評価以外に使用しない。
領域は重複・欠落なく全source-lineを覆い、境界を平均せず担当モデルの予測を連結する。
全checkpointのstrict再推論を確認してから全Tの物理・正規化SNRを評価する。
単一モデル用summaryとは分けて保存し、総parameter数・update数は全モデルの合計を報告する。

| Candidate | Models | Total updates | Normalized SNR [dB] | Physical SNR [dB] |
|---|---:|---:|---:|---:|
| [local_lines8](../../runs/study_040_c3_nersi_target_tuning/20260912T064822Z_03c2fc83df18_local_lines8/metrics.json) | 2 | 10,000 | 11.6881 | 11.1397 |
| [local_lines4](../../runs/study_040_c3_nersi_target_tuning/20260912T065248Z_03c2fc83df18_local_lines4/metrics.json) | 4 | 20,000 | 11.1980 | 10.6453 |
| [local_lines1](../../runs/study_040_c3_nersi_target_tuning/20260912T070016Z_03c2fc83df18_local_lines1/metrics.json) | 16 | 80,000 | 10.4654 | 9.9616 |
| [local_lines8_large_steps20000](../../runs/study_040_c3_nersi_target_tuning/20260912T072852Z_03c2fc83df18_local_lines8_large_steps20000/metrics.json) | 2 | 40,000 | 12.3789 | 11.7461 |
| [local_lines8_large_accumulate8](../../runs/study_040_c3_nersi_target_tuning/20260912T075402Z_03c2fc83df18_local_lines8_large_accumulate8/metrics.json) | 2 | 10,000 | 12.0619 | 11.4127 |
| [local_lines8_large_fitted_shear](../../runs/study_040_c3_nersi_target_tuning/20260912T083434Z_03c2fc83df18_local_lines8_large_fitted_shear/metrics.json) | 2 | 40,000 | 12.3626 | 11.7225 |

領域幅だけを変えた3条件では、狭い領域ほどT補間が悪化した。
全28モデルのcheckpoint再推論は保存予測と完全一致し、入力lockは凍結Study 037と一致する。
`local_lines8_large_steps20000`は`normalized_ema`の大型architecture・正規化・時間整列を保持し、
8本単位の2モデル、各20,000更新を設定する。

`local_lines8_large_accumulate8`は同じ大型モデルでmicro-batch=16 profiles、
`gradient_accumulation_steps=8`（実効128 profiles）、各5,000 optimizer更新を使う。
1更新分を非復元抽出してからmicro-batchへ分け、O trace数に比例したloss重みで勾配を加算する。
Adam、LR schedule、EMAはmicro-batchごとではなくoptimizer更新ごとに1回進める。
比較元の各20,000更新に対して、optimizer更新数は1/4、提示profile総数は2倍。
LR=0.001、EMA=0.999、モデル、seed、時間整列は固定する。純粋な同一計算量比較ではない。
勾配蓄積数を省略した既存configは1として動作する。
局所モデルの最高正規化SNRは12.3789 dBで、単一モデルの勾配蓄積条件12.4088 dBを下回り、15 dBは未達。
今回の勾配蓄積条件は正規化SNRが0.3170 dB、物理SNRが0.3334 dB悪化した。
学習時間は約1,104秒から2,054秒、trace提示回数は4,114,763から8,242,309へ増加した。
GPU peak allocatedは両方とも約13,165 MiB（推論・復元を含むrun全体の最大値）。

`local_lines8_large_fitted_shear`は通常batchの大型20,000更新条件に対し、
時間ずれだけを各領域のOペア相関から推定する。
距離8〜16 receiver-yセルの観測ペアをunit RMS化し、候補shearの平均相関を最大化する。
選択値・全候補相関・Oペア数をmetadataへ保存し、各checkpointに実際の時間整列を保持する。
T振幅を候補選択へ使用しない。推定用ペアがない領域はfail-fastとし、別設定へ切り替えない。
前半・後半の選択shearは3.078125 / 3.046875サンプルで、固定3.0625の比較元を上回らなかった。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python studies/study_040_c3_nersi_target_tuning/run_local.py \
  --config studies/study_040_c3_nersi_target_tuning/local_lines8.yaml \
  --output "runs/study_040_c3_nersi_target_tuning/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_local_lines8"
```

```bash
bash scripts/run_c3_nersi_target_tuning.sh normalized_ema
bash scripts/run_c3_nersi_target_tuning.sh normalized_ema_mixup normalized_ema_steps200000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python studies/study_040_c3_nersi_target_tuning/evaluate_normalized_reference.py \
  --run runs/study_040_c3_nersi_target_tuning/RUN_ID/normalized_ema \
  --output "runs/study_040_c3_nersi_target_tuning/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_normalized_reference"
```

### 座標・最適化の条件

追加比較は`aligned_fractional30625`を基準とする5条件（各50,000 updates）。
`aligned_cosine`はLRを0.001から0.00001へcosine減衰、
`aligned_ema0999`はdecay=0.999のEMA、`aligned_nyquist`は軸別帯域制限、
`aligned_cartesian`はCartesian profile座標、
`aligned_cartesian_nyquist_cosine_ema`は4変更の組合せ。結果判定は全Tの物理SNRで行う。
追加の`aligned_ema09999`はEMA decayだけを0.9999へ変更する。
`aligned_nyquist_dense`と`aligned_cartesian_nyquist_dense_cosine_ema`は、
それぞれの帯域制限条件からfrequency_baseだけを1.25→1.02へ変更する。
160個のFourier周波数を帯域内へ密に置く目的で、学習予算は各50,000 updatesのままとする。

Cartesian profile入力は実幾何の`[CMP-x, CMP-y at first receiver, half-offset-x]`を
全profileのgeometry boundsで[0,1]へ正規化する。half-offset-yは既存decoder軸で表す。
各receiverでは`CMP-y = anchor CMP-y + half-offset-y - first half-offset-y`となる。
これはSIRENの4空間座標点入力とは異なり、NeRSIの面生成decoderを保持する座標化である。
source配置の不規則性を保持し、アフィン近似しない。振幅やmaskは座標fitに使用しない。

帯域上限は各入力軸の最大隣接正規化座標差Δに対してπ/Δ（angular frequency）。
上限を超えるFourier特徴のcos/sin両方をゼロとし、feature幅と初期重み乱数列を維持する。
これはencoder入力特徴の制限であり、非線形decoder出力の厳密な帯域保証ではない。
評価波形、time軸、receiver-y出力軸へlow-passを掛けない。
EMAは最初のoptimizer更新後の重みで初期化し、以降各更新で平均、
最終EMA重みだけをpredictionと`final.pt`に使用する。Tによるcheckpoint選択はしない。

```bash
bash scripts/run_c3_nersi_target_tuning.sh \
  aligned_cosine aligned_ema0999 aligned_nyquist aligned_cartesian \
  aligned_cartesian_nyquist_cosine_ema
bash scripts/run_c3_nersi_target_tuning.sh \
  aligned_ema09999 aligned_nyquist_dense aligned_cartesian_nyquist_dense_cosine_ema
```

- benchmark: `c3_sl25_40_random80_observed_only_v1`
- dataset: `seg_c3_na`
- case: `c3_benchmark_test_random_trace_80_seed42`
- volume: `c3_benchmark_test_random_trace_80_seed42_volume`
- selection: time [0,384]、source_line [25,41]、shot_in_line [28,60]、
  relative_receiver_x [0,8]、relative_receiver_y [18,50]
- shape: [384,16,32,8,32]、O=26,362、T=104,710
- O-only Global RMS: 9.257964353671534
- nominal 80% random mask、crop内のrealized missing fractionは79.8874%
- exact missing fraction: 104710 / 131072 = 0.7988739013671875

T評価による設定選択をユーザーが許可した探索であり、独立な汎化評価ではない。
T真値を学習ラベル・正規化・時間整列のfitには使わない。
未調整の他手法との公平な優劣比較とは扱わない。
Study 036/037と凍結runは変更せず、POCS/DRRは変更・再実行しない。
全候補の成果物を保持し、別条件への切り替えを同一runの再試行として隠さない。

MSE、既知の入力artifact、outer mask、model seed=101、sampling seed=201、
NeRSIの層構成、全Tの物理振幅評価、Oのexact reinsertionを維持する。
幅・Fourier成分・kernel・学習率・batch・予算は各resolved configに明示する。
checkpointは各runの固定最終stepであり、run内のbest checkpointは選択しない。

## 時間整列

参考は[SIREN Study 031](../study_031_c3_siren_10db/README.md)の時間shear。
別validation領域のSNRを現在のTの到達値とは扱わない。
[Study 015](../study_015_strong_fit_budget_extension/README.md)の20 dB超は
学習済みtraceへのfitであり、T補間の結果ではない。

現在のOの隣接receiver-yペア5,153組では、+3サンプル整列により、
cosine類似度中央値が-0.6604から0.9106になる。他の3空間軸の最良整数ずれは0。
[O-only診断](../../runs/study_040_c3_nersi_target_tuning/20260911T163157Z_03c2fc83df18_observed_alignment_diagnostic/summary.json)を保持する。

設定は `time_alignment: {receiver_y_shift_samples_per_cell: 3, boundary: circular}`。
各traceを`3 * (local_receiver_y - floor(receiver_y_count / 2))`サンプル循環シフトする。
全384サンプルとtrace energyを保持し、T格納値は変換前に除外する。
予測後は逆シフトし、Oをexact reinsertして全Tを元の物理振幅で評価する。
周期端点を仮定する処理であり、zero paddingや物理的な外挿ではない。
補正量・境界規則はresolved config、metadata.method_details、
checkpoint.preprocessingへ記録し、strict復元時にも照合する。

`aligned_zero_pad`は同じ補正量で境界を`zero_pad`に変える条件。
shiftの最小値を差し引いて全traceを配置し、必要な93サンプルを8の倍数96へ切り上げ、
モデルの時間gridを384から480へ拡張する。元の全サンプルは保存し、逆変換ではpaddingだけを除く。
paddingのゼロは実観測ではなく、training metadataで別計数する。
訓練MSEは480サンプルのゼロ境界制約を含むため、循環境界と同じ目的関数ではない。
物理評価と入力lockは元の384サンプルのまま。NeRSIの層構成は変えないが、派生したmodel profile_shapeとparameter_countは変わる。

`aligned_fractional30625/29375`は補正量を3.0625/2.9375サンプル／セルとする。
`fourier_periodic`はfloat64 FFTで各周波数の位相を回転し、入力dtypeへ戻す。
DCと実数Nyquist係数は保持する。実数Nyquist係数には任意の単位複素位相を与えられないためである。
この規則ではtrace energyと逆変換が丸め誤差の範囲で保存される。
周期・帯域制限を仮定する小数時間ずれであり、非線形time warpではない。
補正量はconfigに固定し、Tの波形に合わせてfitしない。

## 振幅正規化とデータ拡張

IDW variantはOを各traceのRMSで正規化し、TではOから補間したRMSを掛けて復元する。
基準のGlobal RMSとは実効的なtrace重み付けが異なる。
距離は4空間軸のindex差にaxis_scalesを掛けたEuclidean距離で、
Chebyshev-radius stencil内の全Oを逆距離重み付き平均する。
OのRMSは保持し、zero-energy Oの訓練divisorは1。supportのないTは拒否する。
checkpointには実際の物理RMS場を保存する。

[IDW RMS後処理の比較](../../runs/study_040_c3_nersi_target_tuning/20260911T160859Z_03c2fc83df18_idw_postprocessing/summary.json)は追加学習runではない。
元の11.6646 dBを全5有効候補が下回り、最高は11.4229 dBだった。
support不足の1候補の失敗記録も保持する。

50,000-stepの`aligned_fourier160`への[同じIDW補正](../../runs/study_040_c3_nersi_target_tuning/20260911T182626Z_03c2fc83df18_aligned_idw_postprocessing/summary.json)も、
全5有効候補が元の12.1025 dBを下回り、最高は11.8382 dBだった。

座標jitterは独立seed=501で既知の解析領域内の座標を摂動し、Oラベルを維持する。
近傍profile mixupは隣接receiver-xの両profileに共通するcomplete O traceだけを使う。
同じ係数で座標と波形を線形混合し、元のO batchを保持して合成profileを追加する。
共通Oのない組は追加しない。係数は独立seed=501で[0,max_fraction)から一様に生成する。
両者は局所平滑性を仮定した近似で、正確な波動対称性ではなく、併用しない。
推論時には拡張しない。supervised_trace_presentationsには合成traceも含める。
`aligned_mixup020/050`はaligned_largeへ拡張だけを加えた条件である。
`aligned_fourier16_jitter005/020`はFourier成分16の条件へ0.05/0.2セルのjitterを加え、
近傍profile間の平滑性による汎化を調べる。

## 復元・診断

最高候補`aligned_ema0999`の[GPU監査](../../runs/study_040_c3_nersi_target_tuning/20260912T004118Z_03c2fc83df18_encoding_ema_gpu_audit/aligned_ema0999/result.json)は、
strict checkpoint再推論と保存predictionが完全一致（RMSE/max abs errorとも0）し、再評価も一致した。
Cartesian・帯域制限・cosine・EMAの[組合せ監査](../../runs/study_040_c3_nersi_target_tuning/20260912T004118Z_03c2fc83df18_encoding_ema_gpu_audit/aligned_cartesian_nyquist_cosine_ema/result.json)も完全一致した。

`aligned_fractional30625`の[GPU監査](../../runs/study_040_c3_nersi_target_tuning/20260911T184806Z_03c2fc83df18_fractional_gpu_audit/result.json)は、
strict checkpointから全予測が完全一致し、独立再評価も一致した。

`aligned_steps200000`の[GPU監査](../../runs/study_040_c3_nersi_target_tuning/20260911T184112Z_03c2fc83df18_long_and_padding_gpu_audit/aligned_steps200000/result.json)と、
`aligned_zero_pad`の[GPU監査](../../runs/study_040_c3_nersi_target_tuning/20260911T184112Z_03c2fc83df18_long_and_padding_gpu_audit/aligned_zero_pad/result.json)は、
strict checkpointから全予測が完全一致し、独立再評価も一致した。

`aligned_large`の[GPU復元監査](../../runs/study_040_c3_nersi_target_tuning/20260911T164610Z_03c2fc83df18_aligned_gpu_audit/result.json)では、
strict checkpointから全予測が完全一致し、保存出力の独立再評価も一致した。
[T誤差の空間分布](../../runs/study_040_c3_nersi_target_tuning/20260911T165233Z_03c2fc83df18_target_error_diagnostic/summary.json)は診断専用で、
部分領域のSNRを到達判定には使わない。

`idw_base`の[GPU監査](../../runs/study_040_c3_nersi_target_tuning/20260911T161415Z_03c2fc83df18_idw_audit/gpu/result.json)は完全一致した。
一方、[CPU監査](../../runs/study_040_c3_nersi_target_tuning/20260911T161415Z_03c2fc83df18_idw_audit/cpu/result.json)は
rtol/atol=1e-6でfailed_tolerance、最大振幅差0.0378。
[数値診断](../../runs/study_040_c3_nersi_target_tuning/20260911T161618Z_03c2fc83df18_idw_numerics/summary.json)では、
元runのTF32を無効化したGPU予測がCPU予測へ近づき、CPUのSNR差は約-9.42e-6 dBだった。
元runの数値設定・成果物・許容誤差は変更せず、CPUの不一致を成功とは扱わない。

## 実行

repository rootで実行する。wrapperはOMP/OPENBLAS/MKL_NUM_THREADS=1、
CUDA_VISIBLE_DEVICES未設定、configのcuda:1を使い、共通poc check後に
各候補を別processで逐次実行する。新しいrun rootとソースtarを作成し、既存runを再利用しない。

```bash
bash scripts/run_c3_nersi_target_tuning.sh aligned_fourier80 aligned_fourier160
bash scripts/run_c3_nersi_target_tuning.sh aligned_mixup020 aligned_mixup050
bash scripts/run_c3_nersi_target_tuning.sh aligned_encoder4096 aligned_latent384 aligned_kernel9
bash scripts/run_c3_nersi_target_tuning.sh aligned_zero_pad aligned_lr0003 aligned_kernel9_lr0001 aligned_steps200000
bash scripts/run_c3_nersi_target_tuning.sh aligned_fractional30625 aligned_fractional29375 aligned_fourier16_jitter005 aligned_fourier16_jitter020
bash scripts/run_c3_nersi_target_tuning.sh aligned_kernel1 aligned_kernel3
bash scripts/run_c3_nersi_target_tuning.sh normalized_ema_accumulate8 aligned_ema_accumulate8 aligned_ema_accumulate16
```

完了runの追加集計は既存artifactだけを検証し、振幅配列やT真値を読まない。
`--run`には完了済みdirectoryを渡す。既存のcomparison出力は上書きしない。

```bash
.venv/bin/python studies/study_040_c3_nersi_target_tuning/summarize_candidates.py \
  --previous-summary runs/study_040_c3_nersi_target_tuning/20260911T191149Z_03c2fc83df18_comparison/summary.json \
  --run runs/study_040_c3_nersi_target_tuning/RUN_ID/CANDIDATE \
  --output "runs/study_040_c3_nersi_target_tuning/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_comparison"
```

## 検証

以下の関連検証を使用する。full suiteは実行しない。

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q \
  tests/unit/test_{local_nersi,nersi_local_time_alignment,normalized_trace_reference,nersi_checkpoints,fixed_step_nersi,nersi_pipeline_config,c3_volume_nersi_data,c3_volume_nersi_prediction,c3_nersi_target_tuning_configs,c3_neural_mse_loss_ablation_configs,c3_poc_stage_1b_freeze}.py \
  tests/integration/test_{interpolate_local_nersi,interpolate_nersi}.py --tb=short
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q \
  tests/unit/test_c3_nersi_target_tuning_summary.py \
  tests/integration/test_nersi_run_audit.py --tb=short
git diff --check
```

凍結記録との実ファイルSHA-256照合は、Study 036の44 runファイル・5 formal configと、
Study 037の20 runファイル・3 formal configで全件一致した。
