# C3 SIREN-5D: physical target SNR above 10 dB

固定したQC済みvalidation caseで、SIRENの物理振幅global SNRを厳密に10 dB超へ改善する。
入力はStudy 029の437本除外済みsuite、比較基準はStudy 030のper-trace RMS＋IDW方式である。
評価対象は同じ58,999本・22,655,616 sampleとし、cropやmaskは変えない。

現在は固定shearを明記したSIREN参照モデルを採用する。採用run
`20260909T021337632365Z_1819109f28e1_siren` はomega30・時間倍率12・
shear 0.0006 s/mで、物理target SNR **11.3422 dB**、target RMSE **2.6929**である。
これは補助変換付き条件での10 dB超であり、shearなしの10 dB目標は**未達**のままである。
観測だけから係数を求めることと、他手法と前処理条件が同じであることは別の要件であり、
同一前処理の優位性は主張しない。GNNを含む他手法へ固定shearを導入しない。
[現在の採用判断](../../results/study_031_c3_siren_10db/20260909T055026000000Z_e39df16d563c_shear_reference_adoption/adoption_decision.json)
に採用範囲と出典を記録する。全対象保存予測の再採点は元指標と一致し、事前固定16 targetの
CPU checkpoint復元も所定閾値を満たした。全targetの復元bit一致を意味するものではない。
固定validation caseの結果であり、test partitionの精度を示さない。

固定shearなしの初期時間重み・観測envelope損失の追加3条件も比較を完了した。
既存基準を含む4条件ではenvelopeのみのBが **-1.6301 dB**、target RMSE **11.9903**で
最も良かったが、10 dB超は未達である。全対象保存予測の独立再採点は一致した一方、
A/B/ABの通常CPU checkpoint復元は事前の誤差閾値を満たさなかった。
この追加3モデルは採用しない。現在の`config.yaml`は上記の補助変換付き参照条件である。
[時間方向学習の比較レポート](../../reports/c3_siren_time_learning_20260909.md)に結果と監査範囲を示す。

観測14,729本だけから波形を学習する。各観測traceの384 sampleを自身のRMSで正規化し、
欠測位置の物理尺度は観測8近傍のIDWで求める。距離尺度はsource X/Y・relative receiver X/Yの
`[160,80,40,40] m`、power2を維持する。真値波形は評価境界だけで読み、尺度・座標fitや
学習batchへ渡さない。validation指標による構成選択は行うが、test partitionは使わない。

現在の採用構成はCartesian CMP＋半オフセットの5入力、幅256・4層・omega30/30、
観測全14,729本の完全波形を1更新に使うAdamである。5,000更新、学習率1e-4一定、
時間座標倍率12、`relative_receiver_y_time_shear_s_per_m: 0.0006`とする。
座標に `tau = time_s + 0.0006 * relative_receiver_y_m` を用いるが、元の384時刻の
物理波形を学習・予測し、波形の再サンプリングや評価窓の変更は行わない。
係数は観測だけの事前診断から決め、checkpointへ座標metadataとして保存している。

`training.batch_size=262144`は1forwardの最大point数である。完全trace batchが
これを超える場合はsample数に応じて勾配を蓄積し、全14,729本につき1回だけ
optimizerを更新する。予測forwardの上限は65,536点である。

比較を完了した追加3条件は以下の通り。いずれもshear 0の対照モデル・入力・尺度を
維持し、既存run `20260909T030400486453Z_1819109f28e1_siren` の
target SNR -3.1486 dBを基準にする。条件は専用configに保持する。

| 条件 | 専用config | 初期時間入力重み | 観測envelope損失 |
|---|---|---|---|
| A: time_init3 | [config_time_init3.yaml](config_time_init3.yaml) | `training.initial_time_weight_scale: 3.0` | なし |
| B: envelope | [config_envelope.yaml](config_envelope.yaml) | 標準初期化 | `weight: 1.0`, `sigma_samples: [4.0,8.0]`, `decay_steps: 2500` |
| AB: time_init3_envelope | [config_time_init3_envelope.yaml](config_time_init3_envelope.yaml) | `training.initial_time_weight_scale: 3.0` | Bと同じ |

時間重みの変更は初期化時だけで、omega30/30・time scale12・shear 0は維持する。
envelope項は観測波形だけから求め、`training.loss: l2`を残した補助損失とする。
係数λはstep1で1.0、step2500で0へ線形に減衰し、以降は0。
`train_loss`・`waveform_mse`・`weighted_envelope_mse`は100更新の報告区間平均、
`envelope_weight`は区間末λとして記録する。step2500のλは0でも、その直前の更新を
含む区間の重み付きenvelope平均は正になり得る。重みなしenvelope MSEは独立保存せず、
区間末λから逆算しない。最後のstep5000のMSEと最後100更新の平均も区別する。
λが正の更新では完全trace単位にmicrobatchを分け、682本×384=261,888点を上限とする。
通常のMSE経路では既存の262,144点上限を維持する。論理的な提示点数は同じだが、
microbatch分割と加算順の違いがあるため浮動小数点のbit一致は主張しない。
このため、数学的な損失項の効果だけを完全に分離した比較でもない。

各条件は10-step preflightモデルを破棄し、同じseed20260908の新規初期化から、
観測全14,729本×384点、一定Adam1e-4で5,000更新を完了した。予測は65,536点/forward、
1 action最大7,200秒でCUDA1上を逐次実行した。各条件の累積point数は28,279,680,000。
品質による自動再試行やtestの使用はない。全58,999 target・22,655,616 samplesを採点し、
入力ハッシュ・行ID・尺度一致、有限予測、観測再挿入差0を確認した。結果は1 seed・
固定validationに限られる。これらのshearなし追加条件は10 dB超・復元監査通過の
採用基準を満たさなかった。実行前の詳細判断は[decisions](decisions.md#2026-09-09--predeclare-three-shear-free-initialization-and-envelope-loss-conditions)に記録する。

ユーザーが10 dB超の改善を指定したため、Study 030の一回限り・2,000更新という診断条件は
このstudyへ継承しない。各構成の条件と判断理由を実行前に記録し、失敗・未達のrunも保持する。
採用指標は予め定めた各runの最終予測で判定し、全対象を再採点して確認する。

```bash
python scripts/run_c3_first_results.py \
  --config studies/study_031_c3_siren_10db/config.yaml \
  --inputs studies/study_031_c3_siren_10db/inputs.yaml \
  --action siren --preflight --execute
python scripts/run_c3_first_results.py \
  --config studies/study_031_c3_siren_10db/config.yaml \
  --inputs studies/study_031_c3_siren_10db/inputs.yaml \
  --action siren --execute
```

ユーザー指定のshear有無の対照実験は完了している。上記の`config.yaml`は
採用runと同じshear 0.0006のnative設定を生成する。
[shear 0の対照config](config_omega30_time12_shear0_batch14729_5k.yaml)も保持する。
品質未達による自動再試行は設定していない。
再実行時は1 actionを120分上限の別processで実行する。
元のsuite、旧run、SEG-Y、interim配列を上書きしない。

全対象評価は以下の通りである。表の値は小数4桁に丸めている。

| 条件 | 物理target SNR (dB) | target RMSE | 観測再挿入前のモデルRMSE | 元run |
|---|---:|---:|---:|---|
| Study030、random points・2,000更新 | -0.0002 | 9.9389 | 9.9414 | `20260909T003230820286Z_1819109f28e1_siren` |
| Cartesian5・435完全trace・50,000更新・時間倍率1 | -0.7700 | 10.8599 | 9.0161 | `20260909T005658230199Z_1819109f28e1_siren` |
| Cartesian5・435完全trace・50,000更新・時間倍率4 | -2.5383 | 13.3119 | 5.0725 | `20260909T010327892987Z_1819109f28e1_siren` |
| Cartesian5・435完全trace・5,000更新・時間倍率4 | -1.7051 | 12.0943 | 6.7069 | `20260909T011656120058Z_1819109f28e1_siren` |
| Cartesian5・14,729完全trace・5,000更新・時間倍率4 | -2.9480 | 13.9549 | 5.0412 | `20260909T011832015888Z_1819109f28e1_siren` |
| 同上＋receiver-y時間shear 0.0006 s/m（補助あり診断） | 7.8831 | 4.0102 | 3.3105 | `20260909T015011064447Z_1819109f28e1_siren` |
| 同上＋omega30・時間倍率12（補助変換付き採用参照） | **11.3422** | **2.6929** | 2.3861 | `20260909T021337632365Z_1819109f28e1_siren` |
| omega30・時間倍率12・14,729完全trace・5,000更新・shear 0（対照） | -3.1486 | 14.2809 | 5.2413 | `20260909T030400486453Z_1819109f28e1_siren` |
| 同条件・A: 初期時間重み3倍 | -3.0870 | 14.1799 | 5.2205 | `20260909T041346896156Z_1819109f28e1_siren` |
| 同条件・B: 観測envelope損失 | -1.6301 | 11.9903 | 5.2220 | `20260909T043544755153Z_1819109f28e1_siren` |
| 同条件・AB: 初期時間重み3倍＋観測envelope損失 | -2.7047 | 13.5693 | 5.5394 | `20260909T050203109381Z_1819109f28e1_siren` |

ユーザー指定の完全trace batch拡大調査では、435・1,740・6,960・全14,729本/updateと、
forward最大262,144・524,288点の8条件がすべて実行できた。全観測batchでも
最大forward262,144点ならpeak allocated memoryは2.7630 GBに収まった。
過去Study014の435本/updateは観測プール全量だったが、今回は2.9534%に当たる。
435本と全14,729本をそれぞれ5,000更新で比較し、更新数と総提示点数の両方を記録した。
435本では観測RMSE6.7069・target SNR-1.7051 dB、全本では観測RMSE5.0412・
target SNR-2.9480 dBとなり、batch拡大だけでは補間精度を改善できなかった。
同じ全観測batchで時間shearだけを加えるとtarget SNRは7.8831 dBに改善した。
全対象の独立再採点でも一致し、targetのuncentered cosineは-0.0336から0.9153へ
改善した。予測振幅の大きさはほぼ同じままで、波形の一致が改善したことを
[座標調査レポート](../../reports/c3_siren_coordinate_investigation_20260909.md)にまとめた。
採用した補助変換付きomega30・時間倍率12条件では、この時間shearを維持した。
targetのuncentered cosineは0.9626へ改善した。行IDと保存済み尺度ベクトルは
比較条件間で同一であり、targetの波形一致がさらに改善した結果である。
観測fitの改善を、そのまま欠測箇所の精度改善とは扱わない。
同じtargetのRMSEとSNRは逆方向には動かない。今回の対照的な変化は、観測と欠測という
異なる領域を比較した結果である。保存予測の独立採点、波形の一致度、資源計測は
[batch調査レポート](../../reports/c3_siren_batch_investigation_20260909.md)を参照する。

同じomega30・時間倍率12でshearだけを除くと、targetのuncentered cosineは
0.9626から-0.0540へ悪化した。予測RMSは9.6059と9.7326で近く、行ID・尺度・
入力ハッシュも同一である。全対象の独立再採点は元指標と完全一致した。
この設定・seed・5,000更新では固定shearが有効だったと結論し、SIREN自体が
その関係を原理的に表現できないとは解釈しない。
この診断結果を共通条件での優位性には使わず、他手法への固定shear導入も行わない。
[shear有無の対照レポート](../../reports/c3_siren_shear_ablation_20260909.md)に比較を記録する。
