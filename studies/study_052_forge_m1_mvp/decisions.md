# 固定判断

## 既存C3設定からの移行

M1のtest指標を参照せずに設定を固定する。POCS、DRR、CCNet-5D、NeRSIは
study_042の既存設定を使う。NeRSIは時間shearなしの採用設定を使用する。
ReGSIはstudy_044で採用済みのwidth 128・20,000 step・EMA構成を使用する。
原設定のpathとhash、MVP側の変更理由は各手法configへ保存する。
各手法の学習budgetやパラメータ数を等しくする実験ではない。

元のC3比較はMSEだったが、今回の明示仕様に従い学習lossを全て
`masked_trace_relative_mse`へ統一する。既存lossの完全トレース・float64集計・
zero-energy時のdivisor契約を再利用し、epsilonや時間maskは追加しない。
DRRの5 Hz下限は、手法固有band-pass禁止に従い0 Hzへ変更する。
POCSのthreshold、DRRのrank/iteration、neuralの幅・層数・learning rate等をM1品質で調整しない。

## 4,001サンプルへのshape変更

CCNetは全時刻を含むpatchを使用し、C3のpatchサンプル総数以下になるまで
最長の空間辺を半分にする。同長なら最初の軸を選ぶ。
ReGSIのbatchはC3の時間長比だけで縮小する。real/gridに同じ変更を適用する。
既存ReGSIの内部zero paddingと出力cropで奇数長を扱い、lossと出力は全4,001サンプル。
trace-wise正規化や空間内挿は行わない。

## NeRSIの実座標条件

既存NeRSIは3座標から規則的なtime×receiver断面を出力するC3専用実装であり、
receiverの実位置4成分をそのまま扱えない。
MVPでは同じFourier mapping、2層encoder、3段の畳み込み・PixelShuffle decoderを用い、
実source/receiverの局所連続4座標から1トレースを出力する入出力shapeの拡張を固定した。
座標は格子基底でアフィン変換し、整数への丸めや範囲内へのclipを行わない。

3段の2倍PixelShuffleに必要な最小8列・4,008時刻の出力canvasを作り、
最初の1列・4,001時刻だけをlossと予測に使用する。余分な出力に教師値を作らない。
これは明示的なtrace-wise NeRSI適応であり、C3のprofile-wise実装と同一条件でも
原論文の完全再現でもない。4D実座標を維持するために必要な入出力変更として扱う。
この適応の是非をM1のtest結果を使って選び直さない。

## Maskと表示断面

study IDは`study_052_forge_m1_mvp`に固定した。
SHA-256への文字列化はUTF-8のcompact JSON配列とし、mask seed・cell IDは整数。
digest同値時にはcell IDをtie-breakとする。rankは1始まり。
人工testに入ったQC要確認traceは母集団から削除せず、clean-targetの主評価からだけ分離する。

common-sourceとcommon-receiverは幾何の中央に近い格子cellを事前固定した。
0–4 sの全時刻を表示する。表示振幅上限はobservedの絶対振幅percentileで決め、
表示のためにもtest振幅でscaleをfitしない。

## 準備と本番の区別

preflightはPOCS/DRRの1周波数slice、およびneuralの1 diagnostic updateと
先頭query batch/tileに限定する。neuralは実際の4,001サンプルを使用し、
重みと予測は破棄する。test振幅・test metric・checkpoint選択にはアクセスしない。
preflightの時間・peak memoryは限定処理の計測であり、全本番runの上限保証ではない。
正式なMVP判定は6本の本番runと評価が終了してから行う。

test振幅を開く前に全6手法のmanifest、predictionのhash・shape・cell ID・有限値、
全予測の定数崩壊、ReGSI初期状態の一致を検査する。1手法でも失敗・未完了・
不正ならtestを開かず停止する。部分成功の指標を先に開示してから失敗手法の
メモリ設定等を変更することを防ぎ、完全なmatrixだけを一括評価する。

指標は元振幅で計算し、既存Torch loss、独立したNumPy計算、保存したtrace別表の
再集計、波形からの第二計算を照合する。Pearson相関が未定義な定数予測は
件数を明示し、相関のpercentileは定義可能なtraceに対して保存する。
完全一致時の無限SNRはJSONのnullとstatusで表現する。
paired差は`error_regsi_real - error_regsi_grid`。投影距離四分位は同一test集合上で定め、
因果効果・信頼区間・有意差へ解釈しない。

- 実装hashとsource snapshotには `src/seis_interp/**/*.py` に加え、実際の起動入口 `scripts/forge_m1_mvp.py` を含める。ファイル集合も検証し、run/preflightは子プロセス起動前に改変を拒否する。変更後のprepareとpreflightはclean worktreeから再生成する。
