# study_027_c3_na_benchmark

SEG C3 NAの同じ観測・target領域で、POCS、DRR、SIREN-5D（独自実装）、
CCNet-5D、RelationalTraceGraphInterpolator（提案手法）を比較するためのデータ契約。
工程01〜13を実装し、local C3の正式20ケースを作成・独立再検証済み。
抽出の参照は *5-D Seismic Data Interpolation by Continuous Representation* とし、
ISRは比較しない。本学習・全手法の補間実験はこの準備の対象外。

主評価のshapeは `[384,16,32,8,32]`。軸は既存の `VOLUME_AXIS_ORDER` を正本とする。

| 軸 | 確定range（0-based、終端を含まない） |
|---|---|
| time | `[0,384)` |
| source_line | `[25,41)` |
| shot_in_line | `[28,60)` |
| relative_receiver_x | `[0,8)` |
| relative_receiver_y | `[18,50)` |

現物のtimeは0〜3.064秒、間隔8 ms（125 Hz）、最終sampleは383。
入力の物理時刻をそのまま使う。元の測線番号列がないため、sail line25〜40は
global source-line index25〜40と定義した。全50測線中のこの16本を固定する。
元番号列がある別入力では対応を一度確定して共有し、番号変換で測線集合を変えない。

未指定の開始点は、選択測線に共通する幾何範囲の中央を初期値とした。
shot `[32,64)`、receiver x `[0,8)`、receiver y `[18,50)` の中央候補には
1セルの穴があったため、Manhattan距離、同距離なら辞書順で有効な開始位置を選んだ。
shot開始位置のみ32から28へ変更した。時間・測線・shapeは変更していない。
原著の未公開開始点はこの規則で補い、原著との完全一致は主張しない。

| partition | source-line range | canonical trace数 |
|---|---|---:|
| train | `[0,25)` | 1,146,803 |
| test | `[25,41)` | 740,542 |
| validation | `[41,50)` | 416,664 |

重複物理座標15行は既存の最小array_row規約でcanonical化した。各役割は非空で、
全測線を重複・隙間なく覆い、canonical traceとFFIDが役割をまたがない。
共通train poolはcanonical train rowsとtime `[0,384)`。
validation cropは `[384,9,32,8,32]` で、測線 `[41,50)`、shot `[32,64)`、
receiverとtimeは主cropと同じ範囲を使う。過去利用は調査していない。

| 手法 | 学習情報 | 推論時の振幅 |
|---|---|---|
| POCS / DRR | supervised学習なし | 同じcropのobserved traceのみ |
| SIREN-5D | 対象volumeごとにobserved値のみでfit | 同じcropのobserved traceのみ |
| CCNet-5D / GNN | 共通canonical train rowsとtime `[0,384)` | 同じcropのobserved traceのみ |

target座標は利用可能。target波形は採点専用で、正規化・入力・近傍選択・学習・
停止判定へ渡さない。GNNを含めcrop外contextは禁止する。QC統計はモデルの尺度に
使わない。CCNetの既存RMSは許可train pool内のfit領域、GNNは許可train pool、
SIRENはcrop観測値だけで求める。CCNet/GNNの本学習と凍結推論は後続の実験で行う。
suite用readerの使い方は[実装文書](../../docs/c3_benchmark.md)を参照。

testはrandom trace欠損50/80/90%、whole-FFID欠損50/80%、各seed42/43/44の15ケース。
validationは同じ5条件をseed142で作った。maskはcanonical partition全体を母集団とし、
同一partition内ではcropを共有する。実現率を合わせるseedの引き直しはしていない。
学習seed `[20260908]` はmask seedと分離した設定値であり、学習は未実行。

主指標は `evaluation_target` の `physical_amplitude_global_snr_db`。
物理振幅単位のtarget全体で信号energyと誤差energyをそれぞれ合計してから
`10 * log10(sum(target**2) / sum((prediction-target)**2))` を求める。
traceごとのdB平均を主指標にしない。contextなしqueryもtargetに残す。

正式入力は[benchmark_suite.json](../../data/processed/c3_na/study_027_c3_na_benchmark/benchmark_suite.json)。
SHA-256は `6447a35cc7d4de43532ee8e2e0de3245ac0e93d855efa7ff59c98a85e1631cab`。
144ファイルのhash、全caseのbinding、物理対応、mask seed、学習範囲を再検証済み。
`config.yaml` と `inputs.yaml` は確定値・参照を記録する現在の設定であり、
生成時の設定はmanifestが束ねる `sources/` とresolved設定へ保存してある。
生成時のHEADとdirty状態を後からの文書更新で書き換えない。

[準備報告](../../reports/c3_benchmark_preparation_20260908.md)に各caseの実現欠損率、
QC、生成時の記録、実行・検証コマンドを記載した。
[代表波形と時間energy](../../data/processed/c3_na/study_027_c3_na_benchmark/qc/test/crop_qc.png)
も保存した。生成配列・mask・volumeはGitへ追加しない。既存出力への上書きは拒否する。
工程14の追加連続欠損は今回の20ケースに含めていない。
