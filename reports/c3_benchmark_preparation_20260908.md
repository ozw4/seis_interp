# C3 benchmark準備・固定の報告（2026-09-08）

`c3_benchmark_codex_prompts_v2.zip` の必須工程01〜13を実装し、local C3の
正式20ケースを作成した。別プロセスのread-only検証で、144ファイルのhashと
全caseの物理対応・mask・bindingが一致した。本学習・全手法の補間実験は未実行。
追加工程14の連続欠損条件は、このsuiteに含めていない。

## 確定した入力

| 軸（`VOLUME_AXIS_ORDER`順） | 主test crop | validation crop |
|---|---|---|
| time | `[0,384)` | `[0,384)` |
| source_line | `[25,41)` | `[41,50)` |
| shot_in_line | `[28,60)` | `[32,64)` |
| relative_receiver_x | `[0,8)` | `[0,8)` |
| relative_receiver_y | `[18,50)` | `[18,50)` |
| shape | `[384,16,32,8,32]` | `[384,9,32,8,32]` |

rangeは0-basedで終端を含まない。元の測線番号列がないため、指定された
sail line25〜40はglobal source-line index25〜40と定義した。対応するsource xは
7860〜10260 m、測線間隔160 m、shot間隔80 m、隣接測線のstaggerは40 m。
元番号との同一性や原著の未公開開始位置との完全一致は主張しない。

現物の全time配列は625 samples。使用区間はsample0〜383、物理時刻0〜3.064秒、
sample間隔0.008秒、サンプリングレート125 Hz。時間原点を変更していない。

選択16測線に共通する範囲はshot `[0,96)`、receiver x `[0,8)`、receiver y `[0,68)`。
中央候補の開始点 `(32,0,18)` に対して、既存格子検証がlocal cell
`(source_line=0, shot=28, receiver_x=1, receiver_y=4)` の欠損を検出した。
固定time・測線・shapeを保ち、開始点のManhattan距離、同距離なら辞書順で調べ、
最初の有効候補 `(28,0,18)` を採用した。変更はshot開始点の32→28だけ。
validationは中央候補 `(32,0,18)` のまま成立した。振幅の大小では選択していない。

| QC | test | validation |
|---|---:|---:|
| trace数 | 131,072 | 73,728 |
| 調べた振幅sample数 | 50,331,648 | 28,311,552 |
| 非有限値数 | 0 | 0 |
| RMS（物理振幅） | 9.240818 | 9.939122 |
| 最大絶対振幅 | 153.880005 | 119.197784 |
| 有効なゼロ値の割合 | 23.445545% | 23.173498% |

ゼロ振幅を欠損セルに置き換えていない。QCのRMSやenergyはモデルの尺度には使わない。
[主cropの代表波形・時間energy](../data/processed/c3_na/study_027_c3_na_benchmark/qc/test/crop_qc.png)
と[数値QC](../data/processed/c3_na/study_027_c3_na_benchmark/qc/test/crop.json)を保存した。
代表波形はlocal line/shot/receiver xがすべて0のgather、表示clipは絶対値99 percentile。

## Partition・学習・採点の契約

| partition | 測線range | 元のtrace数 | canonical trace数 |
|---|---|---:|---:|
| train | `[0,25)` | 1,146,816 | 1,146,803 |
| test | `[25,41)` | 740,544 | 740,542 |
| validation | `[41,50)` | 416,664 | 416,664 |

全50測線を重複・隙間なく覆う。重複物理座標15行は既存の最小array_row規約で除外した。
canonical rows・FFID・物理座標が役割をまたがず、全cropが対応partition内にあることを確認した。
共通train poolは1,146,803 canonical rowsとtime `[0,384)`。
学習seedの設定値 `[20260908]` はmask seedから分離し、学習は実行していない。

POCS・DRR・SIREN-5D・CCNet-5Dの共通volume readerと、volume指定必須のGNN入口を用意した。
推論振幅は同じcropのobservedだけ、target波形は採点専用、target座標は利用可能とする。
crop外contextは禁止。CCNetの尺度は許可train pool内のfit領域、GNNは許可train poolの
time `[0,384)`、SIRENはcrop観測値だけから求める。generic partition normalizationは
既存形式を保持する別artifactであり、モデル尺度の契約とは分けて検証する。

主指標はtargetだけの `physical_amplitude_global_snr_db`。
float64で物理振幅の信号energyと誤差energyをそれぞれ全targetにわたって合計し、
`10 * log10(sum(target**2) / sum((prediction-target)**2))` を求める。
contextのないqueryも採点対象に残す。完全再構成・ゼロreferenceは既存のstatus/null-SNR表現を保つ。

## 全ケースの実現欠損率

母集団はcropではなくcanonical partition全体。testは各条件seed42/43/44、
validationはseed142。下表の率は固定crop内の実現値であり、率を合わせるseedの引き直しはない。
whole-FFID原子性はcropだけでなく全candidate集合で検証した。

| partition | kind | nominal | seed | target traces | trace欠損率 | 完全欠損FFIDs | FFID欠損率 |
|---|---|---:|---:|---:|---:|---:|---:|
| test | random_trace | 50% | 42 | 65,276 | 49.801636% | 0 / 512 | 0% |
| test | random_trace | 50% | 43 | 65,434 | 49.922180% | 0 / 512 | 0% |
| test | random_trace | 50% | 44 | 65,607 | 50.054169% | 0 / 512 | 0% |
| test | random_trace | 80% | 42 | 104,710 | 79.887390% | 0 / 512 | 0% |
| test | random_trace | 80% | 43 | 104,841 | 79.987335% | 0 / 512 | 0% |
| test | random_trace | 80% | 44 | 104,766 | 79.930115% | 0 / 512 | 0% |
| test | random_trace | 90% | 42 | 117,877 | 89.933014% | 0 / 512 | 0% |
| test | random_trace | 90% | 43 | 117,912 | 89.959717% | 0 / 512 | 0% |
| test | random_trace | 90% | 44 | 117,999 | 90.026093% | 0 / 512 | 0% |
| test | random_whole_ffid | 50% | 42 | 67,840 | 51.757812% | 265 / 512 | 51.757812% |
| test | random_whole_ffid | 50% | 43 | 61,440 | 46.875000% | 240 / 512 | 46.875000% |
| test | random_whole_ffid | 50% | 44 | 70,144 | 53.515625% | 274 / 512 | 53.515625% |
| test | random_whole_ffid | 80% | 42 | 104,704 | 79.882812% | 409 / 512 | 79.882812% |
| test | random_whole_ffid | 80% | 43 | 102,144 | 77.929688% | 399 / 512 | 77.929688% |
| test | random_whole_ffid | 80% | 44 | 107,776 | 82.226562% | 421 / 512 | 82.226562% |
| validation | random_trace | 50% | 142 | 36,712 | 49.793837% | 0 / 288 | 0% |
| validation | random_trace | 80% | 142 | 58,999 | 80.022515% | 0 / 288 | 0% |
| validation | random_trace | 90% | 142 | 66,302 | 89.927843% | 0 / 288 | 0% |
| validation | random_whole_ffid | 50% | 142 | 38,400 | 52.083333% | 150 / 288 | 52.083333% |
| validation | random_whole_ffid | 80% | 142 | 58,880 | 79.861111% | 230 / 288 | 79.861111% |

## Manifest・生成時の記録

正式manifestは
[benchmark_suite.json](../data/processed/c3_na/study_027_c3_na_benchmark/benchmark_suite.json)。
SHA-256は `6447a35cc7d4de43532ee8e2e0de3245ac0e93d855efa7ff59c98a85e1631cab`。
参照pathの基準はmanifest directory。144ファイルにinterim、partition、QC、train pool、
case/mask/volume、設定snapshotを含み、配列本体は複製しない。

生成開始は `2026-09-08T06:01:41Z`、生成時HEADは
`04f8d2598ae311106c9925197e80e4556b1dad91`、`git_worktree_dirty: true`。
生成時の実装は未commitの作業ツリーに含まれていたため、このHEADだけで生成コードを識別したことにはしない。
現在のstudy設定は実測後の確定値へ更新した。生成に使用した元設定のsnapshotとhash、
生成時のresolved設定はmanifest配下に保持し、後からの文書更新で書き換えていない。
生成後に実装・テスト・実行設定を保存したcommitは `31e88d8`（QCと契約）と
`e7bf297`（suiteとCLI）。これらはmanifestに記録した生成時HEADと区別する。

| 実測したinterimファイル | SHA-256 |
|---|---|
| amplitudes.npy | `7037d750d227a6cfce5ade61d4bc0ca05ac846b294885b8c9d8d945cf2af6e92` |
| dataset.json | `c2a741d5af447b2f413334c047db11a3ef65e9915d3399bc553aa284d5bc187a` |
| time_s.npy | `35ffa4765a7e2877618451d961cdcccb7272ade8e084282754e8cc6c5af1181b` |
| traces.parquet | `cfb8ecb6310adeed54cc00a28c09da5cdb714ba0a064517cca5249879201ef6f` |

## 実行・検証コマンド

作業directoryはrepository root。以下のdry-runは出力作成前に `ready: true`、
不足入力なし・出力衝突なし・20ケースを確認した。続くexecuteは `status: locked`、
独立verifyは同一manifest SHA-256で `status: verified`、20ケースを返した。
現在同じ出力先へ再準備すると既存directoryとして拒否する。

```bash
python -m seis_interp.cli data prepare-c3-benchmark --config studies/study_027_c3_na_benchmark/config.yaml --inputs studies/study_027_c3_na_benchmark/inputs.yaml --json
python -m seis_interp.cli data prepare-c3-benchmark --config studies/study_027_c3_na_benchmark/config.yaml --inputs studies/study_027_c3_na_benchmark/inputs.yaml --execute --plots --json
python -m seis_interp.cli data verify-c3-benchmark --input data/processed/c3_na/study_027_c3_na_benchmark --json
./scripts/verify_seg_c3_na.sh
```

raw検証はSEG-Y 4ファイルすべてサイズとSHA-256が一致、終了コード0。
新しいdataset download、既存artifactの上書き、旧Studyの利用履歴調査は実施していない。

最終チェックの正確なコマンドは以下。ruff checkは成功、format checkは422ファイル変更不要、
pytestは176 passed（12.62秒）、diff checkは成功。full suiteは実行していない。

```bash
ruff check .
ruff format --check .
pytest tests/unit/test_c3_benchmark_contract.py tests/unit/test_c3_benchmark_qc.py tests/unit/test_c3_benchmark_masks.py tests/unit/test_c3_benchmark_metrics.py tests/unit/test_c3_benchmark_cli.py tests/integration/test_c3_benchmark_suite.py tests/integration/test_c3_benchmark_conformance.py tests/unit/test_data_cli.py tests/unit/test_cli.py tests/unit/test_c3_volume_index.py tests/unit/test_c3_receiver_grid.py tests/unit/test_prepare_c3_volume_index.py tests/integration/test_prepare_c3_volume_index.py tests/unit/test_c3_volume_metrics.py tests/unit/test_trace_graph_metrics.py tests/unit/test_relational_trace_graph_prediction.py
git diff --check
```

synthetic検証は中央の穴に対する決定的調整、合法ゼロ、time原点、canonicalization、
partition、mask、採点、全入口の観測/target対応、targetだけを変更した正規bindingの
二つのfixture、CPU forward、SIREN 3 steps、query batch不変性を含む。
入力・予測がtarget振幅に依存しない一方で採点energyは変わることを確認した。
hash改変・missing file・別caseのvolume・誤った指標・crop外context・部分成功によるlockも拒否した。
これらは入力隔離と準備の検証結果であり、各モデルの補間精度を示すものではない。
