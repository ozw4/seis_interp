# FORGE periodic whole-shot50：固定設定5手法比較

標準の5 runはすべて予測・採点を完了した。失敗・未実行は0件。
これはstudy067のFORGE random70で選ばれた設定をperiodic whole-shot50へ移した、1 mask・1 model seedの記述的比較である。

## 実行と入力

- study: `study_068_forge_shot50`、run: [`20261105T000750814115Z_4d0ae2be41a0`](../runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0)。
- branch: `study-068-forge-shot50`。Git base: `4d0ae2be41a049a4ac03fbcae59ad2c8ba679c87`、dirty worktreeの実装を[封印済みsource snapshot](../runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0/source)へ保存。
- 条件の正本: [config.yaml](../studies/study_068_forge_shot50/config.yaml)、[inputs.yaml](../studies/study_068_forge_shot50/inputs.yaml)、run内の[固定コピー](../runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0/study)。
- FORGE 2017、candidate `medium_source_b06_u3400.0_receiver_b13_u1900.0`、空間形状 `[6,32,8,32]`。
- `period=2, observed_phase=0`。元の`source_grid_point`の偶数pointを観測、奇数pointをtestとし、shot内の全eligible receiverを同一roleにした。
- 各line 16 observed / 16 test shot、全体96 / 96 shot。観測24,575 trace、test24,576 trace、eligible trace基準の実欠損率50.0010%。
- natural missingはcell 16597の1セル。観測・教師・採点に含めない。QCは既存フラグを維持し、結果に応じた変更はない。
- testの主評価clean_targetは24,396 trace、副評価all_eligibleは24,576 trace。validation holdoutなし。
- 物理時間0–4 s、1,001 sample、4 ms。既存のanti-alias decimation済み物理波形をhash照合して再利用した。共有500 m/s alignmentのmax delayは1,171 sample、canvasは2,172 sample。
- 観測配列・partition・global RMS・visibilityは新maskから再作成。旧checkpoint・optimizer・予測のimportはない。

## 固定設定と選択元

選択元は[study067 FORGE selection.csv](../results/study_067_forge_c3_common_grid/20261103T231944Z_ba805bd26f6a_analysis/forge/selection.csv)と同runの`workers/*_selection.json`。
FORGE base YAML＋選択cellのgrid overrideを再構成し、保存されたwinner configとの一致を確認した。
全ファイルのhashはinputs.yamlに保存してある。選択cellの保存済みresolved configとも比較し、差分は実行deviceと新しい観測数に応じたReGSI inner mask実効値だけで、他の値は一致した。

| 手法 | 選択cell | 主な選択値 |
| --- | --- | --- |
| ReGSI real | a05_b04 | temporal_dilations=[6,12], neighbors_per_relation=5 |
| CCNet5D grid | a02_b05 | spatial patch=[4,12,4,12], time_dilation=6 |
| profile-wise NeRSI real | a04_b02 | nuclear_norm_weight=0.03, frequency_base=1.075 |
| POCS grid | a04_b01 | window=[192,2,8,4,8], overlap=[96,1,4,3,6], 32反復 |
| DRR grid | a03_b02 | rank=18, window=[4,8,4,8], overlap=[0,4,0,4], 60反復, damping_power=2 |

ニューラル3手法はfresh初期化、model_seed=7301、AdamW、50,000更新、lr=0.001一定、weight_decay=0、64教師trace/update、最終EMA decay=0.999。
ReGSI/CCNetの名目inner maskは0.25。ReGSIは新しい観測数に既存resolverを適用し、6,080 hidden trace、実効値0.2474059003051882となった。
CCNetのbf16・fresh patch・現在重みのforward/backward・既存推論precision、NeRSIのjitter=0.5とcanvas_time_upsampling、POCS K1、DRR Hann合成を維持した。
shadow EMAは主checkpointの選択に使用していない。

## 主・副評価

mean trace SNRは各traceの `10 log10(target energy / residual energy)` の平均。
masked_trace_relative_mseは既存のtraceごとのrelative MSEの平均で、ゼロenergyの既存規約も維持した。
以下は小数4桁表示。全精度値・非有限値の件数は[summary.csv](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/summary.csv)に保存した。全populationでSNRは有限だった。

| 手法 | 状態 | clean SNR (dB) | clean relMSE | all SNR (dB) | all relMSE |
| --- | --- | --- | --- | --- | --- |
| ReGSI | completed | 1.2758 | 0.7991 | 1.3019 | 0.7973 |
| CCNet5D | completed | -0.5667 | 1.1456 | -0.5673 | 1.1458 |
| profile-wise NeRSI | completed | approximately 0 | 1.0000 | 0.0055 | 0.9990 |
| POCS | completed | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| DRR | completed | approximately 0 | 1.0000 | approximately 0 | 1.0000 |

ReGSIのclean mean trace SNRはこの固定比較で最も高かった。POCSは24,576 test traceすべてが厳密なゼロ予測であった。
DRRの最大絶対予測振幅は2.7176e-7で、ほぼゼロ予測だった。NeRSIのclean SNRは8.9466e-7 dBで、G0診断はfailed、最終training lossは0.4609（zero-predictor lossも0.4609）だった。
NeRSIの50,000更新・最終EMA・全test cell予測は完了している。診断のfailedを実行失敗へ置き換えず、予測品質の限界として記録した。
これらの結果を見た追加探索・再学習・反復数延長・設定変更は行っていない。

## 内部／端shot

phase0のpoint31はshot方向の片側支持となる。主評価に含めた上で、内部90 shotと端6 shotを分けた。
完全な4次元外挿判定ではない。clean trace数は内部22,874／端1,522、allは内部23,040／端1,536。
各セルは `mean trace SNR (dB) / relMSE`。これはtraceを重みとする集計である。

| 手法 | clean 内部 | clean 端 | all 内部 | all 端 |
| --- | --- | --- | --- | --- |
| ReGSI | 1.3190 / 0.7917 | 0.6265 / 0.9106 | 1.3448 / 0.7899 | 0.6580 / 0.9077 |
| CCNet5D | -0.5781 / 1.1488 | -0.3958 / 1.0978 | -0.5787 / 1.1490 | -0.3957 / 1.0978 |
| profile-wise NeRSI | approximately 0 / 1.0000 | approximately 0 / 1.0000 | 0.0055 / 0.9990 | 0.0056 / 0.9990 |
| POCS | 0.0000 / 1.0000 | 0.0000 / 1.0000 | 0.0000 / 1.0000 | 0.0000 / 1.0000 |
| DRR | approximately 0 / 1.0000 | approximately 0 / 1.0000 | approximately 0 / 1.0000 | approximately 0 / 1.0000 |

[per_shot.csv](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/per_shot.csv)は96 test shotのsource_cell、元のsource_grid_line/point、元のsource_line/point、eligible/clean数と手法別の値を保持する。
[shot_support.csv](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/shot_support.csv)は上記の集計、[paired_shot_differences.csv](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/paired_shot_differences.csv)は同じshotに対する10組の手法差を保持する。
traceを独立な反復実験の数として扱わない。以下は、まずshot内で集計し、その96 shotを等重みで平均したclean指標であり、上のtrace平均と区別する。

| 手法 | 等重みshot平均 SNR (dB) | 等重みshot平均 relMSE |
| --- | --- | --- |
| ReGSI | 1.2768 | 0.7990 |
| CCNet5D | -0.5669 | 1.1457 |
| profile-wise NeRSI | approximately 0 | 1.0000 |
| POCS | 0.0000 | 1.0000 |
| DRR | approximately 0 | 1.0000 |

全population・内部／端ごとの等重み値は[shot_equal_weight.csv](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/shot_equal_weight.csv)。有意差検定・bootstrapは追加していない。

## 図

- [取得geometry／mask](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/acquisition_mask.png)：実UTMの震源位置と256受振位置、観測／testのshot数とmask率。
- [内部shot 79](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/gather_internal.png)：line2 / point15。
- [端shot 95](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/gather_edge.png)：line2 / point31。

両shotは[学習前のgeometryのみの選択artifact](../runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0/input/gather_selection.json)で固定した。
各shotの全256 eligible receiverをreceiver_cell順に表示し、原記録・5手法予測・残差（予測−原記録）は全て物理時間0–4 s、共通scale ±0.00994393。
scaleは観測波形の絶対振幅99 percentileから定めた。良い結果のshotへ選び直していない。

## 実行時間とメモリ

| 手法 | 秒 | 時間 (h) | CPU RSS peak (GiB) | CUDA allocated peak (GiB) | CUDA reserved peak (GiB) |
| --- | --- | --- | --- | --- | --- |
| ReGSI | 18910.2418 | 5.2528 | 3.9502 | 30.0574 | 88.2012 |
| CCNet5D | 6608.8396 | 1.8358 | 5.3251 | 78.5220 | 85.2559 |
| profile-wise NeRSI | 270.8560 | 0.0752 | 3.1026 | 0.5318 | 0.9570 |
| POCS | 4480.7797 | 1.2447 | 4.0339 | 0.0000 | 0.0000 |
| DRR | 11125.6491 | 3.0905 | 5.3018 | 0.0000 | 0.0000 |

時間は入力load/resolve後から学習／再構成・予測・保存までで、GPU待機とtest採点を含まない。shadow推論の時間は含む。
メモリは各workerで実測したprocess RSS peakとPyTorch CUDA peakで、reservedはallocator予約量。
GPUはH100 NVL×2。ReGSI/NeRSIはcuda:0で逐次、CCNetはcuda:1。既存のCUDA待機・メモリ制御を使用した。
CPUはPOCS 1 thread、DRR 12 workers / 1 thread、OMP/BLAS/MKLは各1。他ジョブへの停止操作は行っていない。
参考runに基づく開始時見通しは約9時間以上だったが、実測の最終採点は2026-11-05 05:28 UTCに完了した。
NeRSIは平均2.0024 profile/updateをdecodeし、学習時間は266.2629秒だった。この実測を過去のrandom70と同条件の速度比較とは扱わない。

## 検証と再現

検証コマンド・exit code・結果の正本は[verification.json](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/verification.json)、実データの照合は[final_audit.json](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/final_audit.json)。
whole-shot原子性、元cell ID、自然欠損除外、phase相補性、共通入力、RMS固定下のtest波形分離、5手法の小fixture、validationなし、採点barrier、失敗NA、完了済みrunの再学習防止を確認した。
共有grid処理は変更しておらず、全test suiteは実行していない。図はラベルと共通scaleも目視確認した。

全入力・予測hash、全test cellの一回被覆、50,000更新checkpoint、保存指標からの再集計、共通mask、96 shotの対応を確認した。
最終の予測確定は05:28:09.068576 UTC、最初の採点完了は05:28:11.027772 UTC。全手法の状態と予測確定を確認した後にだけtest targetを読んだ。

実装の追加はperiodic shot maskとcounts、validationなしのinput assembly、固定比較runnerの予測／採点分離、shot別集計と図表出力に限定した。
既存random-trace＋validationあり経路とモデル・古典手法の学習／計算核は維持した。
実行logは各[workers/*.log](../runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0/workers)と[method_runs/*/execution.log](../runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0/method_runs)。各methodのconfig.resolved.yaml、manifest.json、prediction.npy、cell_ids.npy、neural final.ptをrunに保存してある。

本番と図表のコマンド：

```bash
python scripts/forge_trace_experiment.py prepare --study studies/study_068_forge_shot50
python scripts/forge_trace_experiment.py launch --experiment runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0
python scripts/forge_trace_experiment.py score --experiment runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0
python scripts/forge_trace_experiment.py figures --experiment runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0 --output results/study_068_forge_shot50/<new-result-id>
```

`launch`は既存launchへの重複実行を拒否する。`score`は完了済みmethodを再学習・再採点しない。新規再現にはprepareで別runを作る。
[reproduction.json](../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis/reproduction.json)にGit commit、snapshot hash、入力・mask・予測・resolved configのhash、phaseとseedを記録した。
図のprovenanceは結果exporterのhashも保持し、学習時の封印済みsourceと区別する。大容量artifactは既存Git ignoreに従いrunに置いた。

反対位相は次で新しいrunを準備し、表示された新runに`launch`する。今回は実行していない。

```bash
python scripts/forge_trace_experiment.py prepare --study studies/study_068_forge_shot50 --observed-phase 1
python scripts/forge_trace_experiment.py launch --experiment runs/study_068_forge_shot50/<new-phase1-run-id>
```

## 解釈と制限

shot50用の再最適化ではなく、study067 FORGE random70の選択設定を移した比較である。
同じ元領域を使っており、過去に一切使用していない外部testへの汎化実験ではない。新しいrunの学習波形は当該runのobservedだけに制限した。
random70とは欠損率・構造・validationの扱いが異なるため、過去との数値差から欠損構造の因果効果や改善量を断定しない。
trace_oracleは欠損traceの真のRMSを正規化・振幅復元に使用する条件付き評価であり、観測だけから未取得traceの振幅まで推定する評価ではない。
clean_targetは既存QCによる対象で、ノイズのない真値ではない。1 mask・1 model seedの記述値として扱い、trace数を実験反復数と解釈しない。
phase1は追加設定・コマンドだけを用意し、今回の結果による設定変更や追加実行はしていない。未実行の標準手法はない。
