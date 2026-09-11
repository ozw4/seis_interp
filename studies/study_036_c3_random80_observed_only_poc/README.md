# Fixed random-80 observed-only C3 comparison

POCS、DRR、NeRSI、CCNet-5D、relational trace graphを同じ入力artifactで比較する。
Benchmark IDは `c3_sl25_40_random80_observed_only_v1`。
Status: `stage_1_baseline_frozen`。
全手法は固定selection・outer mask・target-only評価を共有する。
`methods/` と `smoke/` は `config.yaml` を継承し、本実験用 `formal/` は同じ入力契約を
各ファイルに明記した独立したnative configである。

## 入力と比較条件

`inputs.yaml` の相対pathはこのStudyディレクトリを基準とする。
QC済みcase `c3_benchmark_test_random_trace_80_seed42` を使用し、入力artifactは変更しない。
解析領域は `[384,16,32,8,32]`、Oは26,362本、Tは104,710本。
artifactのpartition名は `test` だが、neural学習は同じvolumeのOだけを使用する。
NeRSIはOで直接学習し、CCNetとGNNはO内のinner maskで学習する。
Tの真値は既存の共通評価器だけが読む。

POCSとDRRは物理振幅のまま再構成する。Neural手法はO全体のGlobal RMSを共有の定義で
計算し、`masked_trace_relative_mse` で学習する。予測は物理振幅へ戻し、Oを厳密に再挿入する。
全Tのcoverage、完全なinput lock一致、prediction/checkpointのSHA-256を検査する。
HPO、validation選択、複数seed、retry、最良設定の自動選択は行わない。

## Config

| 本実験config | 固定設定 | 本実験budget | Smoke budget |
|---|---|---:|---:|
| `formal/pocs.yaml` | Fourier POCS、window `[128,8,16,8,32]` | 100 iterations | 2 iterations |
| `formal/drr.yaml` | rank 4、damping 3、window `[5,8,4,8]` | 10 iterations | 1 iteration |
| `formal/nersi.yaml` | encoder 384、latent 96、16 profiles/update | 5000 updates | 2 updates |
| `formal/ccnet5d.yaml` | width 32、kernel 3、linear output | 5000 updates | 2 updates |
| `formal/gnn.yaml` | width 64、2 rounds、2 neighbors/relation | 5000 updates | 2 updates |

`smoke/*.yaml` は対応する `methods/*.yaml` を継承し、budgetとreport intervalだけを短縮する。
解析領域、モデル、batch/patch、乱数seed、全T推論は本実験configと同一。
`formal/*.yaml` は `extends` を使わず、smokeや共有設定の編集が暗黙に波及しない。
Smokeの指標は動作確認用であり、手法の性能比較・選択には使用しない。
本実験はsingle-seed・固定budgetの比較であり、収束や同一計算量を保証しない。
同じ5000 updatesでも教師trace提示数・parameter数・実行時間は異なるため、summaryの
computeとtimingを併記する。CCNet/GNNのinner mask fractionはOに対して0.8に固定する。
CCNetの学習patchは `[384,4,8,4,8]` で、全384時間サンプルを必ず含む。
推論core `[64,8,16,8,16]` の時間分割は、学習patchの全trace条件とは独立である。

Outer seedは42。Neural model initializationは101、NeRSI samplingとGNN episodeは201、
CCNet patch placementは301、inner maskは401。各streamはnative configで独立に指定する。
Neural deviceは `cuda:1`、CPU thread数は以下の実行環境で1に固定する。
GPUを使う手法は本実験・smokeとも同時実行しない。

## 実行

リポジトリrootから実行する。`RUN_ROOT` はUTC時刻・Git SHAを含む未使用pathに設定する。

```bash
study=studies/study_036_c3_random80_observed_only_poc
qc=data/processed/c3_na/study_029_c3_amplitude_qc
case_id=c3_benchmark_test_random_trace_80_seed42
poc_inputs=(
  --interim data/interim/c3_na/all_ffids
  --processed "$qc/partition"
  --mask "$qc/masks/$case_id"
  --case "$qc/cases/$case_id"
  --volume "$qc/volumes/$case_id"
)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m seis_interp.cli poc check "${poc_inputs[@]}" --json
```

各configの構造・継承・smoke差分を入力配列なしで検証する。

```bash
pytest -q tests/unit/test_c3_random80_poc_study_configs.py
```

単独smokeの例。POCS、DRR、CCNetも同名のCLI/configを使う。
GNNはCLI名 `relational-trace-graph`、config名 `gnn.yaml` を使う。

```bash
python -m seis_interp.cli interpolate nersi "${poc_inputs[@]}" \
  --config "$study/smoke/nersi.yaml" --output "$RUN_ROOT/nersi"
```

5件を独立processで順次実行し、成果物を相互検査する場合：

```bash
python -m seis_interp.cli poc run-all "${poc_inputs[@]}" \
  --pocs-config "$study/smoke/pocs.yaml" --drr-config "$study/smoke/drr.yaml" \
  --nersi-config "$study/smoke/nersi.yaml" --ccnet5d-config "$study/smoke/ccnet5d.yaml" \
  --gnn-config "$study/smoke/gnn.yaml" --output "$RUN_ROOT"
```

本実験はsmokeと異なる未使用の `RUN_ROOT` を指定して実行する。

```bash
python -m seis_interp.cli poc run-all "${poc_inputs[@]}" \
  --pocs-config "$study/formal/pocs.yaml" --drr-config "$study/formal/drr.yaml" \
  --nersi-config "$study/formal/nersi.yaml" --ccnet5d-config "$study/formal/ccnet5d.yaml" \
  --gnn-config "$study/formal/gnn.yaml" --output "$RUN_ROOT"
```

既存outputは再利用しない。実行条件・結果の正本はrun内のresolved config、input lock、
metadata、metricsとcomparison summaryである。

## 確認済み結果

全5手法の実artifact smokeを検証済み。各runは現行smoke configと一致し、
T全104,710本のcoverage、Oの厳密な保持、input lock完全一致、artifact hash検査に成功した。
[検証summary JSON](../../runs/study_036_c3_random80_observed_only_poc/20260911T075650Z_466b41f_smoke_verified/summary.json)
と[目視用CSV](../../runs/study_036_c3_random80_observed_only_poc/20260911T075650Z_466b41f_smoke_verified/summary.csv)
が、明示した5件のrun directoryを参照する。

## Stage-1 baseline

固定参照runは `20260911T081726Z_466b41fae15c_formal`。
[凍結lock](stage_1_baseline.lock.json) がrunへの相対path、完全なinput lock、
全44 runファイルと5本の本実験configのSHA-256、各手法のGit来歴を固定する。
[正式summary](../../runs/study_036_c3_random80_observed_only_poc/20260911T081726Z_466b41fae15c_formal/summary.json)
を全5手法のbaselineとして参照し、最良手法の選択は行わない。

| Method | Target SNR (dB) | RMSE |
|---|---:|---:|
| POCS | 10.6881 | 2.6984 |
| DRR | 8.5207 | 3.4632 |
| NeRSI | 10.2829 | 2.8272 |
| CCNet-5D | 6.5988 | 4.3208 |
| Relational trace graph | 9.2729 | 3.1758 |

全5手法が指定budgetを完了し、T全104,710本のcoverage、Oの厳密な保持、
input lock完全一致、summaryと個別artifactの一致、prediction/checkpoint hashを再検証済み。
Checkpointの再推論や再学習は行っていない。

POCSの記録はcommit `466b41fae15c`・dirty=true、後続4手法は `d75ff667c4e6`・dirty=false。
両commitの `src/` は同一だが、POCS開始時の未コミット状態はsource snapshotがないため
遡及的には証明できない。この制約を含めてbaselineを保持し、元のmetadataを補正しない。

凍結は固定pathとchecksumによる保存契約であり、runの内容・権限は変更しない。
このrunと `formal/*.yaml` は上書きせず、次の実験は別config・別run directoryに保存する。
run本体はGit管理外であり、lockはデータのバックアップを代替しない。
