# C3: 25構成探索と追加3領域分割

Status: stopped by user request. 実行状況の正本はrun内の `workers/*.json` とする。

現在の実行: [supervisor](../../runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/workers/supervisor.json)。
実行前確認: [commands and results](../../runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/prelaunch_checks.json)。

ReGSI・CCNet5D・断面単位NeRSI・POCS・DRRの各手法で、明示した2軸5水準の
全25構成をS1で実行する。全5手法のS1完了後、V最小の1設定を固定して
領域分割seed 7311/7312/7313で追加3 runを実行し、集計して終了する。
全体140 run。S2/S3/E/R、全観測を使う別の最終再学習は行わない。

## 入力と領域

- C3 cropは `[384,16,32,8,32]`。source line `[25,41)`、shot `[18,50)`、
  relative receiver x `[0,8)`、y `[18,50)`、時間sample `[0,384)`。
- 時間間隔8 ms、時間shear・再サンプリング・trace RMS・IDWなし。
- 元の観測26,214本はseed 620701の厳密本数SHA-256マスクで固定する。
  v3の26,322本のマスクは使用しない。元データとtrace-to-cell対応は再利用する。
- source line 1本×連続4 shot×receiver全体を1ブロックとし、128ブロックを作る。
  compact JSON `["c3_region_split", seed, block_id]` のSHA-256とblock IDで順位を作り、
  最初の102個がtrain、次の13個がvalidation、残り13個がtest。
- S1領域seedは620702。学習seedは全runで7301固定。
- 入力は元観測集合とtrain領域の共通部分だけ。validation/test領域の観測も入力から外す。
  validation/testはそれぞれブロック内の全13,312本を採点する。
- 全領域の座標は既知。正解値は採点境界だけで読み、モデルへ渡さない。
- 正規化は各分割のtrain観測からfitした1個のGlobal RMSを全手法で共有する。

同一survey内の空間holdoutへの復元を評価する。過去のC3設定はT参照済みのため、
過去の調整の影響まで除いた新しいsurveyでの独立評価とは扱わない。
3 seedは同じcrop内部の割当変動であり、独立surveyの反復ではない。

## 学習・選択

50,000更新、学習率1e-4一定、weight decay 0、EMA 0.999の最終重みを固定して使用。
学習型3手法ともAdamW。1更新あたりloss対象64本、全384サンプル。
CCNetは複数patchを積み上げる。ReGSI内部mask本数は
`64 * floor(0.10 * N_train_observed / 64)`。名目率は0.10固定で、丸めた実現率を記録する。

Vは物理振幅でのトレース平均relative MSE。ゼロ波形の分母は既存実装の384。
丸め前のVを比較し、完全同点は構成ID順。S1でtestは採点しない。
追加3 runはS1重みを流用せず新規学習し、POCS/DRRも再構成する。
予測とcheckpointを保存した後にvalidation/testを各1回採点し、最良seedは選ばない。

NeRSIは3座標 `(sl/15, shot/31, rx/7)`、各座標40成分、共通β、
`omega_i = pi * beta**i, i=1..40`。周波数クリップなし。
核ノルムは正規化済みの予測断面全体の特異値和を `sqrt(384*32)` で割り、
その断面平均にλを掛けてdata lossへ足す。原著係数とは尺度が異なる。

DRRは全時間384 sample、0 Hz〜Nyquist。窓長刻みの厳密非重複配置とし、
端数を未観測としてpaddingして復元後にcropする。NumPy complex128のSVDを用い、
周波数batchを限定してメモリを抑える。CPU worker数8を全構成で固定する。

数値探索水準は `grid.yaml`、固定値は各手法のYAMLが正本。
コード・設定・入力hashを準備時にrunへコピーし、workerはコピーしたコードを実行する。
失敗・中断は記録し、未完了runを上書きせず、設定変更や自動追加探索はしない。
nonfinite学習は候補の失敗として予算内に記録し、有限な完了候補から選択する。

## 実行

```bash
.venv/bin/python scripts/run_c3_region_experiment.py prepare --study studies/study_062_c3_region_grid
PYTHONPATH=<experiment>/source/src .venv/bin/python <experiment>/source/scripts/run_c3_region_experiment.py run --repo /workspace --experiment <experiment>
.venv/bin/python scripts/run_c3_region_experiment.py status --experiment <experiment>
```

POCS/DRRはCPU、CCNet/NeRSIはGPU 0、ReGSIはGPU 1。
各手法workerは25構成を実行後、全手法のS1完了を待って追加3 runへ進む。
集計は各seedのV/test、平均・標本標準偏差（ddof=1）。副指標はglobal NMSE、
global SNR、平均trace SNR、RMSE。非有限trace SNRをclipせず件数を報告する。
