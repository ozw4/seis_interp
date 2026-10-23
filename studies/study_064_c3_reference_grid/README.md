# C3: MSE・lr=1e-3にそろえた3学習手法の25構成探索

Status: running / queued. 実行状況の正本はrun内の `workers/*.json` とする。

現在の実行: [supervisor](../../runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/workers/supervisor.json)。
実行前確認: [コマンドと結果](../../runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/prelaunch_checks.json)。


ReGSI・CCNet5D・NeRSIの各25構成を新規学習し、検証V最小の設定を選ぶ。
探索条件は [config.yaml](config.yaml)、2軸5水準は [grid.yaml](grid.yaml)、
固定値は [regsi_real.yaml](regsi_real.yaml)、[ccnet5d_grid.yaml](ccnet5d_grid.yaml)、
[nersi_profile.yaml](nersi_profile.yaml) を正本とする。
3手法の訓練条件をMSE、AdamW、lr=1e-3、weight decay=0、50,000更新、
1更新64本、EMA 0.999、学習seed 7301に統一する。
構成選択の検証Vは平均トレースrelative MSEとし、最終EMAで評価する。

ReGSIの過去の高性能モデル設定、NeRSIのβ=1.15・λ=0、CCNet5Dの基準設定が
候補内に存在することを、準備時に `reference_settings` の全項目で検査する。
入力・検証・testのトレース割当とGlobal RMSはstudy_063と同一とする。
空間領域は分けず、入力23,593本・検証2,621本・test104,858本を使う。

POCS・DRRはstudy_063のworkerを継続する。
新しい3手法と継続中2手法のS1完了後、各手法で追加3 seedを実行する。
新規runは3×28=84、継続2手法を合わせた採用対象は5×28=140 run。
停止した3手法の探索結果は保存し、新しい25構成の選択には混ぜない。
追加3 seedではトレース割当を変更し、学習型手法は毎回新規学習する。

分割の共通性と継続workerはhashで固定する。実行中のstudy_063のコード・設定・
完了済みrunは保存し、監督処理が新しいS1の完了を既存の待機条件へ通知する。
実行GPUと空きメモリの条件はconfigで固定する。GPUごとのrunロックを
1構成終了時に解放し、全5手法のS1待機中には保持しない。
過去の初期化seed、観測マスク、NeRSIの16 profiles/updateまで再現する実験ではない。

```bash
.venv/bin/python scripts/run_c3_region_experiment.py prepare --study studies/study_064_c3_reference_grid
PYTHONPATH=<experiment>/source/src .venv/bin/python <experiment>/source/scripts/run_c3_region_experiment.py run --repo /workspace --experiment <experiment>
```
