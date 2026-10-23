# study_065_forge_expanded_grid

Status: `stopped` — ユーザーの指示で実行中・待機中のworkerを停止。NeRSI・POCS・DRRの完了結果を保持する。

Forgeの解析領域を拡張し、C3の検証で選んだ最良設定を5手法で1設定ずつ評価する。
NeRSIは断面単位版を使用する。実験条件は[config.yaml](config.yaml)、参照元とhashは
[inputs.yaml](inputs.yaml)、範囲選定の理由は[decisions.md](decisions.md)を正本とする。

格子の規則性を優先し、発振・受信の地理的な端の違いを許容する。
元の格子原点・方向・間隔を保って拡張し、元領域の全地点・実座標・投影座標と、
元の49,151有効トレースを保持する。
新範囲の有効トレースは248,125本で、重複割り当ては0だった。
[範囲図](../../runs/study_051_forge_region_search/20261020T013228601242Z_05c92e79ce80_grid_compatible/grid_compatible_survey.png)と
[準備記録](../../runs/study_051_forge_region_search/20261020T013228601242Z_05c92e79ce80_grid_compatible/metadata.json)を保存した。

既存の `load_candidate_inputs`、`validate_mapping`、実座標／格子座標の `make_geometry` と
断面単位NeRSIの `nersi_profile_layout` を実データの配置で通過した。
すべての発振セルに実地点があり、自然欠損は観測・教師・採点対象に加えない。
格子を満たすための波形の平均・内挿や、重複点の削除は行っていない。

訓練・検証・testは全空間領域内のトレース単位で割り当てる。
学習型手法は新しい訓練入力から学習し、検証・test波形を入力へ混ぜない。
前処理は既存Forgeの方式を継承し、評価対象RMSの使用は正規化・振幅復元の境界に限る。
4 msへの変更は、元の1 ms波形に既存のanti-aliasフィルタを適用してから間引く。
拡大した入力に合わせた時間長・断面幅以外は、C3のモデル幅・層構成・損失・学習設定を維持する。
CCNet5Dのpatchが発振ライン数を超える分は、外側の未観測セルとして補い、実地点のIDを保つ。
ReGSI・CCNet5Dは学習時の中間テンソルをCPUへ退避し、モデルや勾配の定義を変えずにGPUメモリを抑える。
ReGSIでは既存の依存範囲の計算省略・辺メッセージの分割集約・activation checkpointingも有効にする。
パラメータ数は入力に依存する出力層も含めて記録し、モデル幅を合わせ直す処理は行わない。

実行入口は[scripts/forge_trace_experiment.py](../../scripts/forge_trace_experiment.py)。
runには入力割当・設定・実装のスナップショットを保存し、各手法の予測を確定してから
検証・testをそれぞれ1回評価する。再探索・追加seed評価・別の最終再学習は行わない。

[実行run](../../runs/study_065_forge_expanded_grid/20261020T150430809810Z_05c92e79ce80)の
`workers/<method>.json`と`workers/<method>.log`に状態と進捗を保存する。
完了した手法の指標はrun直下の`summary.csv`へ集計する。
