# study_066_forge_c3_reference

Status: `ccnet_stopped_by_user` — CCNet5Dはユーザー指定により27,890更新時点で学習を停止。
NeRSI・POCS・DRRの完了結果と停止runのログを保持する。

実行条件は[config.yaml](config.yaml)、入力とC3選択結果の参照元は[inputs.yaml](inputs.yaml)、
比較条件の理由は[decisions.md](decisions.md)を正本とする。

ReGSI・CCNet5D・断面単位NeRSI・POCS・DRRを各1設定で実行する。
study065からモデル幅・層構成・探索で選んだ値・損失・optimizer・EMAを引き継ぎ、
4 msと学習型50,000更新を維持する。
CCNet5Dの空間patchは今回のユーザー指定を優先する。
時間長と断面幅は元範囲に合わせ、断面単位NeRSIの出力層サイズを記録する。

観測候補と検証の割当にはstudy060の識別子・seedを使い、元の割当との一致を確認する。
検証波形を学習へ戻さず、モデル・再構成を確定した後に検証とtestを採点する。
前処理は既存Forgeのanti-alias間引き、幾何による時間整列、trace_oracle境界を継承する。
CCNet5Dは指定された空間patchを実領域に配置し、教師には実際の観測セルだけを使う。

実行入口は[scripts/forge_trace_experiment.py](../../scripts/forge_trace_experiment.py)。
入力・設定・実装を各runへ保存し、進捗は`workers/<method>.json`と実行ログ、
完了した手法の集計はrun直下の`summary.csv`へ保存する。

[実行run](../../runs/study_066_forge_c3_reference/20261020T225334465289Z_05c92e79ce80)に
study060の割当との一致検査と起動前チェックを保存した。

NeRSI・POCS・DRRは完了。ReGSIは学習開始前にGPU 0へ移し、
[ReGSI実行run](../../runs/study_066_forge_c3_reference/20261021T003958326603Z_05c92e79ce80_regsi_gpu0)で
停止した。CCNet5Dも元runのGPU 1で停止した。
移動元の`execution_relocations.json`で両runを関連付ける。

再開時のCCNet5Dは、BF16で1 patchを1回forwardし、同じ内部マスクから
重複しない64本ずつの勾配を、重みを変更する前に全組分保存する。
保存した勾配を順番にAdamWへ渡し、各更新後にEMAを更新する。
端数は使わず次のpatchへ移り、50,000回の実際のAdamW更新で終了する。
重み・AdamW・EMAはFP32を維持する。訓練・予測とも時間方向を分割しない。
1枚で64本の教師を確保できない小さいpatchは、独立した複数patchをまとめてforwardし、
合計64本を1更新の損失に使う。各patchは1回ずつ処理し、内部マスク率は変更しない。

[CCNet再起動run](../../runs/study_066_forge_c3_reference/20261021T031757030635Z_05c92e79ce80_ccnet_bf16_cached_gradients)は
patch変更の指示により停止した。入力・領域・trace割当を引き継ぐ別runで新規学習する。

[patch変更後の実行run](../../runs/study_066_forge_c3_reference/20261021T044343244080Z_05c92e79ce80_ccnet_patch_6_8_8_8)を
patch変更の指示で停止した。再起動理由と入力内容の一致は各runの`restart_origin.json`に保存する。

[推論で停止したrun](../../runs/study_066_forge_c3_reference/20261021T053559200620Z_05c92e79ce80_ccnet_patch_4_8_4_8)は
50,000更新後にGPU 1の推論メモリ不足で停止した。各patchの教師が64本未満のため、複数patchをまとめて64本の損失を作る。

[ReGSI再実行run](../../runs/study_066_forge_c3_reference/20261021T064203734700Z_05c92e79ce80_regsi_restart)は
GPU 0に必要な空きができたら自動で学習を開始する。
停止runには途中checkpointがないため、同じ設定・入力・seedで最初から50,000更新を実行する。
ReGSIの実装は停止runの保存済みsourceを引き継ぎ、CCNetは並行して継続する。

[推論修正後の再学習run](../../runs/study_066_forge_c3_reference/20261021T231630621271Z_05c92e79ce80_ccnet_bf16_spatial_inference_restart)はユーザー指定で停止した。
学習・推論ともBF16、空間patch／出力coreを[4,8,4,8]とし、時間全長を保つ。
空間haloは4セルとし、時間dilationのhaloを空間へ広げない。
学習終了直後に最終EMAの重みをfinal.ptへ保存してから推論する。
実データ診断の推論ピーク約14.5 GiBに余裕を加え、17 GiB以上の空きで開始する。
