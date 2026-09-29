ReGSI-real の条件探索と本実験条件の固定。

条件は [config.yaml](config.yaml)、入力は [inputs.yaml](inputs.yaml) に固定する。
各実行の実際の設定は run の config.yaml と manifest.json に保存する。
本実験には config.yaml の `main_experiment` を採用する。
採用判断は [decisions.md](decisions.md)、対応する既存runは
inputs.yaml の `main_experiment.reference_run` に記録する。
採用条件の共通clean評価は相対MSE 0.8788、相関中央値 0.2943だった。

14条件の学習と評価を完了した。欠測10%・20%・50%・60%は8 ms・1 msの両方で
探索の成功基準を満たした。2 msでは欠測60%・70%が基準を満たした。
欠測70%は8 msでも基準を満たしたが、1 msでは未達だった。
欠測80%は8 ms・2 msで基準未達だった。
共通cleanトレースの最小相対MSEは低レートで0.3999、元レートで0.4773だった。
図、全指標、振幅復元の追加診断、再実行コマンドは
[結果報告](../../reports/forge_054_regsi_conditions.md)にまとめた。
欠測60%・70%・80%の実行済み8条件について、
[ギャザー比較](../../reports/forge_054_gathers.md)を用意した。

各トレースを自身のRMSで正規化し、テストの正解RMSも診断目的で利用する。
テスト波形の形状はモデル入力・学習教師へ渡さない。
テスト情報を使う探索のため、厳密な未知データに対する性能評価とは扱わない。

時間ダウンサンプリングには、反射padding付きの窓付きsinc低域フィルタを使う。
欠測率比較では元のhash順位を保ち、共通の512本を未観測に保つ。
低レートで改善した条件は、元の1 msサンプリングで確認する。

実行入口は `scripts/forge_regsi_conditions.py`。`prepare` は共通データを準備し、
`run` は一つの条件を独立した初期重みから学習する。
本実験条件の実行例：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python scripts/forge_regsi_conditions.py run \
  --cache runs/study_054_forge_regsi_conditions/20260921T054308507963Z_51acdb640a29_data \
  --factor 2 --missing-fraction 0.8 --normalization trace_oracle \
  --steps 5000 --batch-size 32 --inner-fraction 0.1 \
  --learning-rate 0.001 --time-shift 0 --device cuda:1
```
