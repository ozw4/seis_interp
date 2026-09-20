# FORGE M1 MVP

M1の人工欠測から元トレース位置での評価までを、POCS、DRR、CCNet-5D、
NeRSI、ReGSI-real、ReGSI-gridで比較する。入力固定と限定preflightは完了。
本番の6 runと実データの復元精度評価は未実施であり、MVP受入は未判定。

仕様v0.1の研究条件は[config.yaml](config.yaml)、study_051の入力固定は
[inputs.yaml](inputs.yaml)、採用する準備成果物とpreflightのhashは
[prepared_inputs.lock.json](prepared_inputs.lock.json)を正本とする。
手法別の固定条件は各`<method_id>.yaml`。選定理由とNeRSIの入出力拡張は
[decisions.md](decisions.md)、準備結果は
[報告書](../../reports/forge_2017_m1_mvp_preparation.md)を参照。

主評価はclean-targetの`masked_trace_relative_mse`。
全eligible testでも同じ指標を保存する。自然欠測は採点しない。
1 mask・1 seedの予備結果だけを扱い、有意差・安定性・一般化の最終主張はしない。
SIREN-5D、他領域、追加mask/seed、HPO、bootstrap、空間内挿、denoising評価は含めない。

実行入口はリポジトリrootから使用する。

```bash
# 新しいimmutable準備成果物を作成（固定済み成果物を使う場合は不要）
.venv/bin/python scripts/forge_m1_mvp.py prepare

# 戻り値の準備ディレクトリを指定する。品質指標は計算しない。
.venv/bin/python scripts/forge_m1_mvp.py preflight --preparation <preparation_directory>

# 6手法を順に別processで実行し、全予測の検査通過後に評価する。
.venv/bin/python scripts/forge_m1_mvp.py run \
  --preparation <preparation_directory> \
  --preflight <passing_preflight_directory>
```

準備時にtest波形を保存しない。observedだけの配列と、振幅を含まないgeometryを
runtimeへ渡す。外側testを読むのは6手法すべてのmanifest・予測の検査通過後の
評価処理だけである。失敗・未完了・全予測の定数崩壊があれば`incomplete`として
状態とruntimeだけを保存し、部分的なtest指標も計算しない。
hash・shape・cell ID・有限値・ReGSI初期状態の不整合もtestを開く前に停止する。
grid側のfeaturesは格子定義と整数添字から再計算し、元UTMの混入を検査する。
maskはSHA-256の順位で固定し、全手法が同じhashを読む。

準備成果物と本番成果物は別のtimestamp・Git SHA付きrunへ保存する。
準備ファイルはread-onlyにし、本番起動時に入力・設定・実装hashとpreflightを再検証する。
予測は既存のNumPy形式`prediction.npy`と`prediction_index.parquet`を使用する。
推奨例のZarrに相当する予測保存先であり、新しい依存パッケージは追加しない。

本番runは`runs/<method_id>/`にmanifest、最終checkpoint、学習log、予測、
clean-target/all-eligible指標、trace別指標を持つ。
`aggregate/`に主表、runtime、ReGSI paired差、投影距離四分位別誤差、最終判定を保存する。
全testの有限予測・一意な逆対応・独立指標再計算が通るまで受入にしない。
予測が全て定数へ崩壊した場合も調査対象として記録する。

品質を見た設定変更や自動再試行はしない。障害再試行は同一設定・seedで別runへ保存する。
OOMに対するbatch/accumulation/checkpointing変更はtest指標参照前に新しい準備として固定し、
旧runをinvalidとして保持する。共通処理・評価の修正時は6手法を同じ実装で再実行し、
修正前後の結果を混在させない。
