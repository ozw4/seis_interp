指定runのPOCSを、同じ欠損マスク・2 msサンプリング・トレースRMS正規化・評価対象で調整した。
ユーザー指定の最終合格基準は相対MSE 0.85以下。

NumPyで最終確認した主指標は **0.8381** で、合格基準を満たした。

| 評価領域 | 元のPOCS | 採用POCS |
|---|---:|---:|
| clean_target（34,150本、主指標） | 1.8630 | 0.8381 |
| 全欠損（34,406本） | 1.8610 | 0.8380 |

閾値判定に使用した主指標の値は `0.8380560271314345`。
観測値の最大変更量は0、未被覆サンプル数は0。
最終確認の [指標](../runs/study_057_forge_2ms_random70_trace_rms/20260922T085042853926Z_51acdb640a29_pocs_tuning/pocs_grid_tuned/metrics.json)、
[予測](../runs/study_057_forge_2ms_random70_trace_rms/20260922T085042853926Z_51acdb640a29_pocs_tuning/pocs_grid_tuned/prediction.npy)、
[実行記録](../runs/study_057_forge_2ms_random70_trace_rms/20260922T085042853926Z_51acdb640a29_pocs_tuning/manifest.json) を保存した。

元の結果は [元runのPOCS指標](../runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run/runs/pocs_grid/metrics.json)、
採用設定は [pocs_grid_tuned.yaml](../studies/study_057_forge_2ms_random70_trace_rms/pocs_grid_tuned.yaml)。
採用設定の再計算は [pocs_tuned_verification.yaml](../studies/study_057_forge_2ms_random70_trace_rms/pocs_tuned_verification.yaml) で固定する。

| パラメータ | 元の設定 | 採用設定 |
|---|---|---|
| 反復数 | 100 | 4 |
| 閾値の開始比 | 1.0 | 1.0 |
| 閾値の終了比 | 0.01 | 0.4 |
| 窓サイズ | [128, 8, 16, 8, 32] | [512, 1, 8, 1, 8] |
| 重なり | [64, 4, 8, 4, 16] | [256, 0, 6, 0, 6] |

軸順は時間・source line・source point・receiver line・receiver point。
既存のFourierハード閾値、幾何級数の閾値スケジュール、観測値の再挿入、正のHann窓による合成を維持する。
走時補正、追加フィルタ、振幅補正は導入していない。

探索では82件の候補評価を記録した（元設定の再現確認・重複条件を含む）。
ユーザーの完了指示で残りの探索を中断した。
各探索runに設定・予測・トレース別指標・実装スナップショットを保存した。
共通震源の小窓の一条件は未被覆サンプルが生じたため失敗として記録し、採用対象から除外した。

探索時はCUDAの倍精度FFTを用いた。ハード閾値の境界付近ではFFTの丸め差が選択係数を変えるため、
元設定のCUDA再現値は1.8629、元のNumPy実行値は1.8630だった。
採用条件の最終確認にはNumPyのFFTを用い、既存の逐次実装との一致をテストした。

これは元のtest集合を用いたパラメータ調整結果であり、独立したtest性能の推定ではない。
全欠損34,406本の予測を保存し、従来どおりclean_target 34,150本を主指標とする。
元実験で採用済みのoracle trace RMSを振幅復元に使用する。
欠損波形は指標計算だけで読み、POCSの入力へ渡さない。

再実行コマンド:

```bash
.venv/bin/python scripts/forge_pocs_tuning.py \
  --run runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run \
  --config studies/study_057_forge_2ms_random70_trace_rms/pocs_tuned_verification.yaml
```

検証コマンドと結果は [verification.json](../runs/study_057_forge_2ms_random70_trace_rms/20260922T085042853926Z_51acdb640a29_pocs_tuning/verification.json) に記録した。
