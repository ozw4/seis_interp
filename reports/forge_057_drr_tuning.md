指定runのDRRを、POCSと同じ相対MSE 0.85以下を目標に調整した。
元の欠損マスク、2 msサンプリング、全時間区間、トレースRMS正規化、評価対象を維持した。

NumPy SVDによる最終評価で主指標は **0.8373** となり、合格基準0.85以下を満たした。

| 評価領域 | 元のDRR | 採用DRR |
|---|---:|---:|
| clean_target（34,150本、主指標） | 0.9610 | 0.8373 |
| 全欠損（34,406本） | 0.9612 | 0.8382 |

閾値判定に使用した主指標は `0.8373312766247298`。
観測値の最大変更量は0、未被覆サンプル数は0。
最終確認の [指標](../runs/study_057_forge_2ms_random70_trace_rms/20260922T091622758001Z_51acdb640a29_drr_tuning/drr_grid_tuned/metrics.json)、
[予測](../runs/study_057_forge_2ms_random70_trace_rms/20260922T091622758001Z_51acdb640a29_drr_tuning/drr_grid_tuned/prediction.npy)、
[実行記録](../runs/study_057_forge_2ms_random70_trace_rms/20260922T091622758001Z_51acdb640a29_drr_tuning/manifest.json) を保存した。

採用条件は [drr_grid_tuned.yaml](../studies/study_057_forge_2ms_random70_trace_rms/drr_grid_tuned.yaml)、
再計算条件は [drr_tuned_verification.yaml](../studies/study_057_forge_2ms_random70_trace_rms/drr_tuned_verification.yaml)。

| パラメータ | 元の設定 | 採用設定 |
|---|---|---|
| ランク | 4 | 2 |
| 減衰のべき指数 | 3 | 3 |
| 反復数 | 10 | 6 |
| 空間窓 | [5, 8, 4, 8] | [1, 8, 1, 8] |
| 空間窓の重なり | [0, 0, 0, 0] | [0, 6, 0, 6] |
| 使用周波数 | DC〜Nyquist | DC〜Nyquist |

空間軸順はsource line・source point・receiver line・receiver point。
各窓は全2,001サンプルを使用し、時間窓への分割や追加の帯域制限を行わない。
level-four Hankel行列、最初の非採用特異値に基づく減衰、観測値の再挿入、
正のHann窓による合成という既存DRRの計算条件を維持した。

[探索run](../runs/study_057_forge_2ms_random70_trace_rms/20260922T090940122764Z_51acdb640a29_drr_tuning/manifest.json) で
5候補を評価し、合格条件へ到達した時点で残りの候補を実行せず終了した。
探索のCUDA計算は倍精度のGram行列固有値分解で同じ減衰射影を計算し、
採用条件の最終確認にはNumPy SVDを使用した。
CPUの並列実行でも窓の合成順序を維持する。

主指標の評価対象はclean_target 34,150本、保存する予測は全欠損34,406本。
これは元のtest集合を使ったパラメータ調整であり、独立test性能の推定ではない。
元実験のoracle trace RMSを振幅復元に使用する。
欠損波形は評価時だけで読み、DRRへの入力には渡さない。

再実行コマンド:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/forge_drr_tuning.py \
  --run runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run \
  --config studies/study_057_forge_2ms_random70_trace_rms/drr_tuned_verification.yaml
```

検証コマンドと結果は [verification.json](../runs/study_057_forge_2ms_random70_trace_rms/20260922T091622758001Z_51acdb640a29_drr_tuning/verification.json) に保存した。
