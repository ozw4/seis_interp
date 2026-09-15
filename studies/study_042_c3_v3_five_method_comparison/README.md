# v3 five-method comparison

POCS、DRR、NeRSI、CCNet-5D、Proposed GNNを、固定v3入力と物理振幅のtrace SNR平均で比較する。
全target traceの `10 log10(target energy / error energy)` をdBのまま算術平均する。
global SNR、RMSE、relative L2、mean trace relative MSEは副指標。
ゼロenergyや完全一致をepsilonで置換せず、非有限値は件数とstatusを報告する。

入力の正本は [v3 conditions lock](../study_041_c3_nersi_translated_window/v3_conditions.lock.json)。
source line 25–40、shot 18–49、receiver x 0–7、receiver y 18–49、time 0–383。
shape `[384,16,32,8,32]`、O=26,322、T=104,750。
nominal 80% random mask、crop内のrealized missing fractionは79.9179%。

Neural 3手法はO-only Global RMSとMSEを使用する。入力のtrace別RMS正規化は行わない。
POCS/DRRはStudy 036設定、CCNet/GNNはStudy 037設定を使う。
NeRSIのv3正本は時間shearなしの `nersi_no_time_shear.yaml` と
[採用結果lock](nersi_v3_result.lock.json) に記録したrun。
trace SNR平均は15.5856 dB。実行scriptもこの設定を使用する。
NeRSIは50,000 updates・EMAあり・時間shearなし、CCNet/GNNは5,000 updates。
これは等計算量・等探索量比較ではない。配置とNeRSI設定はT評価で選択されており、独立評価ではない。
全5runを保存し、再試行・HPO・自動best選択は行わない。

現在のGNN正本はStudy 044で採用したEMA結果（trace SNR平均18.2795 dB）。
[GNN採用結果lock](../study_044_c3_v3_gnn_target_tuning/gnn_v3_result.lock.json)
がrunと評価重み`artifacts/ema.pt`を固定する。上記5手法の初期比較と
`gnn.yaml`は5,000更新の比較基準として保持し、採用runの再実行には
[EMA設定](../study_044_c3_v3_gnn_target_tuning/mask10_fourier16_width128_ema999_20k.yaml)
とStudy 044の実行scriptを使用する。

## 実行

リポジトリrootから実行する。GPU 1を使用し、各手法は別processで逐次実行する。
開始時にv3の完全なinput lockとSHAを検証する。従来の `poc check` はv1専用なので使わない。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
unset CUDA_VISIBLE_DEVICES
run_root="runs/study_042_c3_v3_five_method_comparison/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)_formal"
.venv/bin/python scripts/run_c3_v3_comparison.py --output "$run_root"
```

`summary.json` を機械可読正本、`summary.csv` を表示用とする。
成功runは全input lock一致、prediction/checkpoint SHA、full-target coverageを検証する。
一手法が失敗しても残りを実行し、全体を非zero終了する。設定snapshot、source archive、各processのlogを保持する。

## NeRSI time-shear ablation

`nersi_no_time_shear.yaml` は `nersi.yaml` から `time_alignment` だけを除いた設定。
50,000 updates、MSE、EMA、seed、O-only Global RMS、全targetの物理振幅評価を維持する。
shearありの比較元はtrace SNR平均15.6282 dB、global SNR 14.4326 dB。
別runで学習し、shearありの結果を上書きしない。

| Time shear | Mean trace SNR [dB] | Global SNR [dB] | RMSE |
|---|---:|---:|---:|
| 3.0625 samples / receiver-y cell | 15.6282 | 14.4326 | 1.7591 |
| None | 15.5856 | 14.2667 | 1.7931 |

v3正本はshearなしの結果とする。shearありのrunと設定は比較用に保持する。
shearなしでもtrace SNR平均は15 dBを超え、ありとの差は−0.0425 dB。
両runとも全104,750 target tracesを評価し、trace SNRの非有限値は0本。
この単一seed・T参照済みの比較から統計的な同等性は主張しない。

shearなし結果:
`runs/study_042_c3_v3_five_method_comparison/20260914T010033Z_03c2fc83df18_nersi_no_time_shear/nersi/metrics.json`。
比較元:
`runs/study_042_c3_v3_five_method_comparison/20260914T000133Z_03c2fc83df18_formal/nersi/metrics.json`。
