# study_070 decisions

## 2026-11-09 — neural-implicit の代表手法を新規 implicit trace field にする

**Status:** 採用。study_067/068/069 の test 結果を見た**後**に決めた変更であり、事後的であることを開示する。

**Decision:** 比較における neural-implicit NN の代表を `inr_real`
（`src/seis_interp/models/trace_coordinate_field.py::TraceCoordinateField`）とする。既存の NeRSI 実装は
代表として使わない。

**Evidence:**

| 記録 | 測定値 |
|---|---|
| study_067 FORGE test、断面単位 NeRSI 3 seed 平均 | 0.3560 dB — 事前登録の床 0.4 dB 未達（全手法中唯一） |
| study_067 FORGE S1 採用 cell `a04_b02` | validation relMSE 0.9425649713411244 |
| study_068（periodic shot50） | clean relMSE 0.9999997939998186、mean trace SNR approximately 0 dB |
| study_069（random shot50） | clean relMSE 1.0000006756206283、mean trace SNR approximately 0 dB |
| study_069 の断面単位 NeRSI final training loss | 0.46086586 = ゼロ予測の loss |
| study_061 FORGE 最終3 seed | relMSE 1.0001 ± 0.0001、相関中央値 approximately 0 |
| study_060 FORGE、トレース単位 NeRSI | clean-test relMSE 0.8865 |

**理由:** 既存の2つの NeRSI 実装はどちらも時間を conv decoder の出力軸に置く。トレース単位版
`NersiTrace` は3段の 2× pixel shuffle で 8 列の canvas を作り列0だけを返すため、decoder 出力の **7/8** が
返すトレースへ届かない。`inr_real` は canvas を持たず、出力層が厳密に `time_sample_count` 本を出して
1サンプルも捨てない。この構造は `tests/unit/test_trace_coordinate_field.py` が固定する
（conv / transposed-conv / pixel shuffle / upsample の submodule が0個、`output_layer.out_features == T`、
全パラメータに非ゼロ勾配）。

`_fit_nersi` と `models/nersi_trace.py` は変更していない。study_052–060 の RNG 列・records・予測を保持する。

**共有した部品:** `models/nersi.py` の `FourierFeatureMapping` のみ。パラメータを持たない固定の
positional encoding であり、共有により `frequency_base` と `fourier_components` の意味が両手法で同一になる。
アーキテクチャは共有しない。

## 2026-11-09 — 座標は素の4連続座標とし、幾何由来の affine 変換は実装しない

**Status:** 採用。

**Decision:** 入力は `source_line_continuous` / `source_point_continuous` /
`receiver_line_continuous` / `receiver_point_continuous` を `max(spatial_shape − 1, 1)` で正規化した4値のみ。
study_060 のトレース単位 NeRSI が使った midpoint/radius/azimuth への affine 変換と `polar_offset` は
実装しない。

**Evidence:** study_060 の変換は `processing/nersi_trace_coordinates.midpoint_offset_transform` が
厳密な affine 性を要求する。C3 の実座標では成立しない。

```bash
PYTHONPATH=src .venv/bin/python -c "
import pandas as pd
from seis_interp.processing.nersi_trace_coordinates import midpoint_offset_transform
midpoint_offset_transform(pd.read_parquet(
  'runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces/input/geometry.parquet'),
  (16, 32, 8, 32), polar=True)"
# ValueError: actual midpoint/offset geometry is not affine in the local coordinates
```

4つの格子 index への最小二乗回帰で測った最大残差は `source_y_m` / `receiver_y_m` で **23.0588 m**、
`source_x_m` / `receiver_x_m` では 0 m。C3 の source line が y 方向に非 affine なずれを持つ。
同じ関数は FORGE では 4×5 行列を導出できる。

**理由:** FORGE だけに変換を使うと model の forward 経路が dataset 依存になる。これは canvas 適応ではなく
構造差であり、共通 grid の前提が崩れる。`midpoint_offset_transform` の厳密性 assertion を緩めることは
既存の error contract の変更にあたる。したがって両 dataset で素の4座標を使う。

**帰結:** study_070 の arm は study_060 の採用 arm と同一ではない。出力構造・入力 parameterization・
loss・optimizer・更新数・canvas のすべてが異なるため、study_070 と study_060 の関係は記述的のみとする。
FORGE では4座標が真に連続（crooked acquisition）、C3 では格子 index と一致する（C3 preparation が
`real_geometry_equals_grid_geometry` を宣言済み）。

## 2026-11-09 — hidden 幅だけを canvas に合わせ、parameter_count_policy を使わない

**Status:** 採用。

**Decision:** `fourier_components: 40`、`hidden_layers: 4`、`activation: gelu` を両 dataset で共通にし、
`hidden_width` だけを canvas に合わせる。`matched_training` は両 dataset で同一の5キー
（`max_steps` / `learning_rate` / `loss_traces_per_update` / `parameter_count_target` /
`parameter_count_tolerance`）とし、study_067 FORGE の `parameter_count_policy` は引き継がない。

**Evidence:** 出力層は `Linear(hidden_width → T)` で、T は整列 canvas 長（C3 384 / FORGE 2172）。
`matched_hidden_width` による実測（`model_parameter_count` と閉形式が一致）:

| hidden_layers | C3 width | C3 params | 比 | FORGE width | FORGE params | 比 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 345 | 362,979 | −0.0798 % | 137 | 362,619 | −0.1789 % |
| 3 | 284 | 362,484 | −0.2161 % | 131 | 363,339 | +0.0193 % |
| **4** | **249** | **362,679** | **−0.1624 %** | **126** | **364,296** | **+0.2827 %** |
| 5 | 225 | 362,409 | −0.2367 % | 121 | 362,873 | −0.1090 % |
| 6 | 208 | 364,384 | +0.3069 % | 117 | 362,883 | −0.1063 % |

**理由:** 深さ 2–6 のすべてが ±5% 帯の内側にある。4 を採るのは Fourier 特徴 MLP の標準構成であり、
C3 の幅 249 がその標準幅 256 にほぼ一致するため。validation では選んでいない。
`parameter_count_policy` を落とせるのは、trace field が canvas 例外を必要とせず単一の ±5% 規則を
両 dataset で満たすからである（断面単位 NeRSI は `canvas_time_upsampling` を要し、それでも FORGE で
341,485 = −6.0% と帯域外だった）。`validate_matched_training` は budget のキー集合を厳密に要求するため、
6キー目は追加できない。

F=16（study_060 の値）は採らない。base 1.05 では正規化軸上で `1.05**40 / 2 = 1.09` cycle しか回らず、
b01 列が near-constant encoding になる。これはゼロ予測を招く領域である。

## 2026-11-09 — axis_a は decoupled AdamW の実効強度で決める

**Status:** 採用。

**Decision:** axis_a は `training.weight_decay` ∈ {0, 0.001, 0.01, 0.03, 0.1}。optimizer は AdamW。

**Evidence:** AdamW の decay は decoupled（`w ← w(1 − lr·λ)`）なので、lr 1e-3 × 50,000 更新での
残存係数は `exp(−50λ)`。

| λ | 0 | 1e-5 | 5e-5 | 1e-4 | 3e-4 | 0.001 | 0.01 | 0.03 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 残存係数 | 1.0000 | 0.9995 | 0.9975 | 0.9950 | 0.9851 | 0.9512 | 0.6065 | 0.2231 | 0.0067 |

**理由:** study_058 が採用した `weight_decay` 5e-5 は **Adam の coupled L2** での値であり、decoupled AdamW
では 0 と数値的に区別できない。1e-5〜3e-4 の範囲をそのまま使うと axis_a が無効化し 5×5 が実質 1×5 に潰れる。
採用した5値は断面単位 NeRSI の事前登録 axis_a（`training.nuclear_norm_weight`）と同一の5値であり、
正則化軸の形も共通 grid と揃う。λ = 0.1 列がゼロ予測に落ちた場合も規則どおり採点し、除外も再実行もしない。

## 2026-11-09 — axis_b は断面単位 NeRSI と同一の5値をスカラで与える

**Status:** 採用。

**Decision:** axis_b は `model.frequency_base` ∈ {1.05, 1.075, 1.1, 1.15, 1.25} をスカラとして4軸すべてに
適用する。study_060 の軸別 list `[1.05, 1.05, 1.1, 1.1]` は使わない。

**Evidence:** `base**40 / 2` で測る正規化軸あたりの cycle 数と、格子 index の Nyquist `(n − 1) / 2`。

| base | 1.05 | 1.075 | 1.1 | 1.15 | 1.25 |
|---|---:|---:|---:|---:|---:|
| cycles | 3.52 | 9.02 | 22.63 | 133.9 | 3762 |

Nyquist: FORGE source line 2.5、C3 source line 7.5、source point 15.5、receiver line 3.5、
receiver point 15.5。

**理由:** 軸別 list は study_060 の (midpoint, radius, azimuth) parameterization を前提にしており、
素の4 index 軸では軸3–4 を 1.1 に固定する根拠がない。スカラにすると axis_b が断面単位 NeRSI の
事前登録 axis_b と同形になり、出力構造の対比が Fourier 帯域で交絡しない。
b01（場合により b02）が短い軸で解像不足になることは共通 grid を共有する代償として受け入れる。

## 2026-11-09 — study 内 YAML の sha256

launch 時点の正本。

| file | sha256 |
|---|---|
| [c3/config.yaml](c3/config.yaml) | `f3b028d7eb6747459a1ee620e67cb0937c90f2746e0f3db344195dd92607d1d7` |
| [c3/inputs.yaml](c3/inputs.yaml) | `02db0cdeca2570fa8736495e2c8102e82076caf71de5413832475854d4f50ede` |
| [c3/inr_real.yaml](c3/inr_real.yaml) | `365e3b1bbbe36fd7c13243065f05ebfe9e51a7e8ef574f4be96351673231628c` |
| [dataset_differences.yaml](dataset_differences.yaml) | `84407c2bd8b5a9cfca88f7242f40f5b8ef101763cc4c32f9799060ee39cd91d7` |
| [evaluation.yaml](evaluation.yaml) | `cc250b5f1167073b58fd381b4847567c8d1fa7e09eef36edaa69aa2673e00ac0` |
| [forge/config.yaml](forge/config.yaml) | `20708886418dd4b58b0298ab4dfe3b50e2b52f8aec7e0230bf2fbe28faf97d19` |
| [forge/inputs.yaml](forge/inputs.yaml) | `645f339c296102a6e9192a9ca2d515862ee425cd3c8e4a5b65187090ee375021` |
| [forge/inr_real.yaml](forge/inr_real.yaml) | `444a731bc781284f686dd5005fd350aca7b503c09adbc8d96a20785492224137` |
| [grid.yaml](grid.yaml) | `67ba671a7fe290324f7831b5fed5cbf278ff018b6812963e4a7c5efd1b1bffa6` |
