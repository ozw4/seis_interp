# study_071 decisions

## 2026-11-09 — field を観測近傍で条件付ける

**Status:** 採用。study_067 / study_070 の test を見た**後**に決めた変更であり、事後的であることを開示する。

**Decision:** neural-implicit を重みのみの符号化から、query が近傍の観測トレースを読む条件付き field へ変える。

**Evidence:** study_070 の
[FORGE seed 7321 per-trace table](../../runs/study_070_forge_c3_trace_field/20261109T024720614852Z_63414473e4b2_forge/method_runs/inr_real/evaluation/seed_7321/test_per_trace.parquet)
と study_067 の記録を同一の clean_target 34,153 trace 上で比較した。

| 手法 | 予測時に観測トレースを読むか | 相関中央値 | mean relMSE | 相関 > 0.5 の割合 |
|---|---|---:|---:|---:|
| ReGSI | 読む | 0.7275 | 0.5399 | 0.8307 |
| CCNet5D | 読む | 0.5756 | 0.7176 | 0.6221 |
| DRR | 読む | 0.4589 | 0.7981 | 0.4350 |
| POCS | 読む | 0.4544 | 0.8066 | 0.4320 |
| profile NeRSI | 重みのみ | 0.2600 | 0.9442 | 0.1756 |
| `inr_real` | 重みのみ | 0.2159 | 1.0081 | 0.1424 |

補助的な測定:

- per-trace 最適スケール（oracle）を与えた上限は relMSE **0.8887**。振幅校正だけでは `nersi_profile`
  の 0.9442 を僅かに上回るだけで、床 0.4 dB には届かない。
- skill が崩れるのは**短オフセット側**（offset 332 m で相関 0.0659、>900 m で 0.28 に飽和）。
  残留 moveout 仮説は反証された。急峻な近傍場変化を滑らかな座標場が表現できていない。
- study_070 の grid は採用 cell が角にあり両軸が単調で、傾向の極限はゼロ予測だった。
  機構を変えない範囲の調整では到達しない。

**理由:** 差は約2倍で、grid 内のハイパーパラメータ範囲ではない。field を4座標の連続関数のまま保ったまま
観測を**記憶**から**参照**へ変えるのが、測定が指し示す唯一の方向である。

## 2026-11-09 — 時間表現は共有 trace codec を使う

**Status:** 採用。

**Decision:** 時間方向の encode / decode は `models/trace_codec.py` の `TraceNodeEncoder` /
`TraceNodeDecoder` を使う。study_070 の `Linear(hidden_width → T)` 形式の出力層は使わない。

**Evidence と理由:**

1. **rank 上限の解消。** 単一の線形出力層は予測を自身の幅の部分空間に閉じ込める。study_070 の FORGE は
   `Linear(126 → 2172)` なので、どんな入力に対しても予測は 126 次元の部分空間に収まっていた。
   4 ms・4 s で 5–60 Hz の帯域なら有意な時間係数は約 440 本あり、これは物理的な上限である。
   畳み込み decoder には上限がない（`tests/unit/test_trace_context_field.py` が rank > width を固定）。
2. **パラメータ数が時間長に依存しない。** width 120 で両 canvas とも 362,642 parameters（目標比
   −0.1726%）。study_070 が必要とした `hidden_width` の dataset 差分が消え、宣言する手法差分は
   実行時に上書きされる `model.time_sample_count` の1件だけになる。
3. **時間表現を提案法と等しく保つ。** ReGSI も同じ codec を使うので、対比は空間機構に絞られる。
4. codec の decoder head はゼロ初期化なので、field は**ゼロ予測から出発**して data が支持する分だけ
   離れる。study_070 の予測は相関 0.2159 に対し target RMS の 0.55 倍を出す過信状態だった（NMSE 1.0640）
   ので、これは望ましい事前分布である。最初の更新で勾配が届くのは最終 Conv1d の 2 本だけ、
   2 回目から全 22 本に届く。

## 2026-11-09 — 近傍は field 自身が読む4座標だけで決める

**Status:** 採用。

**Decision:** 近傍は `processing/trace_field_neighbors.py::nearest_observed_positions` が選ぶ。
4正規化座標のユークリッド距離による top-k で、同距離は観測位置順。ReGSI の
`processing/trace_graph_neighbors.py`（4関係・物理スケール・半径制限・edge type）は使わない。

**理由:** 近傍を field が読む座標そのもので定義すれば、neighbourhood は宣言した幾何だけの性質であり、
波形には依存しない。関係グラフを借りると baseline が提案法の機構をさらに取り込み、対比の意味が薄れる。
距離計算は query chunk 単位の総当たりで、一時記憶は `chunk_size` × 観測数であって survey の自乗ではない。

**漏洩の契約:** query が観測トレースの場合は自分自身を文脈から除外する（leave-one-out）。fit は観測集合を
query 側と候補側の両方に渡すので除外が効き、test query も同じ観測候補から引く。
`tests/unit/test_forge_context_field.py::test_a_query_is_never_reconstructed_from_its_own_trace` が
この2つの呼び出しを固定する。

## 2026-11-09 — axis_a は近傍数、axis_b は据え置き

**Status:** 採用。

**Decision:** axis_a = `context.neighbor_count` ∈ {1, 2, 4, 8, 16}、
axis_b = `model.frequency_base` ∈ {1.05, 1.075, 1.1, 1.15, 1.25}。

**理由:** 本 study の仮説は「観測を読むことが差を埋める」なので、読む量そのものを主軸にするのが
仮説に対する直接の測定になる。axis_b は study_067 / study_070 と同一の5値で、Fourier 帯域の意味を
3 study 間で揃える。どちらの軸もパラメータ数を変えないので 25 候補が単一の値を共有する。

**k = 1 の縮退（開示）:** 近傍が1本だと softmax が定数 1 になり、`offset_weight` の 2 層（4 本の
parameter tensor）に勾配が届かない。パラメータ数は変わらないので予算は満たす。これは軸の退化端
（最近傍1本だけを読む）として意味があるので除外せず、そのまま採点する。

## 2026-11-09 — admission は実測で決める

**Status:** 採用。

**Decision:** `cuda_execution.minimum_free_memory_gib.cinr_real` を候補別に宣言する。
値は study_067 の C-0 規則 `max(ceil(最大 reserved GiB) + 2, 3)`。

**Evidence:** 本番形状（width 120、batch 64、40 更新）の `torch.cuda.max_memory_reserved`。

| k | C3 ms/update | C3 reserved | admission | FORGE ms/update | FORGE reserved | admission |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.571 | 0.242 GiB | 3 | 4.176 | 0.951 GiB | 3 |
| 2 | 2.604 | 0.342 GiB | 3 | 5.392 | 1.686 GiB | 4 |
| 4 | 2.629 | 0.523 GiB | 3 | 8.019 | 2.646 GiB | 5 |
| 8 | 3.354 | 0.879 GiB | 3 | 14.515 | 4.578 GiB | 7 |
| 16 | 5.166 | 1.598 GiB | 4 | 25.638 | 8.578 GiB | 11 |

**理由:** 文脈は全レートで畳み込まれるので記憶は k にほぼ比例する。k=16 の FORGE で
`parallel_candidates: 4` でも 44 GiB で、1枚 93.08 GiB の範囲内。

## 2026-11-09 — study 内 YAML の sha256

launch 時点の正本。

| file | sha256 |
|---|---|
| [c3/cinr_real.yaml](c3/cinr_real.yaml) | `6d3f825c46b9c71874e9bb36f3072d059caf619933b2174eb99b7dd9fdf5d211` |
| [c3/config.yaml](c3/config.yaml) | `53e0d1d614e88afba327eea7f06aa6ed4dcfee5be541792d145639f5c5a992b2` |
| [c3/inputs.yaml](c3/inputs.yaml) | `02db0cdeca2570fa8736495e2c8102e82076caf71de5413832475854d4f50ede` |
| [dataset_differences.yaml](dataset_differences.yaml) | `74a8ed39ab7fd2c5f1fbcdbf0cb59fbd78239d8f30064c4736b407202bf2449a` |
| [evaluation.yaml](evaluation.yaml) | `b89a6b8ffb6502f48a033ff1a0eb00fefa7fd3086341d28f3f931dd7cf9373de` |
| [forge/cinr_real.yaml](forge/cinr_real.yaml) | `bf1df5de48af2528545a5aac7fc20b4836e25500d714db39a83e0e3712a29085` |
| [forge/config.yaml](forge/config.yaml) | `90519cc3750ef78ebcae1312f331dd3444112c00eb1346f48237f768d6d8b1ea` |
| [forge/inputs.yaml](forge/inputs.yaml) | `645f339c296102a6e9192a9ca2d515862ee425cd3c8e4a5b65187090ee375021` |
| [grid.yaml](grid.yaml) | `8bdc6371567dd676708642aa56bcf2cc5019b393006024032996d05897990f5f` |
