<!-- 2026-11-03 study_067 最終報告。本番 launch tag study-067-launch-v5（5ad36bd）、解析凍結 tag study-067-analysis-v1（0073160）、解析実行 commit ba805bd（解析コードは凍結 tag と byte 一致）。数値の正本は results/study_067_forge_c3_common_grid/ の CSV と各 run の manifest。 -->

# study_067 最終報告：C3 / FORGE 共通 5×5 探索による5手法比較

## 1. 結論

- **ReGSI は C3 と FORGE の両方で最良です。** 事前登録した判定条件を両 dataset で満たしました。
  - 条件は、4つのベースラインすべてについて、seed 平均のペア差 Δ の 95% cluster bootstrap CI の下限が 0 より大きく、かつ3 seed すべてで Δ_s > 0 であることです。
  - 8対比すべてで Holm 補正後の p は 0.0008 です。B = 10,000 で得られる最小の p（1/10,001）に Holm の係数 8 を掛けた値で、この B で到達できる下限です。
  - relMSE でも8対比すべてで ReGSI が優位です。SNR と relMSE の向きが食い違った対比はありません。
- **床（SNR の下限目標）**
  - C3（> 10 dB）：5手法すべて達成しました。
  - FORGE（> 0.4 dB）：4手法が達成しました。**NeRSI は未達です**（S1 の25候補の最良 validation 0.371 dB、選択 cell の test 0.356 ± 0.014 dB）。
  - 規則どおり追加の調整はせず、未達として報告します。
- **感度**
  - S-a（CCNet の inner mask）：C3 で mask 0.1 が 0.25 を 0.27 dB 上回ったため、登録規則により ReGSI − CCNet@0.1 を co-primary の感度として報告します。C3 では +0.971 dB [0.886, 1.058] で、3 seed とも正でした。
  - S-b（FORGE CCNet の FP32 と bf16）：差は −0.006 dB で、FP32 による救済は発動しません。

## 2. 比較の条件

詳細は [設計書](study_067_design_round2.md) と [decisions](../studies/study_067_forge_c3_common_grid/decisions.md) にあります。ここでは要点だけを示します。

**手法と 5×5 grid**（両 dataset で同一。grid の値は sample 単位です）

| 手法 | 軸 a | 軸 b |
|---|---|---|
| ReGSI（提案法、relational trace-graph GNN） | temporal dilations [d, 2d]、d ∈ {1, 2, 3, 4, 6} | relation ごとの近傍数 k ∈ {2, 3, 4, 5, 6} |
| CCNet5D | patch {[4,8,4,8], [4,12,4,12], [6,12,6,12], [8,12,8,12], [8,16,8,16]} | time dilation {1, 2, 3, 4, 6} |
| NeRSI（profile-wise） | nuclear norm 重み λ ∈ {0, 0.001, 0.01, 0.03, 0.1} | frequency base β ∈ {1.05, 1.075, 1.1, 1.15, 1.25} |
| POCS | 時間窓 {64, 96, 128, 192, 256} | 震源窓 [line, point] {[2,8], [2,16], [4,8], [4,16], [6,16]}（受振窓は固定） |
| DRR | rank {6, 12, 18, 24, 30} | 空間窓 {[4,6,4,6], [4,8,4,8], [6,8,6,8], [8,12,8,12], [8,16,8,16]}（overlap は窓から導出） |

**固定値**（両 dataset 共通。dataset 間の差は付録 A の protocol 表を参照）
- ニューラル3手法：AdamW、lr 1e-3（一定）、wd 0、50,000 更新、1更新 64 trace、EMA 0.999（shadow として 0.9995 / 0.9998 も評価）、model seed 7301。
- ReGSI：V3（inner mask 0.25、node Fourier 0）。CCNet：inner mask 0.25、time kernel 3、cuDNN benchmark、FORGE のみ bf16。NeRSI：座標 jitter 0.5、canvas の時間方向 upsampling（C3 は s = 0、FORGE は s = 3）。
- POCS：K1（32反復、受振窓 [4,8]、overlap [3,6]）。DRR：60反復、damping 2、窓から導出した Hann overlap、batched backend。

**手順**
1. S1：固定 split で25候補を学習し、validation で採点します。
2. 選択：validation の relMSE が最小の候補を選びます（同値は candidate ID 順、diverged は最下位）。
3. 評価：選択した cell を、split を引き直した3 seed（7321–7323）で学習し、test は seed ごとに1回だけ採点します。
4. test の採点は、dataset 内の全手法の S1 が終わった後（barrier）に行います。

主指標は test の mean trace SNR です。C3 は全 test trace、FORGE は clean_target の trace を使い、FORGE では trace ごとの oracle test RMS で振幅を戻します。

## 3. 実行の記録

| 項目 | 内容 |
|---|---|
| launch | 2026-10-24、commit 5ad36bd（tag `study-067-launch-v5`）。凍結 source から起動。prepare は 11:07 UTC、FORGE CCNet の2 shard は 11:07 UTC、他の worker（C3 の全手法と FORGE の残り4手法）は 12:30 UTC に開始 |
| 完了 | C3：2026-11-01 11:08 UTC、FORGE：2026-11-03 13:14 UTC。全 worker が exit 0 |
| 再利用 | identity hash が一致した過去 run を S1 に import：C3 30件（POCS 25件を含む）、FORGE 29件（同） |
| 失敗・再起動 | なし。diverged の run もなし |
| 解析 | 2026-10-24 に `study-067-analysis-v1` として凍結（test の採点より前）。2026-11-03 に commit ba805bd で実行。解析コードは凍結 tag と byte 一致 |
| 計算機 | H100 NVL ×2（外部ジョブと共有）、128 core |

**S1 の実行時間**（run ごとの壁時計の合計、時間。GPU を他の run と共有した時間を含みます。import した run は数えません。DRR は CPU 12 worker の壁時計です）

| 手法 | C3 | FORGE |
|---|---:|---:|
| ReGSI | 85.7 | 175.3 |
| CCNet | 106.6 | 314.2 |
| NeRSI | 121.0 | 51.3 |
| POCS | 0（全件 import） | 0（全件 import） |
| DRR | 179.1 | 362.2 |

seed の学習時間は解析の cost 表に含まれていません。cost 表は barrier 後の採点 run の時間だけを記録するためです（第10節）。

**G0 の記録**（登録では FORGE の各ニューラル手法の最初の run に適用。実装では両 dataset の新規ニューラル run すべての manifest に記録する診断で、run は止めず、選択にも使いません。以下は FORGE の値で、更新 901–1000 の平均 loss / L0 と、出力 RMS / target RMS です）
- CCNet：登録した G0 cell `a01_b02`（shard 0 の先頭）は合格（0.919·L0、0.284）。shard 1 の `a05_b01` が約1分早く始まり、これも合格（0.875·L0、0.352）。新規の27 run（S1 24、seed 3）すべてが合格です。
- ReGSI `a02_b01`：合格（0.853·L0、0.384）。27 run すべてが合格です。
- NeRSI `a03_b01`：不合格（0.999·L0、0.036）。新規の27 run（S1 24、seed 3）もすべて不合格です（0.993–1.000·L0）。10-24 の判断時点で、採用構成を決めた N-1 run は 1,000 更新で 0.996·L0 に留まった後、50,000 更新で 0.500·L0 まで下がっていました。本番の `a03_b01` も 50,000 更新で 0.625·L0、validation 0.269 dB と学習が進んでいたため、実装バグではなく座標ネットの遅い立ち上がりと判断しました（decisions に記録）。

## 4. S1（validation）と選択

| dataset | 手法 | 選択 cell（軸 a / 軸 b） | val relMSE | val SNR (dB) | 25候補の最良 SNR | Spearman ρ（relMSE と SNR の順位） | grid 端 |
|---|---|---|---:|---:|---:|---:|---|
| C3 | ReGSI | a02_b05（d=2 / k=6） | 0.01249 | 20.009 | 20.009（同） | 0.972 | **端**（k=6） |
| C3 | CCNet | a05_b02（[8,16,8,16] / 2） | 0.01667 | 18.803 | 18.844（a05_b01） | 0.995 | **端**（最大 patch） |
| C3 | NeRSI | a02_b03（0.001 / 1.1） | 0.06579 | 12.651 | 12.657（a02_b04） | 0.996 | 内部 |
| C3 | POCS | a01_b05（64 / [6,16]） | 0.06031 | 13.792 | 13.792（同） | 0.997 | **端**（角） |
| C3 | DRR | a05_b03（30 / [6,8,6,8]） | 0.02179 | 17.696 | 17.696（同） | 0.987 | **端**（rank 30） |
| FORGE | ReGSI | a05_b04（d=6 / k=5） | 0.52265 | 3.333 | 3.333（同） | 0.978 | **端**（d=6） |
| FORGE | CCNet | a02_b05（[4,12,4,12] / 6） | 0.70415 | 1.807 | 1.807（同） | 0.994 | **端**（dilation 6） |
| FORGE | NeRSI | a04_b02（0.03 / 1.075） | 0.94256 | 0.371 | 0.371（同） | 0.878 | 内部 |
| FORGE | POCS | a04_b01（192 / [2,8]） | 0.79585 | 1.165 | 1.165（同） | 0.996 | **端**（[2,8]） |
| FORGE | DRR | a03_b02（18 / [4,8,4,8]） | 0.78588 | 1.222 | 1.234（a04_b02） | 0.993 | 内部 |

- 10件中7件で、選択 cell が grid の端にあります。登録どおり grid は拡張せず、端フラグとして報告します。
- **床の判定**（S1 validation の best-of-25）
  - C3：最小は NeRSI の 12.657 dB で、全手法が 10 dB を超えました。
  - FORGE：NeRSI の 0.371 dB だけが 0.4 dB 未満です。
- NeRSI の FORGE は、EMA shadow 0.9998 でも選択 cell の validation は 0.389 dB で、0.4 dB に届きません。shadow は診断用で、選択には使いません。

## 5. 主結果（test、3 seed）

**C3**（全 test trace、mean trace SNR。seed は 7321 / 7322 / 7323）

| 手法 | 平均 ± SD (dB) | seed 別 | ReGSI − 手法 Δ [95% CI] | seed 別 Δ | Holm p |
|---|---:|---|---:|---|---:|
| **ReGSI** | **19.992 ± 0.047** | 20.043 / 19.981 / 19.951 | — | — | — |
| CCNet | 18.861 ± 0.025 | 18.888 / 18.839 / 18.858 | +1.130 [1.045, 1.219] | 1.155 / 1.142 / 1.094 | 0.0008 |
| DRR | 17.745 ± 0.038 | 17.713 / 17.736 / 17.786 | +2.247 [2.168, 2.326] | 2.331 / 2.245 / 2.165 | 0.0008 |
| POCS | 13.804 ± 0.027 | 13.832 / 13.779 / 13.802 | +6.188 [6.021, 6.364] | 6.211 / 6.202 / 6.150 | 0.0008 |
| NeRSI | 12.561 ± 0.007 | 12.556 / 12.559 / 12.569 | +7.431 [7.361, 7.501] | 7.487 / 7.423 / 7.383 | 0.0008 |

**FORGE**（clean_target、trace ごとの oracle RMS で振幅を復元）

| 手法 | 平均 ± SD (dB) | seed 別 | ReGSI − 手法 Δ [95% CI] | seed 別 Δ | Holm p |
|---|---:|---|---:|---|---:|
| **ReGSI** | **3.219 ± 0.023** | 3.193 / 3.226 / 3.237 | — | — | — |
| CCNet | 1.703 ± 0.015 | 1.718 / 1.701 / 1.689 | +1.516 [1.466, 1.565] | 1.475 / 1.525 / 1.548 | 0.0008 |
| DRR | 1.145 ± 0.005 | 1.149 / 1.140 / 1.146 | +2.073 [2.006, 2.140] | 2.044 / 2.086 / 2.090 | 0.0008 |
| POCS | 1.103 ± 0.002 | 1.105 / 1.101 / 1.103 | +2.116 [2.045, 2.186] | 2.088 / 2.125 / 2.134 | 0.0008 |
| NeRSI | 0.356 ± 0.014 | 0.355 / 0.370 / 0.343 | +2.863 [2.760, 2.965] | 2.838 / 2.856 / 2.894 | 0.0008 |

- CI は source-cell 単位の cluster bootstrap による percentile 区間です（B = 10,000、RNG 670703、C3 512 cell、FORGE 192 cell）。1 replicate につき1組の multinomial 重みを、seed・手法・対比・指標で共有します。
- Holm は SNR の8対比と relMSE の8対比を別々の族として補正しました。
- relMSE の差（ReGSI − 手法）はすべて負（ReGSI 優位）で、CI は 0 を含みません：
  - C3：CCNet −0.0039、DRR −0.0089、POCS −0.0461、NeRSI −0.0556。
  - FORGE：CCNet −0.183、DRR −0.262、POCS −0.270、NeRSI −0.407。
- 判定：**C3、FORGE とも「SNR では最良、relMSE でも優位」**です（`best_claims.csv`）。

## 6. 感度

**validation を除いた再計算**：S1 の validation に入った trace（pilot を含む和集合）を除いても、8対比の Δ と CI はほとんど変わりません。例として C3 の CCNet は +1.130 [1.044, 1.219]、FORGE の CCNet は +1.516 [1.467, 1.565] です。判定は両 dataset で同じです。

**seed 単位のペア t 区間（df = 2）**：8対比すべてで下限が正です。下限が最も小さいのは C3 の CCNet で、[1.050, 1.211] です。

**EMA shadow**（選択 cell の test mean trace SNR、3 seed 平均。登録した主値は 0.999）

| dataset | 手法 | 0.999 | 0.9995 | 0.9998 |
|---|---|---:|---:|---:|
| C3 | ReGSI | 19.992 | 19.978 | 19.903 |
| C3 | CCNet | 18.861 | 18.870 | 18.868 |
| C3 | NeRSI | 12.561 | 12.590 | 12.528 |
| FORGE | ReGSI | 3.219 | 3.223 | 3.232 |
| FORGE | CCNet | 1.703 | 1.708 | 1.715 |
| FORGE | NeRSI | 0.356 | 0.363 | 0.376 |

どの shadow でも手法の順位は変わらず、FORGE NeRSI は 0.4 dB に届きません。

**S-a：CCNet の inner mask 0.1 と 0.25**（[感度の集計](../results/study_067_forge_c3_common_grid/20261103T233134Z_ba805bd26f6a_sensitivity/summary.md)）
- 登録した単一 cell の比較
  - C3：`a05_b02` で study_064 の mask 0.1 が 19.073 dB、本番の 0.25 が 18.803 dB です。差は +0.270 dB で、登録規則の 0.2 dB を超えました。
  - FORGE：`a03_b02` で mask 0.1 が 1.478 dB、0.25 が 1.592 dB です。差は −0.114 dB（記述的 CI [−0.143, −0.085]）でした。
- 規則により、**ReGSI − CCNet@0.1 を co-primary の感度として報告します。**
  - C3：study_064 の mask 0.1 の勝者 `a05_b02` を本番と同じ3 seed で学習しました。test は 19.021 ± 0.016 dB です。
  - ReGSI − CCNet@0.1 は **+0.971 dB [0.886, 1.058]**（seed 別 1.040 / 0.959 / 0.915、t 区間 [0.815, 1.128]）です。validation を除いても +0.971 [0.886, 1.058] です。
  - 参考：同じ seed での mask の効果 CCNet@0.1 − CCNet@0.25 は +0.159 dB [0.152, 0.166] です。
  - FORGE は登録どおり単一 cell の差だけを報告します（mask 0.1 の方が悪い）。
- 感度の対比は登録された族ではないため、Holm 補正はしていません。共有 bootstrap 重みは主解析と同一です（検証済み）。

**S-b：FORGE CCNet の FP32 と bf16**
- `a03_b02` で FP32 が 1.587 dB、bf16 が 1.592 dB です。差は −0.006 dB（記述的 CI [−0.031, 0.020]）でした。
- 0.2 dB 未満なので、登録した救済（選択 cell の FP32 3 seed）は発動しません。
- bf16 は FORGE CCNet のメモリ上の必要から使っています。その影響はこの cell では検出できませんでした。

## 7. 副次指標（test、3 seed 平均）

**C3**（全 test trace）

| 手法 | global SNR (dB) | global NMSE | 相関の中央値 | trace SNR p10 / p50 / p90 | 0 dB 未満の割合 |
|---|---:|---:|---:|---|---:|
| ReGSI | 19.080 | 0.0124 | 0.9951 | 16.48 / 20.01 / 23.49 | 0.0000 |
| CCNet | 17.968 | 0.0160 | 0.9942 | 15.65 / 19.22 / 21.72 | 0.0000 |
| DRR | 16.624 | 0.0218 | 0.9921 | 14.00 / 17.82 / 21.41 | 0.0000 |
| POCS | 12.292 | 0.0590 | 0.9822 | 9.24 / 14.17 / 17.83 | 0.0002 |
| NeRSI | 11.732 | 0.0671 | 0.9743 | 9.02 / 12.82 / 15.69 | 0.0000 |

**FORGE**（clean_target）

| 手法 | mean trace SNR：oracle / IDW (dB) | scale 不変 relMSE | 相関の中央値 | trace SNR p10 / p50 / p90（oracle） | 0 dB 未満の割合（oracle） | global SNR（oracle） |
|---|---|---:|---:|---|---:|---:|
| ReGSI | 3.219 / 2.850 | 0.509 | 0.730 | 0.38 / 3.26 / 5.96 | 0.069 | −0.644 |
| CCNet | 1.703 / 1.499 | 0.674 | 0.574 | −0.26 / 1.61 / 3.83 | 0.144 | −0.792 |
| DRR | 1.145 / 1.037 | 0.764 | 0.456 | −0.30 / 0.97 / 2.85 | 0.180 | −0.537 |
| POCS | 1.103 / 0.999 | 0.765 | 0.453 | −0.36 / 0.95 / 2.81 | 0.198 | −0.443 |
| NeRSI | 0.356 / 0.269 | 0.874 | 0.265 | −0.73 / 0.22 / 1.63 | 0.381 | −0.358 |

- FORGE の all_eligible でも順位は同じです（oracle の mean trace SNR は ReGSI 3.226、CCNet 1.699、DRR 1.155、POCS 1.109、NeRSI 0.359）。
- oracle を使わない IDW 版の mean trace SNR と scale 不変の relMSE でも、ReGSI が最良です（clean_target と all_eligible の両方。表の clean_target では順位も同じです）。
- **FORGE の global SNR（oracle 復元）は全手法で負です。** global NMSE は振幅の大きい trace の誤差に重みがかかる指標で、trace ごとに正規化する mean trace SNR とは別の性質を測ります。FORGE の 0 dB はゼロ予測と同じ水準です。IDW 復元では ReGSI だけが正です（clean_target で +0.151 dB、他の4手法は −0.012〜−0.023 dB）。
- IDW 版の relMSE（trace 平均）は all_eligible で極端な値になり、この指標だけは **ReGSI が最下位**です（ReGSI 13.71、CCNet 10.64、POCS 6.09、NeRSI 5.79、DRR 4.83）。relMSE は振幅の誤りに上限がないため、IDW の振幅推定が大きく外れる trace が平均を支配すると考えられます。clean_target では ReGSI が最良です（ReGSI 0.687、CCNet 0.816、DRR 0.847、POCS 0.867、NeRSI 1.006）。

## 8. 探索の診断（S1 validation）

| dataset | 手法 | 期待最良 SNR：1 / 5 / 25 候補 | winner's curse（SNR の楽観、dB） |
|---|---|---|---:|
| C3 | ReGSI | 19.65 / 19.90 / 19.98 | 0.0013 |
| C3 | CCNet | 17.27 / 18.51 / 18.82 | 0.0015 |
| C3 | NeRSI | 10.61 / 12.52 / 12.65 | 0.0294 |
| C3 | POCS | 12.93 / 13.53 / 13.75 | 0.0122 |
| C3 | DRR | 13.94 / 16.46 / 17.52 | 0.0016 |
| FORGE | ReGSI | 3.23 / 3.31 / 3.33 | 0.0219 |
| FORGE | CCNet | 1.59 / 1.75 / 1.80 | 0.0115 |
| FORGE | NeRSI | 0.09 / 0.25 / 0.34 | −0.0010 |
| FORGE | POCS | 0.78 / 1.08 / 1.16 | 0.0050 |
| FORGE | DRR | 0.67 / 1.07 / 1.22 | 0.0015 |

- 期待最良は25候補から復元抽出した場合の値です。winner's curse は、S1 の source-cell bootstrap で in-bag 選択し、out-of-bag で評価した差です。
- ReGSI は1候補でも他手法の25候補の最良を上回ります（C3 19.65 対 CCNet 18.82、FORGE 3.23 対 CCNet 1.80）。結論は探索予算の違いに依存しません。
- 選択の楽観は最大でも 0.03 dB で、第5節の差と比べて無視できます。

## 9. 公平性の宣言

設計書 (5) の宣言項目への対応です。手法ごとの protocol 表、調整の履歴、過去の勝ち設定の扱いは、末尾の付録 A にあります。

**守った約束**
- S1 を見た後も test を見た後も、固定値・grid・選択規則は変えていません（post-S1 の変更はありません。全 run が launch 時の凍結 source と config で動いています）。S1 barrier は全手法の S1 完了で自動的に通過する仕組みで、ReGSI が最良かどうかを条件にしていません。
- 選択は validation の relMSE 最小です。early stopping、途中結果による打ち切りや優先順位の変更はしていません。実行順の並べ替えはメモリによるものだけです（第10節の7）。
- 起動は clean worktree の凍結 source から行い、grid.yaml と各 config の SHA256、commit と tag を記録しました。評価 script と `evaluation.yaml` は test の採点前に凍結し、tag `study-067-analysis-v1` と SHA256 を記録しました。
- 未達の目標（FORGE NeRSI の床）は未達のまま報告します。

**宣言した非対称と dataset 間の差**
- inner mask は入力を条件とする ReGSI と CCNet だけに適用します。NeRSI の support mask は N-1 で不採用となり、どの手法にも使っていません。
- gradient clip（1.0）は ReGSI だけです。発動率と、他のニューラル手法の would-clip 率は付録 A に示します。
- 精度：FORGE の CCNet だけが bf16 です（FP32 では a05 で約 110 GiB 必要で、GPU 1 枚の 93 GiB を超えます）。ReGSI は厳密な省メモリ機能で FP32 のままです。この非対称の影響は S-b で −0.006 dB でした。
- パラメータ数：C3 ではニューラル3手法とも 363,269 ±5% に入ります（ReGSI 346,885、CCNet 363,292、NeRSI 363,269）。FORGE も同じ構造で、NeRSI だけが宣言した canvas 適応（s = 3）により 341,485（−6.0%）です。
- grid の時間方向の値は sample 単位です（ReGSI の dilation だけは2倍に間引いた frame 単位）。C3 は 8 ms、FORGE は 4 ms なので、同じ値でも FORGE の時間幅は C3 の半分です。
- FORGE の loss は 2,172 sample の canvas 全体にかかり、有効な sample は 46% です。
- CCNet、POCS、DRR は格子の cell 中心で、ReGSI と NeRSI は実座標で予測します。study_060 では regsi_grid と regsi_real の差は約 0.07–0.1 dB でした。
- FORGE の主指標は、全手法とも trace ごとの oracle test RMS で振幅を戻します。
- model seed は 7301 に固定しています。

**第2弾 pilot の判定**（事前登録した規則による。ベースラインの判定は ReGSI のスコアを参照しません。R-1 は ReGSI 自身の値で判定します。値は S1 validation の mean trace SNR、FORGE は clean_target）

| ID | 内容 | 結果 | 判定 |
|---|---|---|---|
| R-1 | C3 ReGSI V3 `a01_b05` | 19.914 dB（閾値 19.812） | V3 を維持 |
| C-0 | FORGE CCNet の gate とメモリ | G0 は `a01_b02`・`a05_b02` とも合格。cuDNN benchmark は 1.34 倍速 | 予算を維持、cuDNN benchmark を両 dataset で採用 |
| C-1 | CCNet time kernel 3 対 9（同じパラメータ予算：363,292 対 363,227） | tk9 − tk3：C3 −0.081、FORGE +0.069 | T1 不成立、tk3 を維持 |
| N-1 | FORGE NeRSI の床の救済 | s = 3：P3 比 +0.652。support mask：−0.065。jitter なし：−0.302 | s = 3 を採用、support mask は不採用、jitter は保持 |
| N-2 | NeRSI の EMA 0.9998 | shadow − 主値：+0.009（条件 +0.15） | 不採用。床の救済はここで終了 |
| P-1 | POCS の K1 案（screening 由来） | C3 13.792（+2.915）、FORGE 1.165（+0.270） | T2 を満たし採用 |
| D-1 | DRR の窓から決める overlap（screening 由来） | FORGE 1.098（閾値 0.656）、C3 16.087（閾値 15.125） | T2 を満たし採用 |
| D-2 | 最も重い DRR cell の時間と RSS | 12 worker での候補 wall は 72 h 未満の見込み | 開始前報告の条件に当たらず |

**規則を登録した時点で既に見ていた結果**（設計書 (5) の12）
- POCS：P1b grid と K1 の screening（約15案）。P-1 の規則はこれを見た後に作りました。
- DRR：C3 の S1（study_063）、D7 の C3（16.087 dB）と FORGE D5b（1.089 dB）。D-1 の規則もこれを見た後に作りました。
- ReGSI の P2 `a01_b01`、NeRSI の P3（FORGE −0.689 dB）。
- 閾値の基準にした値：R-1 は study_064 の ReGSI `a01_b05`（20.012 dB）、D-1 の FORGE は study_066 の DRR（0.706 dB）。設計書 (1) の現状表の値も既知でした。

**負の結果**（不採用とした項目。括弧内は設計書に記載した根拠の値で、出典は過去 study、study_067 設計時の screening（validation のみ）、第2弾 pilot のいずれかです。第2弾の項目には ID を付けます。run までたどれない値もあります（付録 A））
- ReGSI：wd 0.01（+0.013 dB。wd を 0 に統一するユーザー規則により不可）、EMA 0.9995（−0.007）、EMA 0.9998（−0.13）、mask 0.30（−0.04）、3 rounds（params +24.7%）、width 120 / 136 と time_downsample 1 / 4（params の帯域外）、anisotropy（+0.004）、augmentation と midpoint-free features（証拠のない ReGSI 専用の改良）。
- CCNet：time kernel 9（C-1）。
- NeRSI：λ を 0.1 より大きくする（本番 S1 でも λ 0.1 の5候補は C3 で 3.9–5.8 dB、FORGE でゼロ予測と同じ 0 dB です）、周波数の hard cap（C3 −0.45 dB、study_046 の target 値）、rank-2 latent（FORGE −0.33〜−0.41 dB）、encoder / decoder の幅と profile batch の変更、support mask と jitter を外す案（N-1、−0.065 / −0.302 dB）、EMA 0.9998（N-2）。
- POCS：threshold_end 0.001（C3 −2.5 dB）、受振窓 [8,8]（FORGE −0.36）、予測への low-pass（+0.004 以下）は設計時の screening、threshold_end 0.2（FORGE 0.321 対 0.440）は study_066 の値です。K2 案は設計批評で削除しました。
- DRR：12 反復（C3 10.12、FORGE 0.82）、30 反復（C3 14.00）、damping 1（C3 10.24、FORGE 0.449）、damping 3（C3 13.77、FORGE 0.054）。主に設計時の screening の値で、FORGE の 0.82 は出典が文書にありません（付録 A-5）。

## 10. 事前登録からの逸脱と開示

1. **S-b の時機**：登録規則は「FP32 − bf16 ≥ 0.2 dB なら FORGE の seed を始める前に報告」でした。S-b は 11-01 20:50 UTC に開始し、11-02 13:00 UTC に完了しました。FORGE CCNet の seed の学習・予測（11-01 01:29 UTC までに完了）の後で、test の採点（11-03）の前です。結果は −0.006 dB で救済の条件に当たらないため、結論への影響はありません。
2. **Sprint 3 の時機**：解析 script は、本番の最初の test 採点より前に凍結しました（`study-067-analysis-v1`）。seed の予測は各手法の S1 直後に始まりましたが、test の採点は全手法の S1 barrier 後でした。一時的に「採点が先に始まった」と誤記し、decisions で訂正済みです。
3. **解析の実行 commit**：凍結 tag ではなく ba805bd で実行しました。解析コードと `evaluation.yaml` は tag と byte 一致です。manifest の worktree_changes には、解析の開始時点で study README が未 commit の変更状態だったことが記録されています。README は解析の入力ではありません。
4. **評価 seed の名前空間**：7321–7323 は、study_067 の過去の prepare（14件）で partition だけが事前生成されていました。学習・採点には一度も使われていません。他の study での使用も検出されていません。
5. **S-a の C3 の対**：登録した単一 cell の比較は、study_064（旧実装の乱数 stream）の mask 0.1 の値と、本番の 0.25 の値の比較です。新しい3 seed は現行実装で mask だけを変えています。
6. **G0**：NeRSI の FORGE は 1,000 更新時点の G0 で不合格でしたが、バグではないと判断して続行しました（第3節）。
7. **実行順**：登録どおり、優先度順のうち、その時点の空きメモリに収まる最上位の候補から実行しました。メモリによる並べ替えは設計 (4)・(5) の15で認めたもので、逸脱ではありません。identity hash は実行順で変わりません。ただし cuDNN benchmark を使う CCNet は bitwise の再現性を保証しません。
8. **cost 表の欠落**：解析の cost 表は seed の学習時間を含みません（採点 run だけを記録）。
9. **diverged な評価 seed の扱い**：3 seed がそろわない対比の平均・CI・t 区間と、その族の Holm p を「算出不能」と表示する規則は、設計書にはありません。launch 前の decisions（3完了未満の平均・SD を公表しない）と、解析凍結時の decisions（欠けた箇所の表示。「解析規則の変更ではない」と位置づけ）に記録しました。本番では diverged が0件で、発動していません。
10. **設計書の記述の不正確さ（ReGSI の dilation の時間幅）**：設計書 (5) の6（README と decisions にも同じ文）は「grid の値は sample 単位」とし、「dilation 6 は FORGE で 24 ms、C3 で 48 ms」としています。CCNet の time dilation はこのとおりです。しかし ReGSI の dilation は2倍に間引いた latent frame 上で効くため、ReGSI の d=6 は FORGE で 48 ms、C3 で 96 ms です。grid の値と実行には影響しない、記述だけの問題です（付録 A-1）。
11. **clip 発動率の事前の値**：設計書 (2) の「C3 の S1 で平均 0.53%・最大 1.26%、FORGE で 0.01% 以下」は study_064・study_066 の値でした。本番の S1 25 cell では、C3 が平均 0.46%・最大 0.68%、FORGE が平均 0.009%・最大 0.046% で、FORGE の6 cell が 0.01% を超えました（付録 A-1）。
12. **ギャザー比較図の追加**：ギャザー図は登録した図の一覧になく、test の結果を見た後にユーザーの依頼で追加しました。表示する seed（最初の評価 seed 7321）と断面（grid の中心）は振幅を見ずに決めており、選択・数値・判定には影響しません。

## 11. 本文に書く注意

- FORGE の 0 dB はゼロ予測と同じ水準です。FORGE の値は C3 と桁が違います。
- 3 seed の test 集合は、C3 では互いに約80%、FORGE では約70%重なります。SD は partition とデータ順の変動を表し、初期化の seed は 7301 に固定しています。
- FORGE CCNet の bf16 と ReGSI の FP32 の非対称は宣言済みです。S-b で影響は検出されませんでした。
- grid 端の選択が多く（10件中7件）、grid 外の設定でさらに良くなる可能性は否定できません。全手法に同じ条件です。

## 12. 成果物と再現

- 主解析：`results/study_067_forge_c3_common_grid/20261103T231944Z_ba805bd26f6a_analysis/`（表、図、manifest）。派生 per-trace parquet（316 file、約 444 MiB）と bootstrap 重み（npz）は Git に入れていません。README の command を新しい出力先（`--output`）で実行すると再生成されます（既存の出力先には書き込めません）。重みは RNG 670703 から決定的に作られます。感度の集計はこの2つを読むため、感度を再現するときは先に主解析を再生成します。Git に入れた `cost.csv` と `search_scores.csv`（各 dataset 版を含む6 file）は、run の path を絶対 path からリポジトリ相対 path に置き換えています（値は変えていません）。
- 感度：`results/study_067_forge_c3_common_grid/20261103T233134Z_ba805bd26f6a_sensitivity/`
- ギャザー比較図：`results/study_067_forge_c3_common_grid/20261104T011853Z_35d86916bcc7_gathers/`（[README](../results/study_067_forge_c3_common_grid/20261104T011853Z_35d86916bcc7_gathers/README.md)）。選択 cell の評価 seed 7321 の test 予測を、study_042 / study_060 と同じ体裁で示します（C3 は3断面、FORGE は4ギャザー × 物理振幅・trace RMS 版）。可視化のみで、表示した trace の SNR は保存済みの score と一致します。1200 dpi の PNG / PDF は Git に入れず、README の command で再生成できます。
- 再現 command は [study README](../studies/study_067_forge_c3_common_grid/README.md) にあります。
- 本番 run：`runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces`（C3）、`…/20261024T110745841828Z_5ad36bdb74fd_forge`（FORGE）

## 付録 A. 手法別の protocol・調整の履歴・過去の勝ち設定

設計書 (5) の1と11で登録した表です。各手法の節は、本番 run・study 文書・過去の run を読み直して作成し、別の担当が出典を開いて検算しました。出典はリポジトリ相対の path と行番号です。「扱い」の列は登録した3つのラベル（第2弾で試した／ユーザー規則で不可／証拠により不採用）を使い、現行の固定値や変更していない grid の cell に既に含まれる設定には、登録外の第4ラベル「現行に含まれる」を付けました。「選択に使った指標」の test/target は、文書が評価 target での選択を明記している場合だけに付けています。

### 付録 A-1. ReGSI（`regsi_real`）

本番で選ばれた cell は C3 が `a02_b05`（dilations [2,4]、k=6）、FORGE が `a05_b04`（[6,12]、k=5）です。出典には次の略記を使います（いずれもリポジトリ相対パスです）。

- `[C3run]` = `runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces`、`[FGrun]` = `runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge`
- `[C3]` = `[C3run]/method_runs/regsi_real`、`[FG]` = `[FGrun]/method_runs/regsi_real`
- C3 cfg = `[C3]/S1/a02_b05/config.resolved.yaml`、FG cfg = `[FG]/S1/a05_b04/config.resolved.yaml`（seed の resolved config は device を除き S1 と同一です）
- `[SRC]` = `[C3run]/source/src/seis_interp`（凍結 source）、`[S067]` = `studies/study_067_forge_c3_common_grid`、`[V4]` = `runs/study_062_c3_v4_fair_grid`、設計 = `reports/study_067_design_round2.md`

#### Protocol

| 項目 | C3 | FORGE | 出典 |
|---|---|---|---|
| optimizer/algorithm | AdamW | AdamW | C3 cfg:22、FG cfg:22 |
| 学習率 | 1e-3（一定） | 1e-3（一定） | C3 cfg:23,27、FG cfg:23,27 |
| weight decay | 0 | 0 | C3 cfg:25、FG cfg:25 |
| 更新数/反復数 | 50,000 更新 | 50,000 更新 | C3 cfg:24、FG cfg:24；`[C3]/S1/a02_b05/manifest.json:81`、`[FG]/S1/a05_b04/manifest.json:110` |
| 1更新の loss trace 数 | 64（`masked_trace_mse`、384 sample 全長） | 64（同、2,172 sample の canvas 全体、有効 46%） | C3 cfg:29,70、FG cfg:29,70；`[FG]/S1/a05_b04/manifest.json:131`；設計:231 |
| inner mask（名目/実効） | 0.25 / 0.24957（5,888/23,593） | 0.25 / 0.24595（3,264/13,271） | `[S067]/c3/regsi_real.yaml:30`、`[S067]/forge/regsi_real.yaml:30`；C3 cfg:30、FG cfg:30；`[SRC]/processing/c3_region_search.py:62-73` |
| データの再利用・勾配の鮮度 | 1 mask の隠し trace を 64 本×92 更新で巡回。毎更新 fresh 勾配 | 同、64 本×51 更新。毎更新 fresh 勾配 | `[SRC]/training/c3_poc_trace_graph_episodes.py:53-65`；`[S067]/README.md:273`；設計:224 |
| clip（発動率）または would-clip 率 | norm 1.0 を適用。S1 0.272%、seed 0.304/0.210/0.276% | norm 1.0 を適用。S1 0.036%、seed 0.010/0.020/0.052% | C3 cfg:31、FG cfg:31；`[C3]/S1/a02_b05/manifest.json:97`、`[C3]/evaluation/seed_<s>/manifest.json:128`；`[FG]/S1/a05_b04/manifest.json:126`、`[FG]/evaluation/seed_<s>/manifest.json:185`（s = 7321–7323） |
| EMA（primary と shadow） | primary 0.999（fixed_final_ema）。S1 val：0.999/0.9995/0.9998 = 20.009/19.994/19.927 dB | 同。S1 val（clean_target）= 3.333/3.338/3.346 dB | C3 cfg:33-37,71、FG cfg:33-37,71；`[C3]/S1/a02_b05/manifest.json:123,144,165`；`[FG]/S1/a05_b04/manifest.json:179,229,279` |
| 精度 | FP32（mixed_precision off）、TF32 有効、cuDNN deterministic | 同。省メモリ3機能は exact-by-design | C3 cfg:28,32、FG cfg:28,32,74-79；`[SRC]/training/forge_mvp_methods.py:310-313` |
| パラメータ数 | 346,885（363,269 比 −4.51%） | 346,885（同） | `[C3]/S1/a02_b05/manifest.json:69`、`[FG]/S1/a05_b04/manifest.json:98`；`[S067]/c3/config.yaml:39-40` |
| model seed | 7301（S1 trace seed 620702、評価 seed 7321–7323） | 7301（S1 trace seed 600702、評価 seed 同） | C3 cfg:72、FG cfg:72；各 run の manifest.json:5-6 |
| dataset 固有の固定差：時間・振幅 | 8 ms・384 sample、整列なし、observed_global_rms、trace_rms false | 4 ms（1 ms から decimation）・1,001 sample、500 m/s 整列（canvas 2,172）、trace_oracle、trace_rms true | `[S067]/dataset_differences.yaml:3-8`；`[S067]/c3/config.yaml:16-17,28-29,34`；`[S067]/forge/config.yaml:15-16,26,28,32-33,50-53`；FG cfg:80-82 |
| dataset 固有の固定差：split | 欠損 0.8、[16,32,8,32]、eligible 131,072（自然欠損 0）、観測 26,214（学習 23,593）、test 104,858 | 欠損 0.7、[6,32,8,32]、eligible 49,151（自然欠損 1）、観測 14,745（学習 13,271）、test 34,406 | `[S067]/c3/config.yaml:9-24,52-54`；`[S067]/forge/config.yaml:8-22,59-60`；`[S067]/dataset_differences.yaml:2,14-15,17-18,30-42`；`[C3]/S1/a02_b05/manifest.json:24`、`[FG]/S1/a05_b04/manifest.json:23` |
| dataset 固有の固定差：評価・実行 | 全 val trace（2,621）の relMSE で選択 | clean_target（val 1,471/1,474）の relMSE で選択。省メモリ3機能 on | `[S067]/forge/config.yaml:34-40`；`[S067]/dataset_differences.yaml:19-29,57-64`；FG cfg:74-79；`[C3run]/workers/regsi_real_selection.json`、`[FGrun]/workers/regsi_real_selection.json` |
| grid | dilations [d,2d]（d∈{1,2,3,4,6}）× k∈{2..6}。d は 2 倍間引き後の frame 単位（d=6 は 96 ms）。選択 a02_b05 は k=6 の grid 端 | 同じ grid（d=6 は 48 ms）。選択 a05_b04 は d=6 の grid 端 | `[S067]/grid.yaml:2-44`；C3 cfg:6、FG cfg:6；`[SRC]/models/trace_codec.py:39-45,60`、`[SRC]/models/relational_trace_graph.py:89-95,526-531`；設計:286；`[C3run]/workers/regsi_real_selection.json`、`[FGrun]/workers/regsi_real_selection.json` |

#### 調整の履歴

| study | run 数 | 調整した内容 | 選択に使った指標 | 出典 |
|---|---|---|---|---|
| study_044（C3 v3 窓） | 28 dir（完了25、未完3） | inner mask・Fourier・幅・EMA・k・dilation・幾何 scale・kernel・pooling・更新数（現行の幅 128・500 m・attention RMS の由来） | test/target（評価 target を見て選択） | `studies/study_044_c3_v3_gnn_target_tuning/README.md:17`；設計:240；`runs/study_044_c3_v3_gnn_target_tuning/`（3 dir は `relational_trace_graph/` なし） |
| study_058（FORGE 2 ms） | 6 dir（結果あり3） | EMA 0.9995、wd 0.05–0.2、inner mask 0.2、micro-batch。採用なし | test/target（評価 target を見て選択） | `studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:33-35`；`runs/study_058_forge_2ms_random80_align500_matched20k/*regsi*_matched_tuning/search_results.csv` |
| study_060（FORGE 2 ms） | 32（S1–E 16、R 10、final 6） | topology×mask、Fourier 0/8・anisotropy、wd、mask 0.30、seed | validation のみ | `reports/forge_060_experiment_and_search_conditions.md:41-49,199-224,358` |
| study_062_c3_v4_fair_grid（C3） | 20 dir（完了16、R 4 件中断） | S1 topology×mask（9）、S2 Fourier 0/8・anisotropy、S3 wd、E mask 0.30 | validation のみ | `[V4]/fair_grid/regsi/{S2,S3,E,R}/stage.json`；`[V4]/20261008T212725869571Z_05c92e79ce80_regsi_real_matched_tuning/e_imf30/resolved_config.yaml:6,28-29,65,74` |
| study_062_c3_region_grid（C3） | 25（S1） | 現行と同じ dilation×k grid（lr 1e-4、relMSE、1000 m） | validation のみ | `studies/study_062_c3_region_grid/grid.yaml:2-44`；`studies/study_062_c3_region_grid/regsi_real.yaml:22,60,65`；`runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/workers/regsi_real_selection.json` |
| study_063（C3）/ study_065（FORGE） | 063：S1 16/25 で停止。065：OOM と 200 更新で停止 | 063：062 条件の同 grid（勝者なし）。065：結果なし | 選択なし | `runs/study_063_c3_trace_grid/20261011T145500928121Z_05c92e79ce80_traces/workers/regsi_real.json`；`studies/study_063_c3_trace_grid/regsi_real.yaml:22,60,65`；`runs/study_065_forge_expanded_grid/*/workers/regsi_real.json` |
| study_064（C3） | 25（S1）+ 3 seed | 同 grid、044 基準の固定値（mask 0.1、Fourier 16、500 m、lr 1e-3、MSE） | validation のみ | `studies/study_064_c3_reference_grid/decisions.md:3-17`；`runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/workers/regsi_real_selection.json` |
| study_066（FORGE 4 ms） | 完了2（a01_b05、a01_b01）+ 停止1 | 064 勝者の移植。grid は 2/25 で停止 | 選択なし | `runs/study_066_forge_c3_reference/20261021T064203734700Z_05c92e79ce80_regsi_restart/summary.csv:3`；`runs/study_066_forge_c3_reference/20261021T230418979821Z_05c92e79ce80_regsi_remaining24_grid/regsi_real_summary.csv:2-3`；`studies/study_066_forge_c3_reference/decisions.md:78-86` |
| study_067 P2（第1弾） | 6（V1–V3×2 dataset）+ 親 baseline 2 | inner mask 0.1→0.25、Fourier 16→0。V3 を採用 | validation のみ | `[S067]/decisions.md:73-82`；`[S067]/README.md:34` |
| study_067 R-1（第2弾） | 1 | C3 a01_b05 で V3 を確認（19.914 ≥ 19.812 dB） | validation のみ | `[S067]/decisions.md:130`；設計:126 |

#### 過去の勝ち設定の扱い

| study | 勝った設定 | dataset | 扱い | 根拠 | 出典 |
|---|---|---|---|---|---|
| study_057 | mask 0.1 + Fourier 16（6手法中 relMSE 最小 0.7218） | FORGE 2 ms | 証拠により不採用 | P2 で V3 が FORGE +1.108 dB | `runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run/tables/mvp_metrics.csv:6`；`studies/study_057_forge_2ms_random70_trace_rms/regsi_real.yaml:25,35`；`[S067]/decisions.md:73-82` |
| study_057 | q32・10k 更新・relMSE・scale 1000 m・mean pooling | FORGE 2 ms | ユーザー規則で不可 | 064 でユーザーが 044 基準と学習条件の統一を指定 | `studies/study_057_forge_2ms_random70_trace_rms/regsi_real.yaml:6,29,34,59-60,63`；`studies/study_064_c3_reference_grid/decisions.md:3-6,15-17` |
| study_057 | k2・[1,2]・W128・EMA 0.999 | FORGE 2 ms | 現行に含まれる | grid cell a01_b01 と現行の固定値 | `studies/study_057_forge_2ms_random70_trace_rms/regsi_real.yaml:10,15-17,38,40`；`[S067]/grid.yaml:2-44` |
| study_058 | ReGSI-grid EMA 0.9995（test relMSE 0.9233 対 base 0.9249） | FORGE 2 ms | ユーザー規則で不可 | EMA は 0.999 に統一。本番 FORGE では 0.9995 shadow が +0.005 dB | `runs/study_058_forge_2ms_random80_align500_matched20k/20260923T191157702869Z_25c5193588b2_regsi_grid_matched_tuning/search_results.csv:2`；`studies/study_058_forge_2ms_random80_align500_matched20k/README.md:40`；設計:42；`[FG]/S1/a05_b04/manifest.json:179,229` |
| study_058 | ReGSI-grid wd 0.05（test relMSE 0.9092） | FORGE 2 ms | ユーザー規則で不可 | wd は 0 に統一。ReGSI は V3 で凍結 | `runs/study_058_forge_2ms_random80_align500_matched20k/20260923T213037501970Z_25c5193588b2_regsi_grid_matched_tuning/search_results.csv:2`；設計:20,40；`[S067]/decisions.md:98` |
| study_059 | 隠す本数を 64 の倍数に揃える mask（0.0998 = 1,472 本） | FORGE 2 ms | 現行に含まれる | 規則 64·floor(0.25N/64) として残る。値は P2 で 0.25 に置換 | `studies/study_059_forge_2ms_random70_align500_matched20k/regsi_real.yaml:35,66-69`；`[SRC]/processing/c3_region_search.py:62-73` |
| study_060 | inner mask 名目 0.25（S1 勝者、V 0.6685） | FORGE 2 ms | 現行に含まれる | P2 V3 で採用 | `reports/forge_060_experiment_and_search_conditions.md:217`；`[S067]/decisions.md:73-82` |
| study_060 | node Fourier 0（S2 勝者、V 0.5891） | FORGE 2 ms | 現行に含まれる | P2 V3 で採用 | `reports/forge_060_experiment_and_search_conditions.md:218`；`[S067]/decisions.md:73-82` |
| study_060 | wd 0.01（S3 勝者、V 0.5869、R で採用） | FORGE 2 ms | ユーザー規則で不可 | wd は 0 に統一（設計は +0.013 dB で凍結） | `reports/forge_060_experiment_and_search_conditions.md:219,222`；`studies/study_064_c3_reference_grid/decisions.md:15-17`；設計:40,52 |
| study_060 | EMA 0.999（R で最良） | FORGE 2 ms | 現行に含まれる | 現行の primary | `reports/forge_060_experiment_and_search_conditions.md:222`；C3 cfg:33 |
| study_060 | topology k1/r3/W112、dilations [1,2,4] | FORGE 2 ms | ユーザー規則で不可 | grid の軸と幅 128 は固定。params 355,959 は ±5% 内 | `reports/forge_060_experiment_and_search_conditions.md:224,371`；設計:8,47 |
| study_062_c3_v4 | inner mask 0.25（S1 勝者 k2r2w128、relMSE 0.02129） | C3 | 現行に含まれる | P2 V3 と同じ値 | `[V4]/20261008T114651290388Z_05c92e79ce80_regsi_real_matched_tuning/search_results.csv:2` |
| study_062_c3_v4 | node Fourier 8（S2 勝者、0.02021 対 Fourier 0 0.02071） | C3 | 証拠により不採用 | FORGE 060 S2 では Fourier 0 が優位（0.5891 対 0.6108）。文書上の分類なし | `[V4]/20261008T171714011492Z_05c92e79ce80_regsi_real_matched_tuning/search_results.csv:2`；`[V4]/20261008T171713832029Z_05c92e79ce80_regsi_real_matched_tuning/search_results.csv:2`；`runs/study_060_forge_fair_grid/20260928T121705608184Z_25c5193588b2_regsi_real_matched_tuning/search_results.csv:2`；`runs/study_060_forge_fair_grid/20260928T101843334885Z_25c5193588b2_regsi_real_matched_tuning/search_results.csv:2` |
| study_062_c3_v4 | wd 0.01（S3 勝者、0.020204 対 base 0.020207） | C3 | ユーザー規則で不可 | wd は 0 に統一 | `[V4]/20261008T191218325073Z_05c92e79ce80_regsi_real_matched_tuning/search_results.csv:2`；`[V4]/fair_grid/regsi/S3/stage.json`；`studies/study_064_c3_reference_grid/decisions.md:15-17` |
| study_062_c3_v4 | inner mask 0.30（E 勝者、S3 勝者 wd 0.01 比 +0.028 dB） | C3 | 証拠により不採用 | 設計は FORGE 060 の −0.04 dB を根拠に凍結 | `[V4]/20261008T212725869571Z_05c92e79ce80_regsi_real_matched_tuning/search_results.csv:2`；`[V4]/fair_grid/regsi/E/stage.json`（decision.e_base）；`[V4]/fair_grid/regsi/R/stage.json`；設計:52 |
| study_062（region） | a01_b05（[1,2], k6）を lr 1e-4・relMSE・1000 m で | C3 | ユーザー規則で不可 | ユーザーが lr 1e-3・MSE・500 m に変更。cell は grid に残る | `runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/workers/regsi_real_selection.json`；`studies/study_062_c3_region_grid/regsi_real.yaml:22,60,65`；`studies/study_064_c3_reference_grid/decisions.md:3-6,15-17` |
| study_064 | a01_b05（[1,2], k6）+ mask 0.1・Fourier 16（val 20.012 dB） | C3 | 第2弾で試した | R-1 で V3 が 19.914 ≥ 19.812 dB となり V3 を維持。本番は a02_b05 | `runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/workers/regsi_real_selection.json`；`[S067]/decisions.md:130`；設計:126 |
| study_066 | 064 勝者の移植（val 2.158 dB、完了2 cell 中の最良） | FORGE 4 ms | 証拠により不採用 | P2 で mask・Fourier を V3 に置換（FORGE +1.108 dB）。cell は grid に残る | `runs/study_066_forge_c3_reference/20261021T064203734700Z_05c92e79ce80_regsi_restart/summary.csv:3`；`runs/study_066_forge_c3_reference/20261021T230418979821Z_05c92e79ce80_regsi_remaining24_grid/regsi_real_summary.csv:2-3`；`[S067]/decisions.md:73-82` |
| study_067 P2 | V3（mask 0.25 + Fourier 0） | C3・FORGE | 現行に含まれる | 事前規則で採用（validation のみ） | `[S067]/decisions.md:73-82`；`[S067]/README.md:34` |
| study_067 R-1 | C3 a01_b05 の V3（19.914 dB） | C3 | 第2弾で試した | 閾値 19.812 dB を満たし V3 を維持 | `[S067]/decisions.md:130`；`runs/study_067_forge_c3_common_grid/20261022T162308560491Z_05c92e79ce80_r1_c3_regsi_v3_a01_b05` |

注：「現行に含まれる」は、登録した3ラベル（第2弾で試した／ユーザー規則で不可／証拠により不採用）の外のラベルです。現行の固定値、または変更していない共通 grid の cell に当たる設定に付けました。study_067 の P2・R-1 は対象範囲（studies 057–066）の外ですが、参考として載せています。

#### 補足

- **clip の発動率**：設計:49 と `[S067]/decisions.md:266` にある「C3 の S1 で平均 0.53%・最大 1.26%、FORGE で 0.01% 以下」は、study_064（C3）と study_066（FORGE）の値で、本番の値とは一致しません。本番の S1 全25 cell では C3 が平均 0.460%・最大 0.676%（a02_b01）、FORGE が平均 0.0086%・最大 0.046%（a04_b05）でした。FORGE では、本番4 run のうち3 run（S1 0.036%、seed 7322 0.020%、seed 7323 0.052%）と、S1 の6 cell が 0.01% を超えます（`[C3]/S1/*/manifest.json`、`[FG]/S1/*/manifest.json`。import した cell の値は import 元の pilot の manifest から取りました）。
- **EMA shadow と凍結値**：C3 では primary の 0.999 が最良で、S1 では 0.9995 が −0.014 dB、0.9998 が −0.081 dB でした。FORGE では逆に、EMA が長いほど良い結果でした（S1 で 0.9995 が +0.005 dB、0.9998 が +0.013 dB。3 seed も同じ順）。shadow は診断専用で、選択には使っていません。設計の凍結値（wd 0.01 +0.013、EMA 0.9995 −0.007、EMA 0.9998 −0.13、mask 0.30 −0.04、anisotropy +0.004 dB）は、study_060 FORGE の validation relMSE の 10·log10 比として再現できます（`runs/study_060_forge_fair_grid/fair_grid/regsi/final/stage.json` の seed_scores など）。
- **study_062_c3_v4_fair_grid**：C3 で study_060 と同じ型の S1–R 探索を行った run 群です。条件は 20k 更新、lr 1e-3、relMSE、scale 1000 m で、S1 は Fourier 16 でした。R の4 run は中断しています。study 文書はリポジトリに残っていないため、`[V4]` の search_results.csv と stage.json から内容を再構成しました。S3 の wd 0.01 と base の差は +0.001 dB で、実質的に同じ値です。
- **ラベルの判断**：C3 の Fourier 8 は、設計にも decisions にも分類がありません。FORGE の study_060 S2 で Fourier 0 が 0.157 dB 優位だったため、「証拠により不採用」としました。wd 0.01 と study_057 の学習条件は、study_064 でのユーザーの統一指示に基づいて「ユーザー規則で不可」としました。
- **target を見た調整の範囲**：test/target を見て選んだのは study_044（全 104,750 target trace で採点）と study_058（元の test 集合）だけです。C3 の窓自体も study_041 で target を見て選ばれています（設計:240）。study_057 と study_059 の「勝ち」は、調整していない固定構成が、test で調整した他手法を上回ったものです。study_058 の probe は regsi_grid で行われました。study_061 は ReGSI を実行しておらず、060 の結果と対比しただけです。
- **記録の欠落と出典の注意**：TF32 は manifest に記録がないため、凍結 source のコード（`[SRC]/training/forge_mvp_methods.py:312-313`）から判断しました。import した S1 cell（C3 の a01_b01・a01_b05、FORGE の a01_b01）には would_clip_fraction がありません。ただし clip と would-clip はどちらも閾値 1.0 なので、記録がある run ではすべて一致します。FORGE の seed の manifest では、parameter_count は diagnostics にだけあります。dilation は time_downsample_factor 2 で間引いた latent 上で効くため、d=6 の tap 間隔は C3 で 96 ms、FORGE で 48 ms です。設計:230（`[S067]/README.md:289`、`[S067]/decisions.md:387` も同じ文）の 48/24 ms は、この 2 倍を含んでいません。

### 付録 A-2. CCNet5D（ccnet5d_grid）

出典の略記（いずれも repository root からの相対 path です）
- `C3run/` = `runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces/`、`FGrun/` = `runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge/`、`C1run/` = `runs/study_067_forge_c3_common_grid/20261022T202519313159Z_05c92e79ce80_traces/`（C-1 tk3。C3 S1 `a05_b02` の import 元）、`cc/` = `method_runs/ccnet5d_grid/`
- `cfg` = `C3run/cc/S1/a05_b02/config.resolved.yaml` と `FGrun/cc/S1/a02_b05/config.resolved.yaml`（35 行目までは両方で同じ行番号）
- `C1m` = `C1run/cc/S1/a05_b02/manifest.json`、`FGm` = `FGrun/cc/S1/a02_b05/manifest.json`、`C3seed` = `C3run/cc/evaluation/seed_732{1,2,3}/manifest.json`、`FGseed` = `FGrun/cc/evaluation/seed_732{1,2,3}/manifest.json`
- `S67/` = `studies/study_067_forge_c3_common_grid/`、`設計` = `reports/study_067_design_round2.md`、`感度` = `results/study_067_forge_c3_common_grid/20261103T233134Z_ba805bd26f6a_sensitivity/summary.md`

#### Protocol

| 項目 | C3 | FORGE | 出典 |
|---|---|---|---|
| optimizer/algorithm | AdamW、学習率一定 | AdamW、学習率一定 | `cfg:15,20` |
| 学習率 | 1e-3 | 1e-3 | `cfg:16` |
| weight decay | 0 | 0 | `cfg:18` |
| 更新数/反復数 | 50,000（実測 50,000）。最終 EMA 固定、early stopping なし | 50,000（実測 50,000）。同左 | `cfg:5,17`；`C1m:104`；`FGm:112`；`設計:249` |
| 1更新の loss trace 数 | 64（実測 64.0）。`masked_trace_mse`、384 sample 全長 | 64（実測 64.0）。2,172 sample の整列 canvas 全体（有効 46%） | `cfg:4,27`；`C1m:50123`；`FGm:50131`；`設計:231` |
| inner mask（名目/実効） | 0.25 / S1 0.249999（0.249826–0.250177）、seed 0.250001・0.250000・0.250000 | 0.25 / S1 0.250015（0.249110–0.250896）、seed 0.250005・0.250004・0.249990 | `cfg:35`；`C1m:50124`；`C3seed:50175`；`FGm:50132`；`FGseed:50223`；`S67/README.md:274` |
| データの再利用・勾配の鮮度 | 毎更新 fresh patch 1枚、forward/backward 1回。隠した trace から 64本を教師。可視観測 cell 平均 2,224.8 | 同じ手順（Q1=A）。可視観測 cell 平均 466.5 | `C3run/cc/S1/a05_b02/config.resolved.yaml:44-45`；`FGrun/cc/S1/a02_b05/config.resolved.yaml:46-47`；`C1m:116,50121-50122`；`FGm:124,50129-50130`；`S67/README.md:75`；`S67/decisions.md:98` |
| clip（発動率）または would-clip 率 | clip なし。would-clip（閾値 1.0）S1 0.00486、seed 0.00436・0.0056・0.00492（4値平均 0.004935）。S1 全25 cell 0.00386–0.00752 | clip なし。would-clip S1 0、seed 4e-05・0・2e-05（4値平均 1.5e-05）。S1 全25 cell 0–0.00356 | `C1m:103`；`C3seed:154`；`FGm:111`；`FGseed:202`；`C3run/cc/S1/*/manifest.json`、`FGrun/cc/S1/*/manifest.json`（import の各1 cell は `imported_from` 先の値）；`設計:228` |
| EMA（primary と shadow） | 主 0.999（`fixed_final_ema`）。shadow 0.999・0.9995・0.9998 は診断専用 | 同じ | `cfg:5,22-26`；`C1m:100131`；`FGm:100144`；`S67/decisions.md:100`；`設計:42-43` |
| 精度 | FP32、TF32 有効。cuDNN benchmark on（非決定的） | 学習・予測とも bf16（CUDA forward のみ）、TF32 有効。cuDNN benchmark on。S-b：FP32 − bf16 = −0.006 dB | `C3run/cc/S1/a05_b02/config.resolved.yaml:21,46`；`FGrun/cc/S1/a02_b05/config.resolved.yaml:21,43,49`；`FGrun/source/src/seis_interp/training/ccnet5d_observed_training.py:82-84`；`FGrun/source/src/seis_interp/training/forge_mvp_methods.py:312-313`；`設計:60`；`感度` |
| パラメータ数 | 363,292（363,269 比 +23、+0.0063%）。hidden 53 / intermediate 63、kernel 3、time kernel 3 | 363,292（同じ構造。initialization_hash 3da49b70… が同一） | `cfg:8-11`；`C1m:53,56`；`FGm:98,101`；`studies/study_043_c3_v3_ccnet_target_tuning/README.md:54-56` |
| model seed | 7301。placement・mask・supervision は `SeedSequence(7301).spawn(3)`。partition は S1 620702、評価 7321–7323 | 7301、同じ3系列。partition は S1 600702、評価 7321–7323 | `cfg:6`；`C1m:5,100126`；`FGm:5,100134`；`FGrun/source/src/seis_interp/training/forge_mvp_methods.py:494-496` |
| dataset 固有の固定差 | 384 sample・8 ms、整列なし、`observed_global_rms`。予測 core [64,8,16,8,16]（FP32） | patch 時間長 2172、学習・予測 bf16、予測 core [2172,4,8,4,8]、`full_trace_prediction`、`saved_tensors_on_cpu`（false）。4 ms、500 m/s 整列（max delay 1171）、`trace_oracle` | `S67/dataset_differences.yaml:3-8,68-81`；`C3run/cc/S1/a05_b02/config.resolved.yaml:36-42`；`FGrun/cc/S1/a02_b05/config.resolved.yaml:36-52` |
| grid | 共通 5×5（patch × time dilation、sample 単位）。選択 a05_b02（[8,16,8,16]、d=2、patch 軸の上端） | 同じ grid。選択 a02_b05（[4,12,4,12]、d=6、dilation 軸の上端 = 24 ms） | `S67/grid.yaml:78-125`；`C3run/workers/ccnet5d_grid_selection.json:2-6`；`FGrun/workers/ccnet5d_grid_selection.json:2-6`；`設計:230` |

#### 調整の履歴

| study | run 数 | 調整した内容 | 選択に使った指標 | 出典 |
|---|---|---|---|---|
| study_043（C3） | 6 | inner mask 0.8→0.1、更新 5k→20k→50k、EMA 0.999、AdamW＋64 trace、幅 53/63。target SNR 13.18→17.06 dB | test/target（評価 target を見て選択） | `studies/study_043_c3_v3_ccnet_target_tuning/config.yaml:14`；同 `README.md:3-5,14-22,27-33,51-56,73-92`；`runs/study_043_c3_v3_ccnet_target_tuning/*/ccnet5d/metrics.json`；`studies/study_042_c3_v3_five_method_comparison/ccnet5d.yaml:45` |
| study_057（FORGE 2 ms） | 7（うち中断1）＋記録なしの予備 fit | inner mask 0.8→0.1、kernel 5/7、patch [2,4,2,4]→[4,8,4,8]、EMA 0.999、500 m/s 整列、tk9 d2。採用 0.8175（最小は tk9 の 0.7198） | test/target（評価 target を見て選択） | `reports/forge_057_ccnet_tuning.md:28-39,48,55,59-74` |
| study_058（FORGE 2 ms） | 18 run dir（16 候補名。文書に総数なし） | tk 3/5/9/11 と dilation、幅（param 合わせ）、patch [1,8,1,8]・[1,16,1,16]・[2,8,2,8]・[2,4,2,4]・[2,12,2,12]・[3,8,4,8]、mask 0.1–0.3、EMA 0.999/0.9995/0.9998。選択 tk9 d2 c36・[1,8,1,8]@0.2（0.9479、未達） | test/target（評価 target を見て選択） | `studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:28-36,75-87`；同 `README.md:41`；`runs/study_058_forge_2ms_random80_align500_matched20k/*_ccnet5d_grid_matched_tuning/`（一覧）；同 `20260923T020320772981Z_25c5193588b2_ccnet5d_grid_matched_tuning/search.yaml:5-7` |
| study_059（FORGE 2 ms） | 4（中断2・重複停止1・本番1） | patch [2001,4,8,4,8]・mask 0.25 へ（構造上の理由）。tk9 d2 c36 は維持。結果 0.7140 | test/target（評価 target を見て選択） | `studies/study_059_forge_2ms_random70_align500_matched20k/README.md:14,40`；同 `decisions.md:34-58` |
| study_060（FORGE 2 ms） | 23（S1–E 16・R 4・最終 3）、構成 48 | S1 patch×mask×dilation、S2 tk3 c58・kernel 5・mask 0.30、S3 wd、E 延長。勝者 [6,8,8,8]@0.10・tk9・d4・c36・wd 0・EMA 0.9998 | validation のみ | `reports/forge_060_experiment_and_search_conditions.md:230-269,356-365,373,412`；`runs/study_060_forge_fair_grid/fair_grid/ccnet5d_grid/final/stage.json:303` |
| study_062（C3） | 14（25 cell 中。ユーザーが停止） | patch × time dilation の 5×5（現行の軸の起源）を lr 1e-4・relMSE・mask 0.1 で部分実行 | 選択なし | `studies/study_062_c3_region_grid/README.md:3,33-38`；同 `grid.yaml:45-101`；同 `ccnet5d_grid.yaml:4,16,31`；`runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/method_runs/ccnet5d_grid/S1/` |
| study_063（C3） | 13（停止） | 同じ grid を lr 1e-4・relMSE で実行。study_064 に置き換え、選択には混ぜない | 選択なし | `studies/study_063_c3_trace_grid/README.md:3,8`；同 `ccnet5d_grid.yaml:4,16`；`studies/study_064_c3_reference_grid/README.md:25` |
| study_064（C3） | 28（S1 25＋seed 3） | 5×5 を mask 0.1・tk3・53/63 で実行。ユーザーが MSE・AdamW・lr 1e-3・wd 0・50k・64 trace・EMA 0.999 に統一。勝者 a05_b02（val 19.073 dB） | validation のみ | `studies/study_064_c3_reference_grid/README.md:9-25`；同 `decisions.md:15-18`；`runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/workers/ccnet5d_grid_selection.json:3-4,181` |
| study_065（FORGE 4 ms） | 1（1,100 更新で停止、採点なし） | study_064 の C3 勝者（[1001,8,16,8,16]、d2、mask 0.1）を移植。結果なし | 選択なし | `studies/study_065_forge_expanded_grid/README.md:3`；同 `decisions.md:84-92`；同 `ccnet5d_grid.yaml:12,25-31`；`runs/study_065_forge_expanded_grid/20261020T150430809810Z_05c92e79ce80/workers/ccnet5d_grid.json` |
| study_066（FORGE 4 ms） | 5（初回＋再起動4）、採点なし | ユーザー指定で bf16、cached 勾配、patch [6,8,8,8]→[4,8,4,8]、予測 core [4,8,4,8]。[4,8,4,8] は予測で OOM、再学習は 27,890 更新で停止 | 選択なし | `studies/study_066_forge_c3_reference/decisions.md:19-98`；同 `README.md:3,40-58`；`設計:30`；`runs/study_066_forge_c3_reference/`（CCNet run 一覧）；同 `20261021T053559200620Z_05c92e79ce80_ccnet_patch_4_8_4_8/workers/ccnet5d_grid.json`（OOM）；同 `20261021T231630621271Z_05c92e79ce80_ccnet_bf16_spatial_inference_restart/ccnet_user_stop_20261022.json`（27,890） |
| study_067 C-0（FORGE） | probe 3＋end-to-end 1 | cuDNN benchmark（1.34倍 ≥ 1.3 の速度規則で両 dataset に採用）、メモリ予算 17/17/17/30/64 GiB を維持、G0 合格。score は不使用 | 選択なし | `設計:127`；`S67/decisions.md:134-157` |
| study_067 C-1（C3＋FORGE） | 4 | tk3 と tk9（36/37、363,227）を同じ param で比較。T1：C3 −0.081、FORGE +0.069 dB で不成立のため tk3 を維持。tk3 の2 run を S1 に import | validation のみ | `S67/decisions.md:226-249`；`S67/round2_outcomes.json`（key `c1`）；`設計:128` |
| study_067 本番 S1（C3＋FORGE） | 各 25（新規 24＋import 1）＋評価 seed 各 3 | 共通 5×5 から validation relMSE 最小：C3 a05_b02、FORGE a02_b05 | validation のみ | `S67/README.md:65,71`；`C3run/workers/ccnet5d_grid_selection.json:2-6,191`；`FGrun/workers/ccnet5d_grid_selection.json:2-6,194` |
| study_067 S-a / S-b（11-01、launch 後） | 5（FORGE S1 2＋C3 S-a seed 3） | 報告のみで値は変えていません。FORGE a03_b02 の validation：mask 0.1 は 1.478 dB（0.25 は 1.592）、FP32 は 1.587 dB（bf16 は 1.592） | 選択なし | `S67/decisions.md:3-27`；`感度`；`runs/study_067_forge_c3_common_grid/20261101T204948612373Z_ba805bd26f6a_forge/`、`…/20261101T204953383282Z_ba805bd26f6a_forge/`、`…/20261101T205003059027Z_ba805bd26f6a_traces/` |

#### 過去の勝ち設定の扱い

| study | 勝った設定 | dataset | 扱い | 根拠 | 出典 |
|---|---|---|---|---|---|
| study_057 | 500 m/s 幾何整列 canvas | FORGE | 現行に含まれる※ | FORGE の共通前処理（max delay 1171） | `reports/forge_057_ccnet_tuning.md:38,41-46`；`S67/dataset_differences.yaml:7-8`；`FGrun/cc/S1/a02_b05/config.resolved.yaml:50-52` |
| study_057・059 | patch [4,8,4,8] | FORGE | 現行に含まれる※ | grid a01（両 dataset）。FORGE の選択は a02 | `reports/forge_057_ccnet_tuning.md:33,70`；`studies/study_059_forge_2ms_random70_align500_matched20k/decisions.md:39`；`S67/grid.yaml:80-85` |
| study_057 | 最終 EMA 0.999 | FORGE | 現行に含まれる※ | 主 EMA 0.999（両 dataset） | `reports/forge_057_ccnet_tuning.md:37`；`cfg:22` |
| study_057–060 | time kernel 9（dilation 2/4、幅 32/32・36/36） | FORGE | 第2弾で試した | C-1（36/37 幅）：FORGE +0.069 dB < +0.2 で T1 不成立、tk3 を維持 | `reports/forge_057_ccnet_tuning.md:48,71`；`runs/study_057_forge_2ms_random70_trace_rms/20260922T143700496117Z_25c5193588b2_ccnet_tuning/align500_tk9d2_p4848_inner10_ema_6k/resolved_config.yaml:10-11`；`studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:30`；`runs/study_060_forge_fair_grid/fair_grid/ccnet5d_grid/final/stage.json`；`設計:128`；`S67/decisions.md:226-237` |
| study_057–060 | loss relMSE、Adam、更新 10k（057）/ 20k（058–060） | FORGE | ユーザー規則で不可 | study_064 でユーザーが MSE・AdamW・50k に統一し、study_067 で固定 | `studies/study_057_forge_2ms_random70_trace_rms/ccnet5d_grid_tuned.yaml:6,15,17`；`studies/study_058_forge_2ms_random80_align500_matched20k/ccnet5d_grid.yaml:6,17,19`；`studies/study_059_forge_2ms_random70_align500_matched20k/ccnet5d_grid.yaml:6,17,19`；`studies/study_060_forge_fair_grid/ccnet5d_grid.yaml:6,17,19`；`studies/study_064_c3_reference_grid/decisions.md:15-17`；`設計:39-41` |
| study_057 | 幅 32/32（111,969 params） | FORGE | ユーザー規則で不可 | params を 363,269 ±5% に揃える規則（study_058 のユーザー指定） | `reports/forge_057_ccnet_tuning.md:32`；`studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:7-8,28-31`；`設計:229` |
| study_057・060 | 予測 core [256,6,32,8,32] | FORGE | ユーザー規則で不可 | study_066 でユーザーが時間分割を禁止し、full-time・空間 [4,8,4,8] を指定 | `reports/forge_057_ccnet_tuning.md:39`；`reports/forge_060_experiment_and_search_conditions.md:267`；`studies/study_066_forge_c3_reference/decisions.md:28,88-91`；`設計:59,61` |
| study_057・060 | inner mask 0.1 | FORGE | ユーザー規則で不可 | CCNet の mask は ReGSI に合わせて 0.25。S-a では 0.1 が −0.114 dB | `reports/forge_057_ccnet_tuning.md:34`；`reports/forge_060_experiment_and_search_conditions.md:265-267`；`設計:13,58`；`感度` |
| study_058 | patch [1,8,1,8]・inner 0.2（64本まで積み重ね） | FORGE | ユーザー規則で不可 | 共通 grid の外（grid は固定）。059 で端タップが未学習と判明（登録規則ではない） | `studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:75-87`；同 `README.md:41`；`studies/study_059_forge_2ms_random70_align500_matched20k/decisions.md:43-49`；`設計:8` |
| study_059 | inner mask 0.25 | FORGE | 現行に含まれる※ | 現行値（両 dataset）。採用の理由はユーザー規則 | `studies/study_059_forge_2ms_random70_align500_matched20k/decisions.md:39`；`cfg:35`；`設計:13` |
| study_060 | patch [6,8,8,8]（E の勝者） | FORGE | ユーザー規則で不可 | grid 軸は現状維持・両 dataset 共通。066 の追加予定は入っていない | `reports/forge_060_experiment_and_search_conditions.md:265-267`；`studies/study_066_forge_c3_reference/decisions.md:58-65`；`設計:8`；`S67/grid.yaml:78-109` |
| study_060 | time dilation 4 | FORGE | 現行に含まれる※ | grid b04。FORGE の選択は b05（6） | `reports/forge_060_experiment_and_search_conditions.md:267`；`S67/grid.yaml:110-125` |
| study_060 | EMA 0.9998 | FORGE | ユーザー規則で不可 | 主 EMA は 0.999 固定。0.9998 は shadow（診断）のみ | `reports/forge_060_experiment_and_search_conditions.md:265-267`；`studies/study_064_c3_reference_grid/decisions.md:15-17`；`設計:420-421`；`S67/decisions.md:239-246` |
| study_060 | weight decay 0 | FORGE | 現行に含まれる※ | 両 dataset で wd 0 | `reports/forge_060_experiment_and_search_conditions.md:267`；`cfg:18` |
| study_064 | cell a05_b02（[8,16,8,16]、d2） | C3 | 現行に含まれる※ | 共通 grid の cell。study_067 でも relMSE で再選択 | `runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/workers/ccnet5d_grid_selection.json:3-4`；`C3run/workers/ccnet5d_grid_selection.json:4` |
| study_064 | inner mask 0.1 | C3 | ユーザー規則で不可 | 0.25 に置き換え。0.1 は +0.270 dB（≥ 0.2）のため、co-primary の感度で報告 | `設計:13,58,136-137`；`runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/method_runs/ccnet5d_grid/S1/a05_b02/manifest.json:15,39`；`感度` |
| study_064 | tk3・53/63・MSE・AdamW lr 1e-3・wd 0・50k・64 trace・EMA 0.999・FP32 | C3 | 現行に含まれる※ | 現行の固定値（mask 以外は同じ。FORGE だけ bf16） | `studies/study_064_c3_reference_grid/ccnet5d_grid.yaml:4-23`；`S67/c3/ccnet5d_grid.yaml:4-24` |
| study_066 | （勝ちなし）ユーザー指定の bf16・予測 core [4,8,4,8] | FORGE | 現行に含まれる※ | FORGE の固定値。S-b は −0.006 dB で救済の条件に当たらない | `studies/study_066_forge_c3_reference/decisions.md:34-46,88-91`；`設計:60-61,139-140`；`感度` |
| study_066 | （勝ちなし）ユーザー指定の cached 勾配・reuse_patch | FORGE | ユーザー規則で不可 | Q1=A で C3 と同じ fresh 手順に統一 | `studies/study_066_forge_c3_reference/decisions.md:34-41`；`S67/decisions.md:98` |

※ 「現行に含まれる」は、登録した3ラベル（第2弾で試した／ユーザー規則で不可／証拠により不採用）の外にあるラベルです。現行の固定値、または変更していない共通 grid の cell に一致する設定に付けました。

#### 補足

- C3 S1 の `a05_b02` は、C-1 tk3 pilot の import です（`C3run/cc/S1/a05_b02/manifest.json:54-55`。prediction.npy の SHA256 4d1994f4… が一致）。本番側の manifest には学習診断がありません。このため、C3 S1 の would-clip・実効 mask・更新数は import 元 `C1m` の値で、学習 code も C-1 の source snapshot です（TF32 は `C1run/source/src/seis_interp/training/forge_mvp_methods.py:304-305`、乱数の分割は :475）。評価 seed と FORGE の選択 cell `a02_b05` は、本番の snapshot（:312-313、:494-496）で動いています。FORGE S1 の `a03_b02` も C-1 からの import です。C3 の seed config は S1 と byte 一致で、FORGE は `device` だけが異なります。
- would-clip は、clip 前の大域勾配 L2 ノルムが 1.0 を超えた更新の割合です（`C3run/source/src/seis_interp/training/neural_diagnostics.py:45-57`）。CCNet 自体は clip しません。run ごとの TF32 flag は manifest になく、「TF32 有効」は frozen source と `S67/README.md:274` に基づきます。C3 の最初の更新の kernel 一覧には tf32 kernel があります（`C1m:57`）。FORGE seed の manifest には top-level の `parameter_count` がなく、`diagnostics.parameter_count`（363,292）だけがあります。
- inner mask 0.25 は、CCNet で調整した値ではありません。ReGSI に合わせるユーザー規則による値です（`設計:13,50`）。C3 `a05_b02` では、mask 0.1（study_064）が 19.073 dB、0.25 が 18.803 dB で、差 +0.270 dB が S-a の登録条件を満たしました。このため ReGSI − CCNet@0.1 を co-primary の感度として報告しています（本文第6節）。ただし study_064 は乱数 stream の修正前なので、この対は乱数経路も異なります（`S67/decisions.md:9`）。S-a の C3 3 seed は test を採点しましたが、感度の対比に使っただけで、選択には使っていません。
- 選択指標の区分について説明します。043・057・058・059 は、test/target での選択を文書が明記しています（059 は `README.md:14` で「test 集合の主指標を参考」）。060 は validation V で選びましたが、中心設定は 058/059 の test 選択を引き継いでいます（`reports/forge_060_experiment_and_search_conditions.md:412`）。064 と study_067 の C-1・S1 は validation のみです（selection JSON は `test_used_for_selection: false`、C-1 は `test_scoring_calls` 0）。C3 の窓自体も、study_041 で target を見て選ばれています（`設計:240`）。
- run 数について：058 と 059 は文書に総数がないため、run directory から数えました（058 は `c_base` と `c_p2828_i02_ema` がそれぞれ2回あります）。057 の記録しない予備 fit（`reports/forge_057_ccnet_tuning.md:74`）と、060 の OOM・中断による再試行は数えていません。062/063 の部分 score が study_064 の MSE・lr 1e-3 への切り替えに影響したかどうかは、文書にありません。062 の軸値の由来も文書にありません（study_062 には decisions.md がありません）。
- 061（NeRSI のみ）、062・063（S1 途中で停止、勝者なし）、065（CCNet 停止）は、CCNet の勝ち設定がないため表に載せていません。066 は完走した run がないため、ユーザー指定の設定だけを載せました。「扱い」の対応付けはこの監査によるものです。予測 core [256,6,32,8,32] は、検証ではメモリ由来の「該当なし」とされました。しかし登録ラベルに「該当なし」がないため、study_066 のユーザー指定を根拠に「ユーザー規則で不可」としました。study_066 では、ユーザーが時間方向の分割を明示的に禁止しています（`studies/study_066_forge_c3_reference/decisions.md:28`）。

### 付録 A-3. NeRSI（profile-wise）

選択した cell は、C3 が `a02_b03`（λ 0.001、β 1.1）、FORGE が `a04_b02`（λ 0.03、β 1.075）です。表の出典では次の略記を使います。

- 〈C3〉＝`runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces/`
- 〈FG〉＝`runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge/`
- 〈cfg〉＝選択 cell の `method_runs/nersi_profile/S1/<cell>/config.resolved.yaml`（C3 `a02_b03`、FORGE `a04_b02`。行番号は両方で同じ）
- 〈mC3〉・〈mFG〉＝同じ cell の S1 `manifest.json`。〈eC3〉・〈eFG〉＝`method_runs/nersi_profile/evaluation/seed_732{1,2,3}/manifest.json`
- 〈snap〉＝〈FG〉`source/src/seis_interp/`（C3 run の snapshot と同じ内容）
- 〈067〉＝`studies/study_067_forge_c3_common_grid/`。〈design〉＝`reports/study_067_design_round2.md`

#### Protocol

| 項目 | C3 | FORGE | 出典 |
|---|---|---|---|
| optimizer/algorithm | AdamW（profile-wise の座標ネット） | AdamW | 〈cfg〉:29 |
| 学習率 | 1e-3（一定） | 1e-3（一定） | 〈cfg〉:30, 34 |
| weight decay | 0 | 0 | 〈cfg〉:32 |
| 更新数/反復数 | 50,000（S1 と3 seed 共通。seed の config は device 以外 S1 と同一） | 50,000（同） | 〈cfg〉:31；〈mC3〉:81；〈mFG〉:110 |
| 1更新の loss trace 数 | 64。loss は masked_trace_mse＋λ·核ノルム（平均 11.57 断面を decode） | 64（同。平均 7.89 断面を decode） | 〈cfg〉:5, 41-42；〈mC3〉:97；〈mFG〉:126；〈snap〉`training/forge_mvp_methods.py`:969-994 |
| inner mask（名目/実効） | —（該当なし。座標ネットで、入力による条件付けがない） | —（該当なし。support mask は N-1 で不採用） | 〈067〉`README.md`:275, 285；〈067〉`decisions.md`:174 |
| データの再利用・勾配の鮮度 | 観測 23,593 本すべてを当てはめる。毎更新、断面を無作為な順に引いて64本を非復元抽出し、毎回新しい勾配。trace あたり約135.6回。訓練座標に jitter 0.5 | 観測 13,271 本、同じ手順（約241.1回/本）。jitter 0.5 | 〈mC3〉:24；〈mFG〉:23；〈snap〉`processing/forge_nersi_profiles.py`:72-101；〈snap〉`training/forge_mvp_methods.py`:969-1000；〈cfg〉:43 |
| clip（発動率）または would-clip 率 | clip なし。would-clip は S1 0.128%、seed 0.128 / 0.146 / 0.124% | clip なし。would-clip は S1・3 seed とも 0% | 〈mC3〉:80；〈eC3〉:111；〈mFG〉:109；〈eFG〉:168 |
| EMA（primary と shadow） | primary 0.999。shadow 0.999 / 0.9995 / 0.9998 の S1 validation は 12.651 / 12.678 / 12.623 dB | primary 0.999。shadow は 0.371 / 0.377 / 0.389 dB（clean_target） | 〈cfg〉:6, 36-40；〈mC3〉:123, 144, 165；〈mFG〉:179, 229, 279 |
| 精度 | FP32（mixed_precision off）。TF32 有効（code で設定、manifest に記録なし）。cuDNN は決定的 | 同じ。validator は CCNet 以外に off しか許さないため、bf16 は使えない | 〈cfg〉:35；〈snap〉`training/forge_mvp_methods.py`:307-313；〈snap〉`forge_mvp_config.py`:145；〈067〉`decisions.md`:42 |
| パラメータ数 | 363,269 | 341,485（−5.9967%。±5% 帯の外で、宣言済みの canvas 例外） | 〈mC3〉:69；〈mFG〉:98；〈067〉`decisions.md`:180；〈067〉`README.md`:288 |
| model seed | 7301（S1・評価とも。trace seed は S1 620702、評価 7321–7323） | 7301（同。trace seed は S1 600702、評価 7321–7323） | 〈cfg〉:7；〈mC3〉:5-6；〈mFG〉:5-6；〈eC3〉・〈eFG〉:5-6 |
| dataset 固有の固定差 | 8 ms。整列なし。profile [384,32]、s=0。loss は 384 sample 全体（全 sample が有効）。振幅は observed_global_rms。選択は全 validation target の relMSE | 4 ms。500 m/s 整列（最大遅延 1171）。profile [2176,32]、s=3（latent 34×4）。loss は canvas 2,172 sample 全体（有効 46%）。振幅は trace_oracle。選択は clean_target の relMSE | 〈067〉`dataset_differences.yaml`:4-8, 19, 33, 83；〈cfg〉:18-20；FORGE 〈cfg〉:46-50；〈mC3〉:98；〈mFG〉:127；〈067〉`round2_outcomes.json`:2251, 2321, 2345-2350；〈067〉`README.md`:290, 305；〈C3〉・〈FG〉`workers/nersi_profile_selection.json`:2 |
| grid | λ {0, 0.001, 0.01, 0.03, 0.1} × β {1.05, 1.075, 1.1, 1.15, 1.25}。選択は a02_b03（validation 12.651 dB） | 同じ grid。選択は a04_b02（validation 0.371 dB） | 〈067〉`grid.yaml`:45-77；〈C3〉・〈FG〉`workers/nersi_profile_selection.json`:4；〈mC3〉:44；〈mFG〉:67 |

#### 調整の履歴

| study | run 数 | 調整した内容 | 選択に使った指標 | 出典 |
|---|---|---|---|---|
| study_040（C3） | 単一 model 64 run と局所 model 6条件（28 model）。設計は「76」と記載 | 整列、Fourier、幅、kernel、lr、EMA など。EMA 0.999、lr 1e-3 一定、kernel 5 はここで決まった | test/target（評価 target を見て選択） | `studies/study_040_c3_nersi_target_tuning/README.md`:32, 34-35, 40, 95-96, 256；`runs/study_040_c3_nersi_target_tuning/20260912T101025Z_03c2fc83df18_comparison/summary.json`:3；〈design〉:240 |
| study_041（C3） | 1（3窓のうち最初の窓で目標に到達） | C3 の評価窓（shot と receiver-y が [18,50)）。study_062–067 の C3 crop として使用 | test/target（評価 target を見て選択） | `studies/study_041_c3_nersi_translated_window/README.md`:3-4, 19, 43-49；`studies/study_062_c3_region_grid/README.md`:15-16；〈design〉:240 |
| study_046（C3） | 28 | 363,269 params の配分（Fourier 40 / enc 40 / latent 28 / dec [29,14,8]）、β 1.05–1.15、1更新10断面、AdamW | test/target（評価 target を見て選択） | `studies/study_046_c3_v3_nersi_gnn_parameter_budget/README.md`:10-23, 73-78, 108-111；〈design〉:240 |
| study_061（FORGE 2 ms） | 探索20、最終3 seed、参照19（参照は validation のみで採点） | 座標群ごとの Fourier 帯域、容量、wd、EMA。参照 run で deep-time、rank-2、λ、jitter | validation のみ | `studies/study_061_forge_nersi_profile/README.md`:11-12, 64-79, 109-181；`studies/study_061_forge_nersi_profile/decisions.md`:22-23 |
| study_062（C3） | 25中13で停止 | λ×β（lr 1e-4、relMSE）。ユーザーが全 run を停止 | 選択なし | `runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/workers/nersi_profile.json`；`studies/study_062_c3_region_grid/nersi_profile.yaml`:4, 29 |
| study_063（C3） | 25中11で停止。study_064 に置き換え | β 1.01–1.1（lr 1e-4、relMSE）。validation 診断で旧 study_046 と比較（10.98 対 13.89 dB） | 選択なし | `runs/study_063_c3_trace_grid/20261011T145500928121Z_05c92e79ce80_traces/workers/nersi_profile.json`；`runs/study_063_c3_trace_grid/20261012T144258278233Z_05c92e79ce80_regsi_nersi_validation_diagnostic/manifest.json`；`studies/study_064_c3_reference_grid/decisions.md`:3-12 |
| study_064（C3） | 28（S1 25、3 seed） | λ×β grid（study_046 の β 1.15・λ 0 を含む）。ユーザーが relMSE・lr 1e-4 を MSE・lr 1e-3 に戻し、3手法の条件を統一（AdamW、wd 0、EMA 0.999 は前から同じ） | validation のみ | `studies/study_064_c3_reference_grid/decisions.md`:3-17；`studies/study_064_c3_reference_grid/README.md`:9-20；`studies/study_063_c3_trace_grid/nersi_profile.yaml`:4, 28-35 |
| study_065（FORGE 4 ms、拡張） | 1 | 調整なし。study_064 の C3 勝者を移植（deep-time と jitter は除外）。validation −0.645 dB | 選択なし | `studies/study_065_forge_expanded_grid/decisions.md`:84-92；`runs/study_065_forge_expanded_grid/20261020T150430809810Z_05c92e79ce80/summary.csv`:7 |
| study_066（FORGE 4 ms） | 1 | 調整なし。study_064 の C3 勝者を移植。validation −1.598、test −1.632 dB | 選択なし | `studies/study_066_forge_c3_reference/README.md`:9-13；`runs/study_066_forge_c3_reference/20261020T225334465289Z_05c92e79ce80/summary.csv`:7, 9 |
| study_067 P3 | 2（C3 と FORGE の a02_b04） | jitter 0.5 をユーザー決定で両 dataset に固定。C3 12.657 dB（≥ 10.5 を満たす）、FORGE −0.689 dB | validation のみ | 〈067〉`decisions.md`:65-67；`runs/study_067_forge_c3_common_grid/20261022T035444744870Z_05c92e79ce80_p3_c3_nersi_jitter05/method_runs/nersi_profile/S1/a02_b04/manifest.json`:39；`runs/study_067_forge_c3_common_grid/20261022T035444744870Z_05c92e79ce80_p3_forge_nersi_jitter05/method_runs/nersi_profile/manifest.json`:51, 56 |
| study_067 N-1 | 4（FORGE の a02_b04、s=3、jitter × loss 範囲） | s=3 を採用（P3 比 +0.652 dB）。support mask は不採用（−0.065 dB）。jitter は保持（外すと −0.302 dB） | validation のみ | 〈067〉`decisions.md`:159-178；〈067〉`round2_outcomes.json`:475-478；〈design〉:129 |
| study_067 N-2 | 0（N-1 で採用した arm の shadow を使用） | NeRSI だけ EMA 0.9998 を検討。+0.0088 dB < +0.15 dB で不採用。床の救済はここで終了 | validation のみ | 〈067〉`decisions.md`:176；〈067〉`round2_outcomes.json`:451-467；〈design〉:130 |
| study_067 本番 S1 | 各 dataset 25（a02_b04 は pilot から import）と評価3 seed | λ×β の 5×5 から validation relMSE 最小の cell を選択 | validation のみ | 〈C3〉・〈FG〉`workers/nersi_profile_selection.json`:2-4 |

#### 過去の勝ち設定の扱い

| study | 勝った設定 | dataset | 扱い | 根拠 | 出典 |
|---|---|---|---|---|---|
| study_057 | nersi_real の最良参考設定：内部整列 700 m/s、中点/距離/方位座標（base 1.005 / 1.005 / 1.04 / 1.00001）、lr 3e-4、wd 1e-4、24k 更新（test で選択） | FORGE 2 ms | ユーザー規則で不可 | 整列は共通の 500 m/s 幾何整列だけで、手法別の整列は禁止。lr・wd・更新数は共通の固定値 | `reports/forge_057_nersi_tuning.md`:26-37, 46；`studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md`:8, 21-22；〈067〉`dataset_differences.yaml`:8；〈design〉:39-40 |
| study_058 | 中点/半径/方位の Fourier 座標と軸別 base [1.05, 1.05, 1.1, 1.00001]（test で選択、nb3） | FORGE 2 ms | ユーザー規則で不可 | trace-wise 版にしかない座標。profile NeRSI は C3 の構造と、β 1つの grid 軸に固定 | `studies/study_058_forge_2ms_random80_align500_matched20k/nersi_real.yaml`:11-15, 31, 69-72；〈design〉:8, 71 |
| study_058 | EMA 0.9998（test で選択） | FORGE 2 ms | 証拠により不採用 | N-2 で試行：+0.0088 dB < +0.15 dB | `studies/study_058_forge_2ms_random80_align500_matched20k/nersi_real.yaml`:62, 69-72；`studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md`:33-35；〈067〉`decisions.md`:176 |
| study_058 | wd 5e-5（test で選択） | FORGE 2 ms | ユーザー規則で不可 | wd 0 は study_064（C3 のみ）で統一。study_067 の共通固定値の規則で両 dataset に適用 | `studies/study_058_forge_2ms_random80_align500_matched20k/nersi_real.yaml`:57, 69-72；`studies/study_064_c3_reference_grid/README.md`:13-14, 20；〈design〉:13, 39-40 |
| study_059 | 方位 base 1.1、latent 4、encoder 150（369,839 params、test で選択） | FORGE 2 ms | ユーザー規則で不可 | 方位軸は trace-wise 版にしかない。profile NeRSI は C3 の幅に固定 | `studies/study_059_forge_2ms_random70_align500_matched20k/decisions.md`:60-74；〈design〉:71, 81 |
| study_060 | S1 勝者 s1_mid110_az105（中点 base 1.10、方位 1.05。最終 [1.1, 1.1, 1.1, 1.05]） | FORGE 2 ms | ユーザー規則で不可 | trace-wise 版の軸別 base。grid 軸の β（1つ）は現状のまま | `runs/study_060_forge_fair_grid/fair_grid/nersi_real/S2/stage.json`（decision.base.run）；`runs/study_060_forge_fair_grid/fair_grid/nersi_real/final/final_s3_wd7e-05_ema0.9998_seed7311.yaml`:9-13；〈design〉:8, 71 |
| study_060 | EMA 0.9998（S3 勝者、validation で選択、seed 平均 relMSE 0.9223） | FORGE 2 ms | 証拠により不採用 | N-2：+0.0088 dB < +0.15 dB | `runs/study_060_forge_fair_grid/fair_grid/nersi_real/final/stage.json`（decision）；〈067〉`decisions.md`:176 |
| study_060 | wd 7e-5（S3 勝者） | FORGE 2 ms | ユーザー規則で不可 | 共通の wd 0 | `runs/study_060_forge_fair_grid/fair_grid/nersi_real/final/stage.json`（decision.run）；〈design〉:13, 39-40 |
| study_061 | 事前登録の勝者 e_src_flat（Fourier 16、latent 2 / enc 74 / dec [16,8,4]、wd 1e-4、軸別 base。validation 0.9799、最終 test 1.0001） | FORGE 2 ms | ユーザー規則で不可 | 各手法は現行の C3 grid と共通の固定値を両 dataset で使い、微調整はしない | `studies/study_061_forge_nersi_profile/README.md`:75-79, 89；`studies/study_061_forge_nersi_profile/nersi_profile.yaml`:11；〈design〉:8, 13 |
| study_061 | deep-time（時間方向だけの ×2 段、latent 34×4） | FORGE 2 ms | 第2弾で試した | N-1(i)：FORGE で s=3 を採用（P3 比 +0.652 dB）。C3 は s=0 | `studies/study_061_forge_nersi_profile/README.md`:129-148；〈design〉:349；〈067〉`decisions.md`:172, 178 |
| study_061 | 座標 jitter 0.5 セル（3 seed の validation 0.9289） | FORGE 2 ms | 第2弾で試した | P3 でユーザー決定により固定し、N-1(iii) で保持（外すと −0.302 dB） | `studies/study_061_forge_nersi_profile/README.md`:167, 180；〈067〉`decisions.md`:65-67, 174 |
| study_061 | 核ノルム λ 0.01（jitter 0.5 との組合せで最良、validation 0.9231） | FORGE 2 ms | 現行に含まれる※ | λ 0.01 は grid 軸 a の a03 で、第2弾では判断していない。FORGE の選択は λ 0.03 | `studies/study_061_forge_nersi_profile/README.md`:181；〈067〉`grid.yaml`:53-55；〈067〉`decisions.md`:61 |
| study_061 | EMA 0.9998（ほぼ全 run で最良の水準） | FORGE 2 ms | 証拠により不採用 | N-2：shadow 0.9998 − 0.999 = +0.0088 dB < +0.15 dB | `studies/study_061_forge_nersi_profile/README.md`:64-73, 167；〈design〉:353；〈067〉`round2_outcomes.json`:467 |
| study_064 | grid 勝者 a02_b04（λ 0.001、β 1.15。validation 13.505、test 13.098 dB） | C3 | 現行に含まれる※ | 変えていない共通 grid の1 cell（P3 と N-1 の pilot cell）。C3 の S1 は a02_b03 を選択 | `runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/workers/nersi_profile_selection.json`:4；`runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/method_runs/nersi_profile/S1/a02_b04/manifest.json`:39；`runs/study_064_c3_reference_grid/20261012T235609226847Z_05c92e79ce80_traces/workers/nersi_profile_summary.csv`:10；〈067〉`decisions.md`:61；〈design〉:129 |
| study_064 | jitter なし（study_064 の勝者。P3 の jitter 0.5 より validation が高い：13.51 対 12.66 dB） | C3 | 証拠により不採用 | N-1(iii)（FORGE）：なし − あり = −0.302 dB < −0.05 dB。共通の固定値なので C3 だけ外すことはできない | 〈067〉`decisions.md`:67, 174；〈design〉:352 |
| study_067 N-1 | canvas の時間方向 upsampling s=3（validation −0.037 dB、P3 は −0.689） | FORGE 4 ms | 第2弾で試した | 規則 (i) を満たし、両 dataset の YAML に適用（C3 は s=0 で変化なし） | 〈067〉`decisions.md`:172, 178；〈067〉`round2_outcomes.json`:476 |
| study_067 N-2 | EMA 0.9998 の shadow（+0.0088 dB） | FORGE 4 ms | 証拠により不採用 | 登録した閾値 +0.15 dB に届かない | 〈067〉`round2_outcomes.json`:451-467；〈067〉`decisions.md`:176 |

※「現行に含まれる」は、登録した3つのラベル（第2弾で試した／ユーザー規則で不可／証拠により不採用。〈design〉:241、〈067〉`README.md`:300）の外にあるラベルです。その設定が、すでに現行の固定値か、変えていない共通 grid の cell に含まれている場合に使いました。

#### 補足

- **study_040 の run 数**：設計 (5) 11 は NeRSI の C3 調整を「76+28 run」としています（〈design〉:240）。しかし study_040 の記録から再構成できるのは、単一 model 64 run と局所 model 6条件（28 model）だけです（`studies/study_040_c3_nersi_target_tuning/README.md`:32, 40）。この64 run には、study_038・039 の run directory にある11 run（4＋7）も含まれます（`runs/study_040_c3_nersi_target_tuning/20260912T101025Z_03c2fc83df18_comparison/summary.json` の `runs`）。
- **選択指標の区分**：study_061 は手法内の設定を validation だけで選びました。ただし、study を実施すること自体は study_060 の test 結果（NeRSI が最下位）を見た後に決めています（`studies/study_061_forge_nersi_profile/README.md`:11-12）。study_064 の grid には、target を見て選ばれた study_046 の β 1.15 が含まれます。studies 057–060 は trace-wise の `nersi_real` を調整したものなので、勝ち設定の表にだけ載せています。
- **would-clip**：目的関数全体（data loss ＋ λ·核ノルム）の勾配の global L2 norm が 1.0 を超えた更新の割合です。記録するだけで、勾配は変えていません（〈snap〉`training/neural_diagnostics.py`:44-58）。S1 の24 cell（pilot から import した a02_b04 は記録がないため除外）では、C3 が平均 0.39%・最大 1.28%（a04_b01）、FORGE が平均 0.00008%・最大 0.002%（a05_b03）でした。
- **FORGE の床と G0**：S1 の25候補の最良は 0.371 dB（clean_target）で、0.4 dB に届きませんでした。shadow EMA 0.9998 でも 0.389 dB です。床の救済は、登録どおり N-2 で終えました。G0 は選択 cell と3 seed のすべてで不合格でした（更新 901–1000 の loss は 0.999–1.000·L0。〈mFG〉:117；〈eFG〉:176）。座標ネットの立ち上がりが遅いためで、実装バグではないと判断しました（〈067〉`decisions.md`:39-43）。G0 は診断専用で、選択には使っていません（〈067〉`decisions.md`:31）。
- **jitter 0.5**：登録した規則ではなく、ユーザーの決定で入りました（〈067〉`decisions.md`:65。この行には根拠の study が書かれていません）。0.5 という値の先行証拠は、study_061（FORGE 2 ms）で validation だけを採点した参照 run です（`studies/study_061_forge_nersi_profile/README.md`:167, 180）。C3 では約 0.85 dB 下がります（12.66 対 13.51、丸め値。〈067〉`decisions.md`:67）。jitter を残すか外すかを判定した N-1(iii) は、FORGE だけで行いました。
- **勝ち設定の表の範囲**：study_062・063（C3、途中で停止）と study_065・066（FORGE、単一設定の移植）には NeRSI の勝ち設定がないため、表に載せていません。第2弾で試して不採用になった設定には「証拠により不採用」を付け、根拠の列に試行名を書いています。

### 付録 A-4. POCS（`pocs_grid`）

POCS は学習パラメータを持たない古典手法です。学習に関する項目は「—（該当なし）」とし、代わりに対応する固定値（反復数、threshold、窓）を示します。

出典の略記：
- `C3R` = `runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces`
- `FGR` = `runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge`
- `C3S1` = `C3R/method_runs/pocs_grid/S1/a01_b05`、`FGS1` = `FGR/method_runs/pocs_grid/S1/a04_b01`
- `C3EV`・`FGEV` = 各 run の `method_runs/pocs_grid/evaluation/seed_7321`（7322・7323 も同じ行）
- `SRC` = `C3R/source/src/seis_interp`（引用した file は FGR の snapshot と byte 一致）
- `S067` = `studies/study_067_forge_c3_common_grid`、`DES` = `reports/study_067_design_round2.md`、`SEL` = `workers/pocs_grid_selection.json`
- `…_xxx` は、同じ study の run のうち接尾辞 xxx を持つ唯一のディレクトリを指します。

#### Protocol

| 項目 | C3 | FORGE | 出典 |
|---|---|---|---|
| optimizer/algorithm | なし（古典）。窓付き Fourier POCS（hard threshold、毎反復の観測再挿入、正 Hann 合成） | 同左（実装 file は byte 一致） | `SRC/processing/pocs_batched.py:43-62,77-124`、`FGR/source/src/seis_interp/processing/pocs_batched.py`（SHA256 一致） |
| 学習率 | —（該当なし） | —（該当なし） | `C3S1/config.resolved.yaml:1-25`、`FGS1/config.resolved.yaml:1-28`（training 節なし） |
| weight decay | —（該当なし） | —（該当なし） | 同上 |
| 更新数/反復数 | 32 反復 | 32 反復 | `C3S1/config.resolved.yaml:8`、`FGS1/config.resolved.yaml:8` |
| threshold（開始→終了） | 1.0→0.01、等比。窓・周波数ごとの最大振幅に対する比 | 同左 | `C3S1/config.resolved.yaml:9-10`、`FGS1/config.resolved.yaml:9-10`、`SRC/processing/pocs_batched.py:51-55` |
| 固定の受振窓・overlap | 受振窓 [4,8]、overlap [3,6]（75%）。時間軸・震源軸は 50% | 同左（method YAML は byte 一致） | `S067/c3/pocs_grid.yaml:15-16,21-22`、`S067/forge/pocs_grid.yaml`（SHA256 一致）、`S067/decisions.md:69,132,301` |
| 1更新の loss trace 数 | —（該当なし）。`loss` キーは選択指標の名前を指すだけ | —（該当なし） | `C3S1/config.resolved.yaml:2`、`FGS1/config.resolved.yaml:2` |
| inner mask（名目/実効） | —（該当なし） | —（該当なし） | `DES:226` |
| データの再利用・勾配の鮮度 | 勾配なし・決定的。各観測 trace は自分を覆う窓だけで使用（観測 23,593 本、10,725 窓、空窓含む） | 同左（観測 13,271 本、50,050 窓） | `SRC/processing/pocs_batched.py:57,62,101-123`、`C3EV/manifest.json:24,87`、`FGEV/manifest.json:23,144` |
| clip（発動率）または would-clip 率 | —（該当なし、勾配なし）。S1 と3 seed の manifest に key なし | 同左 | `DES:228`、`C3S1/manifest.json`、`C3EV/manifest.json`（3 seed）、`FGS1/manifest.json`、`FGEV/manifest.json`（3 seed） |
| EMA（primary と shadow） | —（該当なし。`checkpoint_rule: none`） | —（該当なし） | `C3S1/config.resolved.yaml:3`、`FGS1/config.resolved.yaml:3` |
| 精度 | CPU の NumPy float64／complex128 FFT。入出力 float32（global RMS で正規化）。TF32 の設定は CPU 経路に効かない | 同じ FFT・精度。入力は trace ごとの RMS で正規化（global_rms = 1.0）し canvas 2,172 に整列。出力は整列を解除し test RMS で復元 | `SRC/processing/pocs_batched.py:19-20,46-47`、`SRC/training/forge_mvp_methods.py:247,256,285-290,312-316,321,336`、`SRC/data/forge_mvp_artifacts.py:150-172`、`FGEV/manifest.json:145-146` |
| パラメータ数 | 0 | 0（manifest に key なし。experiment.json の値） | `C3S1/manifest.json:79`、`C3R/experiment.json:818`、`FGR/experiment.json:74` |
| model seed | null（決定的）。S1 の trace_seed 620702、評価 7321–7323 | null。S1 の trace_seed 600702、評価 7321–7323 | `C3S1/config.resolved.yaml:4`、`C3S1/manifest.json:5-6`、`FGS1/manifest.json:5-6`、`C3EV/manifest.json:5-6`、`FGEV/manifest.json:5-6` |
| dataset 固有の固定差 | 宣言は実行専用 key だけ（device、cpu_threads、workers）。8 ms、observed_global_rms、整列なし | 宣言 key は同じ。入力側で 4 ms、trace_oracle、500 m/s 整列（最大遅延 1,171）、canvas 2,172 | `S067/dataset_differences.yaml:4-8,88-91`、`FGS1/config.resolved.yaml:26-28`、`FGEV/manifest.json:145` |
| grid | 共通 5×5：時間窓 {64,96,128,192,256} × 震源窓 [line, 点] {[2,8],[2,16],[4,8],[4,16],[6,16]}、50% overlap | 同一（sample 単位。FORGE 4 ms で 256–1024 ms、C3 8 ms で 512–2048 ms） | `S067/grid.yaml:204-256`、`DES:230` |
| 選択 cell | relMSE 最小 → a01_b05（64／[6,16]） | clean_target relMSE 最小 → a04_b01（192／[2,8]） | `C3R/SEL:2-4`、`FGR/SEL:2-4` |
| 実行 | batched、CPU、cpu_threads 1、nice 10 | 同左 | `C3S1/config.resolved.yaml:5,23-25`、`C3S1/manifest.json:12`、`FGS1/manifest.json:12` |
| S1 の出所 | 25 cell すべてを P-1 の K1 pilot から import（新規 0） | 同左 | `S067/README.md:67,71,73`、`C3S1/manifest.json:54-59`、`FGS1/manifest.json:83-88` |

#### 調整の履歴

| study | run 数 | 調整した内容 | 選択に使った指標 | 出典 |
|---|---|---|---|---|
| study_036・042（C3） | 0 | 正式 config（100 反復、窓 [128,8,16,8,32]）を固定値として使用。調整の記載なし | 選択なし | `studies/study_036_c3_random80_observed_only_poc/README.md:28`、`studies/study_042_c3_v3_five_method_comparison/README.md:14` |
| study_041（C3 窓） | 1（候補3件の1件目で到達） | C3 の評価窓を target 到達で選択。全手法に共通の窓 | test/target（評価 target を見て選択） | `studies/study_041_c3_nersi_translated_window/README.md:4,19`、`DES:240` |
| study_057（FORGE 2 ms、欠損70%） | 82 候補評価 | 窓・overlap・反復・threshold。採用は 4 反復、end 0.4 | test/target（評価 target を見て選択） | `reports/forge_057_pocs_tuning.md:21-27,33,42`、`studies/study_057_forge_2ms_random70_trace_rms/decisions.md:23-27` |
| study_058（FORGE 2 ms、欠損80%） | 7 候補 | 反復・end・時間窓・空間窓。勝者 p_it8_end02（8 反復、end 0.2） | test/target（評価 target を見て選択） | `studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:33-37`、`runs/study_058_forge_2ms_random80_align500_matched20k/…_pocs_grid_matched_tuning/search.yaml:7-20`、同 `search_results.csv:2-8` |
| study_059（FORGE 2 ms、欠損70%） | 0 | study_058 の test 選択設定をそのまま継続 | 選択なし | `studies/study_059_forge_2ms_random70_align500_matched20k/pocs_grid.yaml:9-28` |
| study_060（FORGE 2 ms） | 16 + 最終 1 | 反復 {8,16} × end {0.1,0.2} × 時間窓 {512,1024} × 空間窓 2種。勝者 16 反復・end 0.2 | validation のみ | `runs/study_060_forge_fair_grid/fair_grid/pocs_grid/final/stage.json:211-214`、`reports/forge_060_experiment_and_search_conditions.md:319-330,361` |
| study_062（C3 region） | 25（study はユーザー指示で停止） | 時間窓 {64..256}（現行 axis_a の起源）× 震源窓 {[2,8],[4,8],[4,16],[8,16],[16,16]}。100 反復、受振 [8,16] | validation のみ | `studies/study_062_c3_region_grid/grid.yaml:136-188`、同 `README.md:3`、`runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/SEL:2-31` |
| study_063（C3 trace、主） | 25 + 3 seed（test 採点） | study_062 と同じ軸、100 反復。勝者 a01_b03（14.416 dB） | validation のみ | `runs/study_063_c3_trace_grid/…_traces/SEL:2-31`、同 `method_runs/pocs_grid/S1/a01_b03/manifest.json:39`、`studies/study_063_c3_trace_grid/README.md:23-26` |
| study_063（forge-matched） | 50（16 反復 25 + 32 反復 25） | ユーザー指定で震源窓を現行 axis_b に揃え、次に 16→32 反復。勝者 12.116／14.470 dB | validation のみ | `studies/study_063_c3_trace_grid/decisions.md:21-33`、同 `pocs_forge_matched.yaml:9-15`、`runs/study_063_c3_trace_grid/…_pocs_forge_matched_iterations16_grid/SEL:4,80`、`…_pocs_forge_matched_iterations32_grid/SEL:4,71` |
| study_065（FORGE 4 ms） | 1（移植） | 調整なし。study_063 の C3 勝者を移植（validation −0.713 dB） | 選択なし | `studies/study_065_forge_expanded_grid/decisions.md:88-90`、`runs/study_065_forge_expanded_grid/20261020T150430809810Z_05c92e79ce80/summary.csv:11` |
| study_066（FORGE 4 ms） | 75（移植 1 + 新規 24 + 25 + 25） | 移植（−1.218 dB）の後、承認済み震源窓5種（現行 axis_b の起源）で 100 反復、16 反復、16 反復・end 0.2 の3 grid | validation のみ | `runs/study_066_forge_c3_reference/20261020T225334465289Z_05c92e79ce80/summary.csv:11`、`runs/study_066_forge_c3_reference/…_pocs_drr_grid/SEL:42-48`、同 `search_plan.json:21-42`、`…_pocs_iterations16_grid/search_plan.json:47-53`、`…_pocs_iterations16_end02_grid/search_plan.json:47-53` |
| study_067 P1 | 75 新規 + 既存の 16／32 反復 run | 反復 {8,16,32}（受振 [8,16]）。規則は FORGE ≥ 0.5 かつ C3 ≥ 10.5 dB で、該当なし | validation のみ | `S067/decisions.md:69,71`、`runs/study_067_forge_c3_common_grid/…_p1_{c3_pocs_iterations8,forge_pocs_iterations8,forge_pocs_iterations32}/SEL` |
| study_067 P1b | 100（4 grid × 25） | 16 反復で受振 [4,8]・[2,8] を [8,16] と比較。[4,8] を採用 | validation のみ | `S067/decisions.md:71`、`runs/study_067_forge_c3_common_grid/…_p1b_{c3,forge}_pocs_iterations16_rx{2x8,4x8}/SEL` |
| study_067 設計時 screening | 約15案（run 数は repo になし。設計書の74 run は DRR を含む） | 受振窓、受振 overlap 75%、反復、end 0.001、low-pass など。最上位が K1（2 cell で C3 13.79／13.76、FORGE 1.081／1.166 dB） | validation のみ | `DES:1,85-91,131`、`S067/decisions.md:132` |
| study_067 第2弾 P-1 | 50（5×5 × 2 dataset） | K1 と現行（16 反復、[4,8]／[2,4]）を T2 で比較し、K1 を採用 | validation のみ | `S067/decisions.md:132`、`DES:85,131` |
| study_067 本番 S1 | 新規 0（P-1 の50 run を import） | 5×5 の cell 選択だけ（C3 a01_b05、FORGE a04_b01） | validation のみ | `S067/README.md:67,71`、`C3R/SEL:2-4`、`FGR/SEL:2-4` |

#### 過去の勝ち設定の扱い

| study | 勝った設定 | dataset | 扱い | 根拠 | 出典 |
|---|---|---|---|---|---|
| study_057 | 4 反復、end 0.4、窓 [512,1,8,1,8]／[256,0,6,0,6]（test で選択、relMSE 0.8381） | FORGE 2 ms | ユーザー規則で不可 | 窓が共通 grid の軸の外（軸は現状維持）。4 反復・end 0.4 は study_067 で未判定 | `reports/forge_057_pocs_tuning.md:8,21-27,42`、`DES:8,13` |
| study_058（study_059 でも継続） | p_it8_end02：8 反復、end 0.2、窓は study_057 と同じ（relMSE 0.8583） | FORGE 2 ms | 証拠により不採用 | 8 反復は P1 で FORGE 0.402 < 0.5。end 0.2 は 0.321 対 0.440。窓は grid の外 | `runs/study_058_forge_2ms_random80_align500_matched20k/…_pocs_grid_matched_tuning/search_results.csv:7`、同 `search.yaml:17-18`、`S067/decisions.md:71`、`DES:88` |
| study_060 | 16 反復、end 0.2、窓 [512,2,8,2,8]／[256,1,6,1,6]（V 0.7452） | FORGE 2 ms | 証拠により不採用 | end 0.2 は不採用リスト。16 反復は P-1 の T2 で 32 に置換。窓は grid の外 | `runs/study_060_forge_fair_grid/fair_grid/pocs_grid/final/stage.json:211-214`、`reports/forge_060_experiment_and_search_conditions.md:330`、`DES:88`、`S067/decisions.md:132` |
| study_062 | 100 反復、受振 [8,16]／[4,8]、a01_b02（64／[4,8]）、7.93 dB | C3 | 証拠により不採用 | FORGE の 100 反復は best −0.372 dB（16 反復は 0.440）。受振は P1b で置換。16 反復への切替はユーザー指示 | `runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/SEL:2-31`、同 `method_runs/pocs_grid/S1/a01_b02/manifest.json:39`、`runs/study_066_forge_c3_reference/…_pocs_drr_grid/SEL:181`、`…_pocs_iterations16_grid/SEL:181`、`studies/study_063_c3_trace_grid/decisions.md:23-25`、`S067/decisions.md:71` |
| study_063（主） | 100 反復、受振 [8,16]／[4,8]、a01_b03（64／[4,16]）、14.416 dB | C3 | 証拠により不採用 | study_062 と同じ。FORGE への移植は −0.713（065）、−1.218 dB（066） | `runs/study_063_c3_trace_grid/…_traces/SEL:2-31`、同 `method_runs/pocs_grid/S1/a01_b03/manifest.json:39`、`runs/study_065_forge_expanded_grid/20261020T150430809810Z_05c92e79ce80/summary.csv:11`、`runs/study_066_forge_c3_reference/20261020T225334465289Z_05c92e79ce80/summary.csv:11`、`S067/decisions.md:71` |
| study_063（forge-matched 16 反復） | 16 反復、受振 [8,16]／[4,8]、a01_b05、12.116 dB | C3 | 証拠により不採用 | P1 規則が不成立（FORGE 16 反復 0.440 < 0.5）。その後 P1b、K1 で置換 | `runs/study_063_c3_trace_grid/…_pocs_forge_matched_iterations16_grid/SEL:4-31,80`、`S067/decisions.md:71` |
| study_063（forge-matched 32 反復） | 32 反復、受振 [8,16]／[4,8]、a01_b04、14.470 dB | C3 | 証拠により不採用 | P1 規則が不成立（FORGE 32 反復 0.205 dB）。32 反復は後に K1 で受振 [4,8]／[3,6] と組んで採用 | `runs/study_063_c3_trace_grid/…_pocs_forge_matched_iterations32_grid/SEL:4-31,71`、`S067/decisions.md:71` |
| study_063 → study_067 | cell a01_b05（64／[6,16]）。C3 の forge-matched 16 反復・P1b・K1 で勝者 | C3 | 現行に含まれる | 共通 grid の cell。本番でも規則どおり再選択 | `runs/study_063_c3_trace_grid/…_pocs_forge_matched_iterations16_grid/SEL:4`、`C3R/SEL:4` |
| study_066（100 反復 grid） | 100 反復、窓 [192,2,8,8,16]／[96,1,4,4,8]（a04_b01）、−0.372 dB | FORGE 4 ms | 証拠により不採用 | 床 0.4 dB 未満。16 反復 grid（0.440）に劣る | `runs/study_066_forge_c3_reference/…_pocs_drr_grid/SEL:4-31,181`、`…_pocs_iterations16_grid/SEL:181`、`DES:12` |
| study_066（16 反復 grid） | 16 反復、end 0.01、受振 [8,16]／[4,8]、a04_b01、0.440 dB（3 grid の最良） | FORGE 4 ms | 証拠により不採用 | 受振は P1b で [4,8] に、16 反復は P-1 の T2 で 32 に置換 | `runs/study_066_forge_c3_reference/…_pocs_iterations16_grid/SEL:4-31,181`、`S067/decisions.md:71,132` |
| study_066（16 反復・end 0.2 grid） | end 0.2、a04_b01、0.321 dB | FORGE 4 ms | 証拠により不採用 | 不採用リスト（0.321 対 0.440） | `runs/study_066_forge_c3_reference/…_pocs_iterations16_end02_grid/SEL:4-31,181`、`DES:88` |
| study_066 → study_067 | cell a04_b01（192／[2,8]）。FORGE の全 POCS grid で勝者 | FORGE 4 ms | 現行に含まれる | 共通 grid の cell。本番でも規則どおり再選択 | `runs/study_066_forge_c3_reference/…_pocs_drr_grid/SEL:4`、`FGR/SEL:4` |
| study_067 P1b（[2,8]） | 16 反復、受振 [2,8]／[1,4]（FORGE 1.154、C3 9.98 dB） | C3 + FORGE | 証拠により不採用 | P1b 規則の C3 ≥ 10.5 dB を満たさない | `S067/decisions.md:71`、`runs/study_067_forge_c3_common_grid/…_p1b_{c3,forge}_pocs_iterations16_rx2x8/SEL:4-31` |
| study_067 P1b（[4,8]） | 16 反復、受振 [4,8]／[2,4]（C3 10.877、FORGE 0.895 dB） | C3 + FORGE | 第2弾で試した | 第2弾 P-1 の比較基準。T2 で K1 に置換 | `S067/decisions.md:71,132`、`DES:85,131` |
| study_067 設計時 screening | K2 案 | C3 + FORGE | 証拠により不採用 | 設計の不採用リスト。批評により第2弾から削除、T2 では未判定 | `DES:91,144,385`、`S067/decisions.md:308,361` |
| study_067 設計時 screening | threshold_end 0.001 | C3 | 証拠により不採用 | C3 で −2.5 dB | `DES:87`、`S067/decisions.md:304` |
| study_067 設計時 screening | 受振窓 [8,8] | FORGE | 証拠により不採用 | FORGE で −0.36 | `DES:89`、`S067/decisions.md:306` |
| study_067 設計時 screening | 予測への low-pass | 記載なし | 証拠により不採用 | 改善が +0.004 以下 | `DES:90`、`S067/decisions.md:307` |
| study_067 第2弾 P-1 | K1：32 反復、end 0.01、受振 [4,8]／[3,6] | C3 + FORGE | 第2弾で試した | T2 で採用（C3 +2.915、FORGE +0.270 dB）。現行値。screening 由来 | `S067/decisions.md:132`、`S067/c3/pocs_grid.yaml:8-22`、`S067/forge/pocs_grid.yaml:8-22` |

注：「現行に含まれる」は、登録した3区分（第2弾で試した／ユーザー規則で不可／証拠により不採用）の外の区分です。現行の固定値か、変更していない共通 grid の cell にそのまま含まれる設定に使います。study_061・064・065 には POCS の新しい勝ち設定がありません。study_061 には POCS の設定 file がなく、study_060 の POCS と比べただけです。064 は study_063 の探索を継続し、065 は C3 勝者を移植しただけです（`studies/study_061_forge_nersi_profile/README.md:91,99`、`studies/study_064_c3_reference_grid/decisions.md:19`、`studies/study_065_forge_expanded_grid/decisions.md:88-90`）。

#### 補足

- **本番 S1 は全件 import**：本番の POCS S1 で新たに実行した run は0件です。P-1 の K1 pilot の50 run を、resolved config・partition・観測波形の一致を確かめて import し、本番の採点コードで再採点しました（`S067/README.md:67,71,73`）。
  - 予測は pilot の snapshot で計算されています。`pocs_batched.py`・`pocs_windows.py` は pilot・C3・FORGE の snapshot で byte 一致です。`forge_mvp_methods.py`・`c3_region_methods.py` は pilot と本番で異なりますが、その差は POCS の計算に影響しません。
  - 3 seed の評価は本番 snapshot で新たに計算しました。
- **K1 は screening 由来**：K1 は設計時 screening（約15案、validation のみ）の最上位案です。P-1 の規則は screening の結果を見た後に登録しました（事後的、開示）。screening の run ごとの記録は repo にありません（`DES:1,131`）。
- **100 反復をやめた経緯**：16 反復への切替は、2026-10-22 のユーザー指示です（`studies/study_063_c3_trace_grid/decisions.md:23-25`）。指示の理由は文書にありません。study_066 の FORGE では、100 反復の grid が −0.372 dB、16 反復の grid が 0.440 dB でした。P1 の候補 {8,16,32} から 100 を外した理由も文書にありません。
- **不採用リストは T1/T2 を経ていません**：設計の不採用リスト（threshold_end 0.2・0.001、受振窓 [8,8]、low-pass、K2）は T1/T2 で判定していません。表のうち、このリストを根拠にした行の「証拠により不採用」は、事前規則による判定ではありません。K2 は設計批評で外しました。理由は、隠れた best-of-75 と C3 偏重の規則を避けるためです（`DES:86-91,385`）。
- **thread 数の記録**：manifest の `threads_at_start`（選択 cell で 4）は、プロセスの OS thread 数です。`cpu_threads` 1 は torch の intra-op thread 数だけを決めます。NumPy 1.26.4 の FFT は単一 thread で動きます。S1 の値は import 処理（選択 cell で C3 約1秒、FORGE 約6秒）で記録されたものです（`SRC/pipelines/run_c3_region_experiment.py:181,239-241`、`SRC/pipelines/run_forge_grid_experiment.py:173,234-236`、`C3R/experiment.json:22`、`C3S1/manifest.json:13,87`、`FGS1/manifest.json:13,114`）。
- **target を見た過去の調整**：C3 の評価窓は study_041 で target を見て選ばれ、全手法に共通です。study_057・058 は test で設定を選んでおり、study_058 の test は study_057 の test と 87.5% 重なります（`studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:37`）。これら FORGE 2 ms の設定は study_067 に持ち込んでいません。

### 付録 A-5. DRR（`drr_grid`）

出典の略記：`C3/` = `runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces/`、`FG/` = `runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge/`、`C3s1/` = `C3/method_runs/drr_grid/S1/a05_b03/`、`FGs1/` = `FG/method_runs/drr_grid/S1/a03_b02/`、`S67/` = `studies/study_067_forge_c3_common_grid/`、`src/` = `C3/source/src/seis_interp/`（FORGE の凍結 source と同一）、`design` = `reports/study_067_design_round2.md`。`:n` は行番号です。

#### Protocol

| 項目 | C3 | FORGE | 出典 |
|---|---|---|---|
| optimizer/algorithm | —（該当なし）。damped rank reduction（level-4 Hankel、NumPy SVD）＋観測の再挿入 | 同左 | `C3s1/config.resolved.yaml:1-25`、`FGs1/config.resolved.yaml:1-25`、`src/processing/drr_batched.py:42-53,93-97` |
| 学習率 | —（該当なし） | —（該当なし） | 両 resolved config（key なし） |
| weight decay | —（該当なし） | —（該当なし） | 両 resolved config（key なし） |
| 更新数/反復数 | 60 反復（block・周波数 batch ごと） | 60 反復 | `C3s1/config.resolved.yaml:10`、`FGs1/config.resolved.yaml:10`、`src/processing/drr_batched.py:95-97` |
| 1更新の loss trace 数 | —（該当なし）。`loss` は選択用 validation 指標の名前 | —（該当なし） | `C3s1/config.resolved.yaml:2`、`FGs1/config.resolved.yaml:2` |
| inner mask（名目/実効） | —（該当なし） | —（該当なし） | `design:226`（ReGSI・CCNet のみ） |
| データの再利用・勾配の鮮度 | —（勾配なし、決定論的）。各 block の観測 trace で毎反復を拘束し、厳密に再挿入。観測 23,593 本 | 同左。観測 13,271 本 | `src/processing/drr_batched.py:93-97,110,113-136,237`、`C3s1/manifest.json:24`、`FGs1/manifest.json:23` |
| clip（発動率）または would-clip 率 | —（該当なし）。key なし（未記録） | 同左 | `C3s1/manifest.json`、`C3/method_runs/drr_grid/evaluation/seed_732{1,2,3}/manifest.json` と FORGE の同じ4件（`clip` 0 件） |
| EMA（primary と shadow） | —（該当なし） | —（該当なし） | 同じ8 manifest（`ema` 0 件） |
| 精度 | 入出力 float32、内部 float64（rFFT complex128・SVD）。CPU のみ、CUDA peak 0 | 同左 | `src/training/c3_region_methods.py:24,35`、`src/processing/drr_batched.py:78-80,109-110,202-204,236`、`C3s1/manifest.json:65-66`、`FGs1/manifest.json:95-96` |
| パラメータ数 | 0（S1・3 seed） | 0（S1）。3 seed の manifest は key なし | `C3s1/manifest.json:54`、`C3/method_runs/drr_grid/evaluation/seed_732{1,2,3}/manifest.json:85`、`FGs1/manifest.json:83`、`FG/method_runs/drr_grid/evaluation/seed_732{1,2,3}/manifest.json` |
| model seed | なし（null、決定論的）。S1 trace seed 620702、評価 seed 7321–7323 | なし（null）。S1 trace seed 600702、評価 seed 7321–7323 | `C3s1/config.resolved.yaml:4`、`C3s1/manifest.json:5-6`、`FGs1/manifest.json:5-6`、`S67/c3/config.yaml:47-51`、`S67/forge/config.yaml:270-273` |
| dataset 固有の固定差 | なし（両 dataset の method yaml は SHA256 一致） | 手法値の差はなし。入力側：500 m/s 整列（max_delay 1,171）、trace_oracle 振幅 | `S67/decisions.md:418,428`、`S67/dataset_differences.yaml:2-8,92-95`、`FGs1/config.resolved.yaml:31-33` |
| grid | rank {6,12,18,24,30} × 空間窓5種（overlap は窓から導出）。両 dataset 同一 | 同左 | `S67/grid.yaml:126-142`（rank）、`S67/grid.yaml:143-203`（窓・overlap） |
| 時間長・空間形状・周波数 | [16,32,8,32]、384 sample・8 ms → rFFT 512（257 bin）、0–62.5 Hz | [6,32,8,32]、1,001 sample・4 ms を整列 canvas 2,172 sample に配置 → rFFT 4096（2,049 bin）、0–125 Hz | `S67/c3/config.yaml:9-17`、`S67/forge/config.yaml:8-16`、`S67/dataset_differences.yaml:2-4`、`FGs1/manifest.json:88`、`C3s1/config.resolved.yaml:11-12`、`src/processing/drr.py:79-87` |
| damping・overlap・境界 | damping 2。overlap [0,4,0,4]、clip_and_shift、正の interior-Hann 合成 | 同左 | `C3s1/config.resolved.yaml:9,18-25`、`FGs1/config.resolved.yaml:9,18-25`、`S67/dataset_differences.yaml:11` |
| S1 選択 cell | a05_b03（rank 30・[6,8,6,8]）、val 17.696 dB | a03_b02（rank 18・[4,8,4,8]）、val 1.222 dB（clean_target） | `C3/workers/drr_grid_selection.json:2-4`、`FG/workers/drr_grid_selection.json:2-4`、`C3s1/manifest.json:44`、`FGs1/manifest.json:67` |
| 実行設定（結果に影響なし） | CPU、workers 12、thread 1、nice 10、同時候補 2 | 同左、同時候補 4 | `C3s1/config.resolved.yaml:26-30`、`C3s1/manifest.json:12`、`FGs1/manifest.json:12`、`S67/c3/config.yaml:141-142`、`S67/forge/config.yaml:158-159`、`S67/decisions.md:224` |

#### 調整の履歴

| study | run 数 | 調整した内容 | 選択に使った指標 | 出典 |
|---|---|---|---|---|
| study_039（C3） | pilot 12（T の 4,091 本で採点）＋全領域確認 1 | rank・反復・damping・fmin。rank 12・60反復・0 Hz を採用（8.52→12.13 dB） | test/target | `studies/study_039_c3_drr_target_tuning/README.md:8-9,26,30-47,60-63` |
| study_041（C3 の窓） | 1（候補3窓の最初で到達） | C3 crop 窓 shot [18,50)・receiver-y [18,50)。NeRSI の target SNR で選択（DRR 固有ではない） | test/target | `studies/study_041_c3_nersi_translated_window/README.md:3-4,8-12,19`、`design:240` |
| study_057（FORGE 2 ms、70%） | 5（relMSE ≤ 0.85 で打切り） | rank・反復・窓・overlap（damping は全候補で 3）。relMSE 0.9610→0.8373 | test/target | `reports/forge_057_drr_tuning.md:20-41`、`runs/study_057_forge_2ms_random70_trace_rms/20260922T090940122764Z_51acdb640a29_drr_tuning/search_results.csv:2-6` |
| study_058（FORGE 2 ms、80%）／059 | 058：test 4 候補＋設定エラー 1。059：0 | rank 1/2/3・窓 16。d057_tuned が最良（0.8845）。059 は継承のみ | test/target | `runs/study_058_forge_2ms_random80_align500_matched20k/20260923T012846041044Z_25c5193588b2_drr_grid_matched_tuning/search_results.csv:2-5`、`runs/study_058_forge_2ms_random80_align500_matched20k/20260923T012641301338Z_25c5193588b2_drr_grid_matched_tuning/manifest.json:361`（failure_reason）、`studies/study_058_forge_2ms_random80_align500_matched20k/decisions.md:33-37`、`studies/study_059_forge_2ms_random70_align500_matched20k/drr_grid.yaml:25-29` |
| study_060（FORGE 2 ms fair grid） | S1 16＋最終 1 | rank {1–4}×反復 {6,12}×damping {2,3}（窓 [1,8,1,8]・overlap [0,6,0,6] 固定）。rank 3・12反復・damping 2（V 0.7952） | validation のみ | `reports/forge_060_experiment_and_search_conditions.md:42-47,338-348,362`、`studies/study_060_forge_fair_grid/decisions.md:40` |
| study_062_c3_v4_fair_grid（C3、2026-10-08） | 計画 16：完了 9・中断 3・未開始 4 | rank {6,12,18,24}×反復 {60,120}×damping {2,3}（窓 [5,8,4,8]・overlap 0）。決定なし、study 文書は削除済み | 選択なし | `runs/study_062_c3_v4_fair_grid/fair_grid/drr_grid/S1/stage.json:18`、同 `S1/*.yaml`（16 件）、`runs/study_062_c3_v4_fair_grid/*_drr_grid_matched_tuning/manifest.json`（status：completed 9・interrupted 3）、`runs/study_062_c3_v4_fair_grid/*_drr_grid_matched_tuning/search_results.csv:2`（完了 9 件） |
| study_062（C3 region holdout） | 25 中、完了 19・停止 1（validation で採点） | 現行の DRR grid（rank 6–30 × 窓 [4,6,4,6]〜[8,16,8,16]）の初出。ユーザーが中止 | 選択なし | `studies/study_062_c3_region_grid/README.md:3,8-11,50-52`、`studies/study_062_c3_region_grid/drr_grid.yaml:7-22`、`studies/study_062_c3_region_grid/grid.yaml:189-241`、`runs/study_062_c3_region_grid/20261009T135126835519Z_05c92e79ce80_regions/method_runs/drr_grid/S1/*/manifest.json`（status） |
| study_063（C3 trace holdout） | S1 25＋3 seed（7311–7313、test 採点） | 勝者 a02_b02（rank 12・[4,8,4,8]・overlap 0）、val 15.175 dB。grid と固定値を study_067 に継承 | validation のみ | `runs/study_063_c3_trace_grid/20261011T145500928121Z_05c92e79ce80_traces/workers/drr_grid_selection.json`、同 run の `method_runs/drr_grid/S1/a02_b02/manifest.json:39`、`S67/decisions.md:61`、`design:242` |
| study_065／066（FORGE 4 ms） | 各 1 | 調整なし。study_063 の C3 勝者を移植。066 は val 0.706 dB（D-1 の基準） | 選択なし | `studies/study_065_forge_expanded_grid/decisions.md:90`、`studies/study_066_forge_c3_reference/decisions.md:3,6`、`runs/study_066_forge_c3_reference/20261020T225334465289Z_05c92e79ce80/method_runs/drr_grid/manifest.json` |
| study_066（FORGE 4 ms、validation のみの追加 grid） | 新規完了 32＋停止 3（pocs_drr 3+1、rank3to15 4+1、joint25 25、small_windows 0+1。small_windows の他 24 は未開始、うち無効 11） | pocs_drr：60反復・rank 6 の3窓。rank3to15・joint25：12反復で rank 1–15・小窓。採用なし | 選択なし | `runs/study_066_forge_c3_reference/20261021T071138056985Z_05c92e79ce80_pocs_drr_grid/drr_grid_summary.csv:3-5`、`runs/study_066_forge_c3_reference/20261021T225834225753Z_05c92e79ce80_drr_iterations12_rank3to15_grid/drr_grid_summary.csv:2-5`、`runs/study_066_forge_c3_reference/20261022T012118982457Z_05c92e79ce80_drr_iterations12_joint25_grid/drr_grid_summary.csv:2-26`、`runs/study_066_forge_c3_reference/20261022T011447389059Z_05c92e79ce80_drr_iterations12_small_windows_grid/invalid_candidates.json`、`S67/forge/config.yaml:287-294` |
| study_067 設計時 screening（D0–D7） | DRR 14（C3 8・FORGE 6）、scratch 実装 | 12/30反復、damping 1/3、overlap [0,4,0,4]（30/60反復）。不採用リストと D-1 規則を作成 | validation のみ | `design:1,93-101,111`、設計時の scratch 記録（リポジトリ外） |
| study_067 D-1（第2弾 pilot） | 4（FORGE D7 は watchdog 停止後に再実行、C3 D7、FORGE D5b）、完了 3 | 窓由来 overlap（line 0・point 50%）を T2 で採用。D7：C3 16.087、FORGE 1.098 dB | validation のみ | `S67/decisions.md:182-194`、`S67/round2_outcomes.json`（d1）、`design:107,111,132` |
| study_067 D-2（コスト） | 計時 4（2反復・worker 1、overlap [0,4,0,4]、採点なし） | workers・同時候補数・priority のみ。アルゴリズム値は不変 | 選択なし | `S67/decisions.md:196-222`、`S67/round2_outcomes.json`（d2、drr_estimates） |
| study_067 S1 本番（参考） | S1 25×2（各 dataset 新規 24＋D7 import 1）＋3 seed×2 | 選択のみ：C3 a05_b03、FORGE a03_b02 | validation のみ | `C3/workers/drr_grid_selection.json:2,169`、`FG/workers/drr_grid_selection.json:2,169`、`S67/README.md:68` |

study_062_c3_v4_fair_grid の結果（validation relMSE）を補います。damping 2 は、完了した rank 6・12 で damping 3 より良好でした（60反復で 0.0601 対 0.0617、0.0420 対 0.0502）。damping 2 での120反復の改善は relMSE 0.001 未満でした。完了した中の最良は r18_it60_damp2 の 0.0392 です。

#### 過去の勝ち設定の扱い

| study | 勝った設定 | dataset | 扱い | 根拠 | 出典 |
|---|---|---|---|---|---|
| study_057 | rank 2・damping 3・6反復・窓 [1,8,1,8]・overlap [0,6,0,6]（test で選択） | FORGE 2 ms | ユーザー規則で不可 | rank 2・窓 [1,8,1,8] は共通軸の外。FORGE 専用の固定値も不可 | `reports/forge_057_drr_tuning.md:20-26,41`、`design:8,13`、`S67/grid.yaml:126-203` |
| study_058 | 同じ設定（d057_tuned）が test 4 候補で最良（0.8845） | FORGE 2 ms | ユーザー規則で不可 | study_057 と同じ | `runs/study_058_forge_2ms_random80_align500_matched20k/20260923T012846041044Z_25c5193588b2_drr_grid_matched_tuning/search_results.csv:2-5` |
| study_059 | 同じ設定を再選択なしで継承（0.7915） | FORGE 2 ms | ユーザー規則で不可 | 新しい選択なし。study_057 と同じ設定 | `studies/study_059_forge_2ms_random70_align500_matched20k/drr_grid.yaml:9-29`、`studies/study_059_forge_2ms_random70_align500_matched20k/README.md:39` |
| study_057–059 | damping 3（勝ち設定の一部） | FORGE 2 ms | 証拠により不採用 | 設計の不採用リスト：screening D6（60反復）で C3 13.77、FORGE 0.054 dB | `design:101`、`reports/forge_057_drr_tuning.md:23`、`studies/study_058_forge_2ms_random80_align500_matched20k/drr_grid.yaml:11`、`studies/study_059_forge_2ms_random70_align500_matched20k/drr_grid.yaml:11` |
| study_057–060 | overlap の形（line 軸 0・point 軸のみ） | FORGE 2 ms | 第2弾で試した | D-1 で point 軸 50% を共通 grid の窓で試し、採用 | `S67/decisions.md:182-192`、`studies/study_060_forge_fair_grid/decisions.md:40` |
| study_057–060 | point 軸 75% overlap（窓 [1,8,1,8] に [0,6,0,6]） | FORGE 2 ms | ユーザー規則で不可 | 窓 [1,8,1,8] は共通窓軸の外。75% は共通窓では未検証 | `reports/forge_057_drr_tuning.md:25-26`、`studies/study_060_forge_fair_grid/decisions.md:40`、`design:8,96`、`S67/decisions.md:192` |
| study_060 | rank 3・窓 [1,8,1,8] | FORGE 2 ms | ユーザー規則で不可 | 共通の rank 軸・窓軸の外 | `reports/forge_060_experiment_and_search_conditions.md:338-348`、`design:8` |
| study_060 | 12 反復 | FORGE 2 ms | 証拠により不採用 | 設計の不採用：C3 10.12、FORGE 0.82 dB（事前登録規則ではない） | `design:99`、`reports/forge_060_experiment_and_search_conditions.md:348` |
| study_060 | damping 2 | FORGE 2 ms | 現行に含まれる※ | 両 dataset の現行固定値 | `design:94`、`C3s1/config.resolved.yaml:9`、`FGs1/config.resolved.yaml:9` |
| study_062 v4（決定なし） | 完了中の最良：rank 18・60反復・damping 2・窓 [5,8,4,8]・overlap 0（0.0392） | C3 | ユーザー規則で不可 | 窓 [5,8,4,8] は共通窓軸の外。rank 18・60反復・damping 2 は現行に含まれる | `runs/study_062_c3_v4_fair_grid/20261008T205002533967Z_05c92e79ce80_drr_grid_matched_tuning/search_results.csv:2`、`runs/study_062_c3_v4_fair_grid/fair_grid/drr_grid/S1/stage.json:18` |
| study_063 | 60反復・damping 2・0 Hz〜Nyquist | C3 | 現行に含まれる※ | 設計で【決定】。両本番 config に含まれる | `design:94`、`runs/study_063_c3_trace_grid/20261011T145500928121Z_05c92e79ce80_traces/workers/drr_grid_selection.json`（winner.config.algorithm）、`C3s1/config.resolved.yaml:9-12` |
| study_063 | overlap [0,0,0,0]（勝者 a02_b02、val 15.175 dB） | C3 | 証拠により不採用 | D-1（T2、screening 由来）で窓由来 overlap に置換。旧 overlap の run は import 拒否 | `S67/decisions.md:124,182-194`、`S67/README.md:71`、`S67/round2_outcomes.json`（imports.c3.study063_rejected_candidates） |
| study_063 | cell rank 12 × [4,8,4,8] | C3 | 第2弾で試した | D7 として overlap 付きで実行し、S1 a02_b02 に import。両 dataset で非選択（C3 relMSE 0.03073 対 0.02179、FORGE 0.80027 対 0.78588） | `S67/decisions.md:194`、`C3/workers/drr_grid_selection.json`・`FG/workers/drr_grid_selection.json`（s1_records） |
| study_066（validation のみの grid） | 12反復（最良は rank 15・[4,8,4,8] の 0.840 dB、rank 12 は 0.820 dB） | FORGE 4 ms | 証拠により不採用 | 12反復は設計の不採用（C3 10.12 dB）。rank 15・窓 [1,4,1,4]〜[2,6,2,6] は軸外（ユーザー規則） | `runs/study_066_forge_c3_reference/20261022T012118982457Z_05c92e79ce80_drr_iterations12_joint25_grid/drr_grid_summary.csv:2-26`、`design:99` |
| study_067 D-1 | 窓由来 overlap（line 0・point 50%） | C3・FORGE | 第2弾で試した | T2 合格で採用：D7 が C3 16.087（基準 15.175）、FORGE 1.098（基準 0.706） | `S67/decisions.md:182-192`、`S67/dataset_differences.yaml:11`、`S67/round2_outcomes.json`（d1） |
| study_067 D5b | 30反復＋overlap [0,4,0,4]（FORGE 1.089、C3 14.76 dB） | C3・FORGE | 証拠により不採用 | 30反復は設計の不採用（overlap 0 で C3 14.00 dB）。D5b は再現確認だけで、import していません | `design:100,132`、`S67/decisions.md:194`、`reports/study_067_drr_cpu_check_plan.md:10-11` |

※「現行に含まれる」は、設計 (5) 11 で登録した3ラベル（第2弾で試した／ユーザー規則で不可／証拠により不採用）の外にある第4ラベルです。現行の固定値か、変更していない共通 grid の cell に既に含まれる設定に付けます（`design:241`）。

study_061・062（region）・064・065 には、DRR の新しい勝ち設定がありません。061 は study_060 の DRR を再利用し、062 は選択前に中止し、064 は study_063 を継続し、065 は study_063 を移植しました（`studies/study_061_forge_nersi_profile/README.md:51-52`、`studies/study_062_c3_region_grid/README.md:3`、`studies/study_064_c3_reference_grid/README.md:22`、`studies/study_065_forge_expanded_grid/decisions.md:90`）。

#### 補足

- **未記録と該当なし**：clip・would-clip・EMA の key は、S1 と評価 seed の8 manifest のどれにもありません（history.csv も空です）。表の「—」は「0 を測った」ではなく「記録がない」ことを示します。FORGE の3 seed manifest には parameter_count もありません。seed の config SHA256 は S1 と一致します。予測は barrier 前（2026-10-28）に計算し、test の採点は barrier 後に行いました（barrier 通過は C3 10-31、FORGE 11-03。`C3/workers/s1_barrier.json:10`、`FG/workers/s1_barrier.json:10`。採点 run は 1.9–10.8 s）。
- **固定値の由来**：damping 2・60反復・rank×窓 grid を選んだ理由は、リポジトリの文書に書かれていません。実測の裏付けとして、2026-10-08 の C3 validation run（`runs/study_062_c3_v4_fair_grid/`、上の調整の履歴の注）があります。
- **規則の性質**：事前登録した DRR の規則は D-1（T2）だけで、これも screening 由来（事後的、開示）です。12/30反復と damping 1/3 の不採用は、設計の不採用リスト（主に screening 14 run）によるもので、事前登録規則ではありません。screening の記録はリポジトリ外の scratch にしかありません。damping 1 は30反復（D3）でしか試していません。
- **推定・代理値**：
  - 設計の「FORGE 0.82」（12反復）には出典が書かれていません。screening の FORGE には12反復の run がありません（30反復の D5a は 0.825）。study_066 の rank 12・12反復（0.8197 dB）と値は一致しますが、対応は推定です。
  - Nyquist（C3 62.5 Hz、FORGE 125 Hz）は sample 間隔からの導出値です。
  - D-2 の計時は overlap [0,4,0,4] で行ったため、採用 overlap のコストは代理値です。
- **文書にない run**：study_066 の FORGE validation 専用 DRR grid（新規完了32 run）は、study_066 の README と decisions に記載がありません。`S67/forge/config.yaml:287-294` の import_from にだけ現れます。study_062_c3_v4_fair_grid も study 文書が削除されており、run だけが残っています。
- **target の使用**：test/target を直接選択に使ったのは study_039・041・057・058 です。設計前には、study_063（C3 test 約 15.2 dB）、study_065（FORGE val 0.376・test 0.350 dB）、study_066（FORGE test 0.637 dB）の DRR test 値も採点済みでした（各 run の manifest。閲覧の有無は文書にありません）。study_057/058 の勝ち設定は、relMSE ≤ 0.85 で打ち切った探索で採用した値であり、全 grid の最適値ではありません。C3 の選択 cell は rank 軸の上端（30）です。
