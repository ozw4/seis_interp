# 全 FFID・train 50% 条件で 20 dB を目指した段階実験レポート

- 実施日: 2026-08-29
- 対象データ: SEG C3 Narrow-Azimuth
- 対象 study: [`study_018_all_ffid_50pct_neighbor_inpainter`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/README.md)
- 正式な成功条件: `oracle_per_trace_unit_rms_global_snr_db > 20.0`
- 完了済み diagnostic 最良: Stage 09、`18.2484 dB`
- 正式 run: `20260829T075432Z_ee3d9e5_formal_50000_steps`
- 正式結果: **`20.4604 dB`、成功**

## 結論

50% 条件で完了している Stage 01–15 は、全 4,780 eligible FFID、全 287,933
validation traces、リーク監査を含む同じ formal scope で比較した。事前の
diagnostic 最良は Stage 09 の **18.2484 dB** で、この時点では
20 dB まで **1.7516 dB** 残っていた。

切り分けでは、source-x 座標、同一 source-x line 内の K274 aperture、FiLM、全 trace を
覆う temporal receptive field、width 384、target-coordinate neighbor gate、軽量な
neighbor-wise alignment FIR が正の差分を示した。最大の改善要因は学習 budget であり、
width 256 条件を 2,500 から 10,000 updates へ伸ばすと
`+1.8819 dB` 改善し、最終 checkpoint でも上昇中だった。これらをまとめた
Stage 15 architecture を 50,000 updates 学習する条件を formal candidate として凍結した。

その fresh formal run は 30,000 step で初めて 20 dB を超え、50,000 step の best
checkpoint で **20.4604 dB** に到達した。厳密な閾値への余裕は
**+0.4604 dB** である。checkpoint 再評価は保存値と完全一致し、
formal scope checks も全て通過したため、指定された成功条件を満たす。

> **Formal result: SUCCESS — `20.4604 dB > 20.0 dB`**

## 契約(split・成功条件)

split 契約、「train ratio 50% FFID」の解釈、acceptance criteria の正本は
[`study_018` README](../studies/study_018_all_ffid_50pct_neighbor_inpainter/README.md)、
入力契約は [`inputs.yaml`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/inputs.yaml)、
実行条件は [`config.yaml`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/config.yaml)
である。本報告が依拠する要点:

- 各 eligible FFID 内で trace の 50% を train、holdout の 25% / 75% を validation / test とする。
  FFID の半数を丸ごと train にする shot extrapolation 条件ではない。
- effective train / validation / test = 1,151,731 / 287,933 / 863,801(全 4,780 eligible FFID、
  canonicalization で 9 / 2 / 4 行除去)。FFID 1746 の 544 traces は amplitude QC で `excluded`。
- 全 survey で物理座標 `[source_x_m, source_y_m, receiver_x_m, receiver_y_m]` が重複する cell が
  15 個・30 行あった。split や振幅を参照せず最小 `array_row` だけを残す global canonicalization を
  適用し、validation target と同じ物理 cell の twin trace が train neighbor に入る経路を閉じた。

成功判定は次による。validation trace ごとに target 自身の RMS で unit-RMS 化し、モデルの
raw prediction と比較する。prediction の再 unit-RMS 化は診断値に限る。

```text
oracle_per_trace_unit_rms_global_snr_db
  = 10 log10(
      sum_i,t(target_unit[i,t]^2)
      / sum_i,t((target_unit[i,t] - prediction_raw[i,t])^2)
    )

success ⇔ oracle_per_trace_unit_rms_global_snr_db > 20.0
          AND formal scope checks are all true
```

比較は厳密な `>` であり、20.0 dB ちょうどは失敗とする。20 dB は、この評価領域で global error
energy が signal energy の 1% 未満であることに対応する。

## リーク監査と formal scope

Stage 01–15 と正式 run の各 `metrics.json` では `scope_success: true` を確認した。
各 run で次を機械的に検査している。

- effective train / validation / test counts が
  1,151,731 / 287,933 / 863,801 と一致
- eligible FFID 数が 4,780、sample 数が 625、fully excluded FFID が `[1746]` と一致
- canonicalization 後の重複物理 cell が 0
- train geometry の coordinate collision cell / row が 0
- train-validation の物理座標 overlap row が 0
- train-validation の完全一致 unit-amplitude duplicate row が 0
- neighbor offset に target center `(0, 0, 0, 0)` が 0 個
- neighbor amplitude の供給元が train split のみ
- test / excluded amplitude value row を materialize しない
- validation target は checkpoint 選択と metric にのみ使用
- test target は checkpoint 選択に不使用
- 保存 checkpoint の raw validation metric を tolerance `1e-8` で再現

入力固定のため `amplitudes.npy` 全体の byte 列は hash 計算するが、test / excluded の
数値 row は tensor として読み込まない。この区別は各 run の `amplitude_access` に記録した。

## 共通の実験条件

特記しない Stage では次を共通とした。

| 項目 | 条件 |
|---|---|
| seed | 42 |
| target normalization | per-trace RMS |
| optimizer | AdamW |
| learning rate | `5e-4` |
| schedule | cosine、minimum `1.5e-5` |
| weight decay | `1e-5` |
| batch size | 96 complete traces |
| loss | MSE + `0.1 × first-difference MSE` |
| neighbor dropout | 0.05 |
| gradient clip | 1.0 |
| mixed precision | bfloat16 |
| checkpoint metric | raw oracle per-trace unit-RMS global S/N |
| validation scope | 全 287,933 traces |
| device | NVIDIA H100 NVL、`cuda:1` |

K104 では relative receiver-x ±1、source shot ±2、relative receiver-y ±3 を同一
source-x line 内で探索する。K274 では relative receiver-x radius を 2、staggered source-y
half-shot radius を 4、relative receiver-y radius を 5 に拡張する。source-x line radius は
0 のままで、target center は除外する。欠損または train 以外の位置には振幅 0 と
availability `false` を与える。

## Stage 01–15 の段階的切り分け

表の dB は各 immutable run の
`oracle_per_trace_unit_rms_global_snr_db` である。「差分」の比較先を明示し、複数条件を
同時変更した Stage は bundle としてのみ解釈した。全 Stage で best checkpoint は最終
step、`scope_success=true`、`metric_success=false`、`success=false` だった。

| Stage | 切り分け条件 | K / width / steps | Primary dB | 比較差分 | 採否 |
|---:|---|---|---:|---:|---|
| [01](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T055137Z_ee73dba_stage01_baseline/metrics.json) | Study 017 model を 50% split へそのまま移植、3 target coords | 104 / 128 / 2,500 | 15.376 | 基準 | baseline |
| [02](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T060524Z_251ccb8_stage02_source_x_coordinate/metrics.json) | source-x を追加した 4 coords、staggered multiline geometry | 104 / 128 / 2,500 | 15.5931 | Stage 01 比 `+0.2171` | 採用 |
| [03](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T060845Z_a9aa723_stage03_receiver_aperture/metrics.json) | 同一 source-x line の receiver aperture を K274 へ拡大 | 274 / 128 / 2,500 | 15.7029 | Stage 02 比 `+0.1098` | 採用 |
| [04](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T061249Z_4db20f5_stage04_crossline_aperture/metrics.json) | source-x line radius 1 の flat crossline aperture | 272 / 128 / 2,500 | 15.3473 | Stage 02 比 `-0.2458` | 不採用 |
| [05](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T061731Z_655bcbb_stage05_receiver_aperture_10000_steps/metrics.json) | Stage 03 を 10,000 updates へ延長 | 274 / 128 / 10,000 | 17.424 | Stage 03 比 `+1.7211` | budget 延長を採用 |
| [06](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T062603Z_3917d7a_stage06_receiver_aperture_film/metrics.json) | 各 temporal block を 4 target coords で FiLM conditioning | 274 / 128 / 2,500 | 15.776 | Stage 03 比 `+0.0731` | 採用 |
| [07](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T063112Z_41301b6_stage07_full_receptive_width256/metrics.json) | width 256、13 blocks、full-trace receptive field | 274 / 256 / 2,500 | 16.3666 | Stage 06 比 `+0.5906` | bundle を採用 |
| [08](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T063843Z_f8ef160_stage08_pure_mse_no_dropout/metrics.json) | derivative weight 0、neighbor dropout 0 | 274 / 256 / 2,500 | 16.3555 | Stage 07 比 `-0.0111` | bundle を不採用 |
| [09](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T064550Z_2d1e8d9_stage09_full_receptive_width256_10000_steps/metrics.json) | Stage 07 を 10,000 updates へ延長 | 274 / 256 / 10,000 | **18.2484** | Stage 07 比 `+1.8819` | budget 延長を採用、完了済み最良 |
| [10](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T070218Z_fdacfa0_stage10_receiver_aperture_epoch_sampling/metrics.json) | Stage 05 の target sampling を epoch without replacement 化 | 274 / 128 / 10,000 | 17.4221 | Stage 05 比 `-0.0019` | 改善なし、不採用 |
| [11](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T071037Z_fdacfa0_stage11_stem_kernel31/metrics.json) | Stage 07 の stem kernel を 15→31 | 274 / 256 / 2,500 | 16.1662 | Stage 07 比 `-0.2003` | 不採用 |
| [12](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T071831Z_186bf32_stage12_target_coordinate_neighbor_gate/metrics.json) | target-coordinate masked-softmax neighbor gate | 274 / 256 / 2,500 | 16.4172 | Stage 07 比 `+0.0507` | 採用 |
| [13](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T072453Z_d1227f6_stage13_width384/metrics.json) | Stage 07 の width を 256→384 | 274 / 384 / 2,500 | 16.6023 | Stage 07 比 `+0.2357` | 採用 |
| [14](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T073548Z_645d4d4_stage14_neighbor_alignment_fir31/metrics.json) | identity 初期化した per-neighbor depthwise FIR、kernel 31 | 274 / 256 / 2,500 | 16.4975 | Stage 07 比 `+0.1309` | 採用 |
| [15](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T074408Z_3524037_stage15_combined_width384_gate_fir31/metrics.json) | width 384 + gate + FIR31 を結合 | 274 / 384 / 2,500 | **16.8037** | Stage 07 比 `+0.4371` | formal architecture に採用 |

Stage 02 の新 multiline geometry は、Stage 01 の K104 offsets と物理的に同じ近傍を
表現する。実データ 10,000 targets の neighbor row matrix が完全一致することを確認した
ため、Stage 01→02 は主として source-x target coordinate の効果として解釈できる。

Stage 07 は width と dilation 列を同時に変更したため、それぞれの単独効果には分解しない。
同様に Stage 08 は derivative loss と dropout を同時に外した bundle であり、個別の正負は
主張しない。Stage 10 の差は約 -0.002 dB と小さく、epoch sampling の明確な利得はないと
判断して、既存の replacement sampler を formal 条件に残した。

### 学習 budget の推移

10,000-step run はいずれも最終評価まで改善が続いた。total-step 数に応じて cosine
schedule 自体も変わるため、下表は「update 数だけ」の純粋な差ではなく、学習 horizon と
schedule の bundle である。

| Step | Stage 05: width 128 (dB) | Stage 09: width 256 + FiLM + full RF (dB) |
|---:|---:|---:|
| 1 | 0.8963 | 1.2097 |
| 2,500 | 15.7711 | 16.2697 |
| 5,000 | 16.6988 | 17.4218 |
| 7,500 | 17.23 | 18.0219 |
| 10,000 | **17.424** | **18.2484** |

## 補助診断

以下の 2 件は次の Stage を選ぶために terminal 上で行った read-only probe であり、独立した
run directory、Git SHA、`inputs.lock.json`、`metrics.json` を持たない。そのため、方向性の
判断材料には使うが、20 dB 成功の正式証拠には使わない。

### Train-only integer-lag 診断

canonical effective train から seed 4 で 2,000 targets を非復元抽出し、K274 のうち
pure relative-receiver-y offsets `(0, 0, 0, Δry)` だけを調べた。target と neighbor は
ともに unit-RMS とし、integer lag `[-20, 20]` を探索後、全対象を pool した least-squares
scalar を 1 個 fit した。validation / test amplitudes は使っていない。

| Δry | Available n | Best integer lag (samples) | Pooled LS scalar | Train S/N (dB) |
|---:|---:|---:|---:|---:|
| -5 | 882 | +15 | 0.5669 | 1.6838 |
| -4 | 950 | +12 | 0.5974 | 1.9172 |
| -3 | 943 | +9 | 0.6473 | 2.3584 |
| -2 | 987 | +6 | 0.6718 | 2.6073 |
| -1 | 977 | +3 | 0.7892 | 4.2350 |
| +1 | 1,018 | -3 | 0.7956 | 4.3514 |
| +2 | 983 | -6 | 0.6724 | 2.6124 |
| +3 | 950 | -9 | 0.6354 | 2.2431 |
| +4 | 964 | -12 | 0.6102 | 2.0203 |
| +5 | 907 | -15 | 0.5767 | 1.7552 |

最適 lag は relative receiver-y 40 m あたりほぼ 3 samples、すなわち 24 ms で、正負も
対称だった。この規則性は train-only neighbor の局所時間ずれをモデル化する根拠になり、
±15 samples を覆う kernel-31 alignment FIR を Stage 14 で試した。一方、lag と scalar
だけの S/N は 1.68–4.35 dB に留まるため、単一近傍の剛体 shift だけでは補間できない。

### Checkpoint ensemble probe

全 validation 287,933 traces から seed 44 で 500 traces を抽出した。478 FFID を含む。各 target の
K274 input が train-only であることを assert し、Stage 05 / 07 / 08 best checkpoint の
raw output を CPU float32 で再計算した。

| Prediction | 500-trace sample S/N (dB) | Full-validation S/N (dB) | Sample best との差 |
|---|---:|---:|---:|
| Stage 05 | **17.60536** | 17.424 | 0.00000 |
| Stage 07 | 16.46565 | 16.3666 | -1.13971 |
| Stage 08 | 16.47349 | 16.3555 | -1.13187 |
| equal mean: Stage 05 + 07 | 17.33827 | — | -0.26709 |
| equal mean: Stage 05 + 08 | 17.34377 | — | -0.26159 |
| equal mean: Stage 07 + 08 | 16.59478 | — | -1.01058 |
| equal mean: Stage 05 + 07 + 08 | 17.16858 | — | -0.43678 |

| Checkpoint pair | Flat prediction Pearson r | Per-trace MSE Pearson r |
|---|---:|---:|
| Stage 05 / 07 | 0.85672 | 0.96285 |
| Stage 05 / 08 | 0.85616 | 0.97288 |
| Stage 07 / 08 | 0.94316 | 0.99247 |

Stage 07+08 の平均はその 2 本の単独値より `+0.12129 dB` 良いが、sample best の
Stage 05 より 1.01058 dB 低い。誤差相関も高く、単純等重み ensemble では 20 dB への
差を埋められないと判断して見送った。validation subset を見て決めた probe なので、
ensemble weight を最適化して formal 候補にすることも避けた。

## 凍結した formal candidate

Stage 12–15 の正の model 差分と Stage 09 の学習曲線を受け、commit
`ee3d9e5` で次の条件を [`config.yaml`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/config.yaml)
に凍結した。

| 項目 | Formal 条件 |
|---|---|
| neighborhood | K274、same source-x line、rx radius 2、source-y half-shot radius 4、ry radius 5 |
| target coordinates | source-x、source-y、relative receiver-x、relative receiver-y |
| hidden width | 384 |
| temporal blocks | 13、dilations `[1,2,4,8,16,32,64,32,16,8,4,2,1]` |
| temporal receptive field | CNN 1,155 samples、alignment FIR を含む amplitude path 1,185 samples（入力 625 samples 全体を包含） |
| coordinate conditioning | FiLM |
| neighbor gating | target-coordinate masked softmax |
| neighbor alignment | identity-center 初期化、depthwise FIR、kernel 31 |
| parameter count | 9,210,121 |
| objective | MSE + 0.1 × first-difference MSE |
| neighbor dropout | 0.05 |
| target sampler | with replacement |
| total steps | 50,000 |
| evaluation interval | 5,000 steps |
| training audit | 287,933 train traces、seed 44 |
| formal validation | 全 287,933 traces |

対象 run directory は
[`20260829T075432Z_ee3d9e5_formal_50000_steps`](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T075432Z_ee3d9e5_formal_50000_steps/)
である。Git commit `ee3d9e5d5fce73e3ce0450b3471fe3284af616a1` をコード基準とし、
NVIDIA H100 NVL（`cuda:1`）で実行した。

### Formal result

| 項目 | 値 |
|---|---|
| run status | `success` |
| `oracle_per_trace_unit_rms_global_snr_db` | **`20.4604 dB`** |
| best step | `50000` |
| threshold margin | **`+0.4604 dB`** |
| `metric_success` | `true` |
| `scope_success` | `true` |
| `success` | **`true`** |
| checkpoint revalidation | 保存値 = 再計算値 = `20.4604 dB`、`revalidation_matches=true`（atol / rtol `1e-8`） |
| validation signal / error energy | `179958125.0042889` / `1618586.4733` |
| clean validation traces | `287933` |
| train audit | `20.7557 dB`、`287933` traces、seed 44 |

途中 checkpoint や学習曲線の外挿は成功判定に使わず、正常終了後の
`metrics.json` と `run.json` のみで確定した。学習中の全評価点は次のとおりである。

| Step | Formal validation S/N (dB) | 20 dB 判定 |
|---:|---:|---|
| 1 | -3.2054 | 未達 |
| 5,000 | 17.8862 | 未達 |
| 10,000 | 18.7106 | 未達 |
| 15,000 | 19.2885 | 未達 |
| 20,000 | 19.6402 | 未達 |
| 25,000 | 19.8833 | 未達 |
| 30,000 | 20.1082 | **初回達成** |
| 35,000 | 20.2611 | 達成 |
| 40,000 | 20.3675 | 達成 |
| 45,000 | 20.434 | 達成 |
| 50,000 | **20.4604** | **best / 達成** |

### 独立成果物監査

正式 pipeline の自動監査に加え、run 終了後に成果物と processed split を read-only で
再集計した。結果は次のとおりである。

- signal / error energy からの `10 log10(signal/error)` と保存 S/N、error mean square が一致。
- checkpoint payload 内の step / metric / model config と `metrics.json`、`run.json`、
  `inputs.lock.json` が一致。CPU load した 141 tensors、9,210,121 parameters は全て finite。
- 重複除去前は 4,780 FFID のそれぞれで train / validation / test が厳密に
  50% / 12.5% / 37.5% で、全 split が非空（FFID あたりの最小数は 56 / 14 / 42）。
  global canonicalization の 15 行除去後は、
  影響した 15 FFID のみ厳密比率からずれ、effective 全体の train 比率は
  `49.9999%` となる。
- 14 個の formal scope checks は全て true、学習曲線の 11 評価点は finite かつ
  S/N が単調増加。

この独立監査は保存成果物の整合性と processed split を再計算したもので、GPU で
287,933 validation traces の全再推論は繰り返していない。全再推論による checkpoint
revalidation 自体は正式 pipeline が実行し、その保存値と checkpoint payload を独立に
照合した。

## Run provenance

各 run directory に `config.resolved.yaml`、`inputs.lock.json`、`metrics.json`、
`run.json`(seed、Git SHA、環境、開始・終了時刻)、`artifacts/best.pt` を保存する。

| Run | 位置づけ |
|---|---|
| [`20260829T064550Z_2d1e8d9_stage09_full_receptive_width256_10000_steps`](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T064550Z_2d1e8d9_stage09_full_receptive_width256_10000_steps/) | 完了済み diagnostic 最良 |
| [`20260829T074408Z_3524037_stage15_combined_width384_gate_fir31`](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T074408Z_3524037_stage15_combined_width384_gate_fir31/) | formal architecture の短時間確認。K274、width 384、9,210,121 parameters |
| [`20260829T075432Z_ee3d9e5_formal_50000_steps`](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T075432Z_ee3d9e5_formal_50000_steps/) | 正式 run。drawn / unique training targets 4,800,000 / 1,134,025(effective train の 98.4627%、未抽出 17,706) |

補助 lag / ensemble probes はこの provenance 一式を持たないため、上記 run artifacts と
同格には扱わない。

## 再実行方法

リポジトリの root で実行する。外部 SEG-Y は manifest の SHA-256 と照合済みで
あることを前提とする。

### 50% per-FFID split の準備

新しい出力先へ準備する場合は次を実行する。

```bash
python -m seis_interp.cli data prepare-baseline \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_per_ffid_50pct_train_amplitude_qc \
  --config studies/study_018_all_ffid_50pct_neighbor_inpainter/config.yaml
```

生成済み出力を置換する場合だけ、対象を確認した上で `--overwrite` を明示する。

### 凍結 formal 条件の fresh run

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_formal_50000_steps"

python -m seis_interp.cli train neighbor-inpainter \
  --config studies/study_018_all_ffid_50pct_neighbor_inpainter/config.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_per_ffid_50pct_train_amplitude_qc \
  --output "runs/study_018_all_ffid_50pct_neighbor_inpainter/$RUN_ID" \
  --device cuda:1
```

既存 run directory を再利用せず、fresh run ID を使う。各 diagnostic を再現する場合は、
同じ command の `--config` を
[`variants/`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/variants/) 内の該当 YAML に
置き換える。

### 品質ゲート

`ruff check .`、`ruff format --check .`、`pytest`、`python -m seis_interp.cli doctor` は全て pass。

## 制約と解釈上の注意

- Stage 01–15 の短時間 diagnostic 最良は 18.2484 dB で閾値未満だったが、
  凍結後の formal 50,000-step run は 20.4604 dB で成功した。
- 同じ validation split を Stage の選択と各 run の checkpoint 選択に繰り返し使っている。
  Stage 15 / formal architecture の値には model-selection optimism があり得る。
- test target は一切評価していない。formal success は事前に定めた validation metric の
  成功であり、固定 test split での最終 generalization 証拠ではない。
- primary metric は target 自身の RMS を使う oracle waveform 指標である。実運用で未知 trace
  の物理振幅を復元するには、別の train-only gain model が必要である。
- 選択モデルは座標だけでなく train-only neighbor waveforms を条件とする。coordinate-only
  implicit field の成功を意味しない。
- 50% は FFID 内 trace density であり、未観測 FFID への extrapolation は検証していない。
- SEG C3 NA、seed 42、単一 split、単一 training seed での POC であり、別 seed・別 survey・
  別 acquisition geometry への再現性は未検証である。
- formal run は `cudnn_benchmark=true`、`cudnn_deterministic=false` であり、bitwise の
  完全再現を保証する条件ではない。
- run は Git commit を記録するが dirty-worktree flag や output hash manifest を内包しない。
- checkpoint は model と評価に必要な情報が中心で、optimizer、scheduler、RNG state を
  含まないため、完全な training resume 用ではない。
- 外部 SEG C3 NA の利用権は source manifest どおり利用前に別途確認が必要である。
- flat crossline aperture は悪化したが、crossline geometry 自体が不要とは断定できない。
  moveout-aware alignment や anisotropic gating は未検証である。
- train-only lag と ensemble の probe は独立した immutable artifact を持たず、診断用途に限る。
- Stage 07 と Stage 08 は複数因子を同時に変更したため、個別因子の因果効果には分解できない。

## 最終判断欄

**SUCCESS** — 正式 run は `oracle_per_trace_unit_rms_global_snr_db =
20.4604 dB` を記録し、厳密な `> 20.0 dB` 条件を
`+0.4604 dB` 上回った。checkpoint 再評価と formal scope checks は
全て通過し、`metric_success=true`、`scope_success=true`、`success=true` である。
