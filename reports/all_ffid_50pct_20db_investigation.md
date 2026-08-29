# 全 FFID・train 50% 条件で 20 dB を目指した段階実験レポート

- 実施日: 2026-08-29
- 対象データ: SEG C3 Narrow-Azimuth
- 対象 study: [`study_018_all_ffid_50pct_neighbor_inpainter`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/README.md)
- 正式な成功条件: `oracle_per_trace_unit_rms_global_snr_db > 20.0`
- 完了済み diagnostic 最良: Stage 09、`18.24842244059429 dB`
- 正式 run: `20260829T075432Z_ee3d9e5_formal_50000_steps`
- 正式結果: **`20.460355529598864 dB`、成功**

## 結論

50% 条件で完了している Stage 01–15 は、全 4,780 eligible FFID、全 287,933
validation traces、リーク監査を含む同じ formal scope で比較した。事前の
diagnostic 最良は Stage 09 の **18.24842244059429 dB** で、この時点では
20 dB まで **1.7515775594057104 dB** 残っていた。

切り分けでは、source-x 座標、同一 source-x line 内の K274 aperture、FiLM、全 trace を
覆う temporal receptive field、width 384、target-coordinate neighbor gate、軽量な
neighbor-wise alignment FIR が正の差分を示した。最大の改善要因は学習 budget であり、
width 256 条件を 2,500 から 10,000 updates へ伸ばすと
`+1.8818672636854856 dB` 改善し、最終 checkpoint でも上昇中だった。これらをまとめた
Stage 15 architecture を 50,000 updates 学習する条件を formal candidate として凍結した。

その fresh formal run は 30,000 step で初めて 20 dB を超え、50,000 step の best
checkpoint で **20.460355529598864 dB** に到達した。厳密な閾値への余裕は
**+0.460355529598864 dB** である。checkpoint 再評価は保存値と完全一致し、
formal scope checks も全て通過したため、指定された成功条件を満たす。

> **Formal result: SUCCESS — `20.460355529598864 dB > 20.0 dB`**

## 「train ratio 50% FFID」の解釈

本実験では、ユーザー指定の「train ratio 50% FFID」を、先行 Study 017 の
`split_scope: per_ffid` と trace interpolation の研究目的に沿って、**各 eligible FFID
内部で amplitude-eligible trace の 50% を train に割り当てる**条件と解釈した。

- seed 42 の deterministic whole-trace permutation を FFID ごとに適用する。
- 50% を train、残る 50% を holdout とする。
- Study 017 の holdout 内比率を維持し、holdout の 25% を validation、75% を test にする。
- survey 全体では canonicalization 前に train / validation / test が
  50% / 12.5% / 37.5% になる。
- 同じ trace の time samples を複数 split に分けない。
- 全 4,780 eligible FFID を train、validation、test の対象に残す。

これは FFID の半数を丸ごと train、残りを丸ごと holdout にする条件ではない。後者は
未観測 FFID への shot extrapolation を検証する別の研究課題であり、本報告の成功条件には
含めない。この解釈と理由は [`decisions.md`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/decisions.md)
に固定した。

## 成功指標と判定規則

validation trace ごとに target 自身の RMS で unit-RMS 化し、モデルの raw prediction と
比較する。prediction の再 unit-RMS 化は診断値に限り、checkpoint 選択や成功判定には
使わない。

```text
target_unit[i] = target[i] / RMS(target[i])

oracle_per_trace_unit_rms_global_snr_db
  = 10 log10(
      sum_i,t(target_unit[i,t]^2)
      / sum_i,t((target_unit[i,t] - prediction_raw[i,t])^2)
    )

success
  ⇔ oracle_per_trace_unit_rms_global_snr_db > 20.0
     AND formal scope checks are all true
```

比較は厳密な `>` であり、20.0 dB ちょうどは失敗とする。20 dB は、この評価領域で
global error energy が signal energy の 1% 未満であることに対応する。

## データ、split、canonicalization

入力契約は [`inputs.yaml`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/inputs.yaml)、
実行条件は [`config.yaml`](../studies/study_018_all_ffid_50pct_neighbor_inpainter/config.yaml)
に固定した。振幅品質条件は all-zero trace の除外と `max_abs_amplitude <= 10000.0` である。
観測 FFID 2–4782 のうち、FFID 1746 の 544 traces は全て amplitude QC で
`excluded` となり、残る 4,780 FFID が eligible である。全 trace は 625 samples を持つ。

| Split | 50% split 準備後 | Canonicalization 除去 | Formal effective count |
|---|---:|---:|---:|
| train | 1,151,740 | 9 | **1,151,731** |
| validation | 287,935 | 2 | **287,933** |
| test | 863,805 | 4 | **863,801** |
| eligible 合計 | 2,303,480 | 15 | **2,303,465** |

50% は canonicalization 前の amplitude-eligible trace に対する割当比である。重複行を
split ごとに 9 / 2 / 4 行除いたため、formal effective train count は総数の厳密な半分から
わずかにずれる。成功判定は上表の effective counts との完全一致を要求する。

全 survey には同じ物理座標
`[source_x_m, source_y_m, receiver_x_m, receiver_y_m]` を持つ cell が 15 個、該当行が
30 行あった。split や振幅値を参照せず、各 cell で最小 `array_row` の行だけを残す
global canonicalization を適用した。これにより、validation target と同じ物理 cell の
twin trace が train neighbor に入る経路を閉じた。

### 入力 source lock

| Source | SHA-256 |
|---|---|
| `SEG_C3NA_ffid_2-1200.sgy` | `8913f18606e21b116280755cf7367fee90fea05b06dc7c85d919ae17a1c9c4a9` |
| `SEG_C3NA_ffid_1201-2400.sgy` | `50c4ea99349375fb8b0a400ec860b16db82583e3eb132accd405c42f266099c7` |
| `SEG_C3NA_ffid_2401-3600.sgy` | `8b14ab894bc78991aebb14d96f49f0ecd9dc492ca1c993e60ebcc1dfcb741aeb` |
| `SEG_C3NA_ffid_3601-4781.sgy` | `16a66c1e165da86d98186e4ff13510678291b484a64cdae549b5336932b9d1e9` |

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
| [01](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T055137Z_ee73dba_stage01_baseline/metrics.json) | Study 017 model を 50% split へそのまま移植、3 target coords | 104 / 128 / 2,500 | 15.375973554927757 | 基準 | baseline |
| [02](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T060524Z_251ccb8_stage02_source_x_coordinate/metrics.json) | source-x を追加した 4 coords、staggered multiline geometry | 104 / 128 / 2,500 | 15.593087315175396 | Stage 01 比 `+0.2171137602476385` | 採用 |
| [03](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T060845Z_a9aa723_stage03_receiver_aperture/metrics.json) | 同一 source-x line の receiver aperture を K274 へ拡大 | 274 / 128 / 2,500 | 15.702883799604487 | Stage 02 比 `+0.1097964844290917` | 採用 |
| [04](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T061249Z_4db20f5_stage04_crossline_aperture/metrics.json) | source-x line radius 1 の flat crossline aperture | 272 / 128 / 2,500 | 15.347270702609082 | Stage 02 比 `-0.2458166125663137` | 不採用 |
| [05](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T061731Z_655bcbb_stage05_receiver_aperture_10000_steps/metrics.json) | Stage 03 を 10,000 updates へ延長 | 274 / 128 / 10,000 | 17.424001143453978 | Stage 03 比 `+1.7211173438494907` | budget 延長を採用 |
| [06](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T062603Z_3917d7a_stage06_receiver_aperture_film/metrics.json) | 各 temporal block を 4 target coords で FiLM conditioning | 274 / 128 / 2,500 | 15.775978343484828 | Stage 03 比 `+0.0730945438803410` | 採用 |
| [07](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T063112Z_41301b6_stage07_full_receptive_width256/metrics.json) | width 256、13 blocks、full-trace receptive field | 274 / 256 / 2,500 | 16.366555176908804 | Stage 06 比 `+0.5905768334239756` | bundle を採用 |
| [08](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T063843Z_f8ef160_stage08_pure_mse_no_dropout/metrics.json) | derivative weight 0、neighbor dropout 0 | 274 / 256 / 2,500 | 16.355473474147036 | Stage 07 比 `-0.0110817027617678` | bundle を不採用 |
| [09](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T064550Z_2d1e8d9_stage09_full_receptive_width256_10000_steps/metrics.json) | Stage 07 を 10,000 updates へ延長 | 274 / 256 / 10,000 | **18.24842244059429** | Stage 07 比 `+1.8818672636854856` | budget 延長を採用、完了済み最良 |
| [10](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T070218Z_fdacfa0_stage10_receiver_aperture_epoch_sampling/metrics.json) | Stage 05 の target sampling を epoch without replacement 化 | 274 / 128 / 10,000 | 17.422069591085755 | Stage 05 比 `-0.0019315523682231` | 改善なし、不採用 |
| [11](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T071037Z_fdacfa0_stage11_stem_kernel31/metrics.json) | Stage 07 の stem kernel を 15→31 | 274 / 256 / 2,500 | 16.16620796677095 | Stage 07 比 `-0.2003472101378527` | 不採用 |
| [12](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T071831Z_186bf32_stage12_target_coordinate_neighbor_gate/metrics.json) | target-coordinate masked-softmax neighbor gate | 274 / 256 / 2,500 | 16.41721085832344 | Stage 07 比 `+0.0506556814146357` | 採用 |
| [13](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T072453Z_d1227f6_stage13_width384/metrics.json) | Stage 07 の width を 256→384 | 274 / 384 / 2,500 | 16.602279113641742 | Stage 07 比 `+0.2357239367329385` | 採用 |
| [14](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T073548Z_645d4d4_stage14_neighbor_alignment_fir31/metrics.json) | identity 初期化した per-neighbor depthwise FIR、kernel 31 | 274 / 256 / 2,500 | 16.49749605788384 | Stage 07 比 `+0.1309408809750359` | 採用 |
| [15](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T074408Z_3524037_stage15_combined_width384_gate_fir31/metrics.json) | width 384 + gate + FIR31 を結合 | 274 / 384 / 2,500 | **16.80368309012617** | Stage 07 比 `+0.4371279132173669` | formal architecture に採用 |

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
| 1 | 0.8963412578127673 | 1.2097423161228786 |
| 2,500 | 15.771058734345157 | 16.269681881414854 |
| 5,000 | 16.69879313603435 | 17.421843884545737 |
| 7,500 | 17.230025759943107 | 18.02191998718019 |
| 10,000 | **17.424001143453978** | **18.24842244059429** |

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

全 validation 287,933 traces から seed 44 で 500 traces を抽出した。478 FFID を含み、row
集合の SHA-256 は
`8f8c0bf87f7472004ed71921188def02078c035e81173d0bd04287133a39d4c2` である。各 target の
K274 input が train-only であることを assert し、Stage 05 / 07 / 08 best checkpoint の
raw output を CPU float32 で再計算した。

| Prediction | 500-trace sample S/N (dB) | Full-validation S/N (dB) | Sample best との差 |
|---|---:|---:|---:|
| Stage 05 | **17.60536** | 17.424001143453978 | 0.00000 |
| Stage 07 | 16.46565 | 16.366555176908804 | -1.13971 |
| Stage 08 | 16.47349 | 16.355473474147036 | -1.13187 |
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
| `oracle_per_trace_unit_rms_global_snr_db` | **`20.460355529598864 dB`** |
| best step | `50000` |
| threshold margin | **`+0.460355529598864 dB`** |
| `metric_success` | `true` |
| `scope_success` | `true` |
| `success` | **`true`** |
| checkpoint revalidation | 保存値 = 再計算値 = `20.460355529598864 dB`、`revalidation_matches=true`（atol / rtol `1e-8`） |
| validation signal / error energy | `179958125.0042889` / `1618586.4732541414` |
| clean validation traces | `287933` |
| train audit | `20.75565366696424 dB`、`287933` traces、seed 44 |

途中 checkpoint や学習曲線の外挿は成功判定に使わず、正常終了後の
`metrics.json` と `run.json` のみで確定した。学習中の全評価点は次のとおりである。

| Step | Formal validation S/N (dB) | 20 dB 判定 |
|---:|---:|---|
| 1 | -3.205396415623824 | 未達 |
| 5,000 | 17.886170882558204 | 未達 |
| 10,000 | 18.710620881540358 | 未達 |
| 15,000 | 19.288523315019916 | 未達 |
| 20,000 | 19.640234529910284 | 未達 |
| 25,000 | 19.883258883479705 | 未達 |
| 30,000 | 20.10820274905015 | **初回達成** |
| 35,000 | 20.261123524291797 | 達成 |
| 40,000 | 20.367518873776586 | 達成 |
| 45,000 | 20.43403178720546 | 達成 |
| 50,000 | **20.460355529598864** | **best / 達成** |

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
  `49.99993488071232%` となる。
- 14 個の formal scope checks は全て true、学習曲線の 11 評価点は finite かつ
  S/N が単調増加。

この独立監査は保存成果物の整合性と processed split を再計算したもので、GPU で
287,933 validation traces の全再推論は繰り返していない。全再推論による checkpoint
revalidation 自体は正式 pipeline が実行し、その保存値と checkpoint payload を独立に
照合した。

## 実装した再利用機能

### Geometry、model、checkpoint

- [`multiline_neighbor_geometry.py`](../src/seis_interp/processing/multiline_neighbor_geometry.py):
  staggered source lattice、source-x / source-y half-shot / receiver aperture、train-only lookup、
  collision audit
- [`neighbor_trace_inpainter.py`](../src/seis_interp/models/neighbor_trace_inpainter.py):
  可変 target-coordinate 数、可変 stem / residual kernel、可変 dilation、FiLM、
  target-coordinate masked-softmax gate、identity-initialized depthwise alignment FIR
- [`neighbor_inpainter_checkpoints.py`](../src/seis_interp/training/neighbor_inpainter_checkpoints.py):
  新しい model 条件の保存・strict load と Study 017 checkpoint の backward compatibility

### Training と pipeline

- [`neighbor_inpainter_trainer.py`](../src/seis_interp/training/neighbor_inpainter_trainer.py):
  可変 target-coordinate batch と既存の loss / validation 経路
- [`train_neighbor_inpainter.py`](../src/seis_interp/pipelines/train_neighbor_inpainter.py):
  legacy single-line と multiline geometry の切替、config-driven model 構築、
  replacement / epoch-without-replacement target sampling、sampling provenance、formal scope
- [`test_study_018_50pct_contract.py`](../tests/integration/test_study_018_50pct_contract.py):
  50% split counts、strict 20 dB、凍結 formal architecture / budget の契約
- [`test_multiline_neighbor_geometry.py`](../tests/unit/test_multiline_neighbor_geometry.py):
  stagger parity、offset order、center 除外、train-only lookup、coordinate scaling の検証
- [`test_neighbor_target_sampling.py`](../tests/unit/test_neighbor_target_sampling.py):
  epoch sampler の permutation、wrap、seed、unique-draw accounting の検証
- [`test_neighbor_trace_inpainter.py`](../tests/unit/test_neighbor_trace_inpainter.py):
  FiLM、gate、alignment FIR、shape、初期化の検証

主要 commit は次のとおりである。

| Commit | 内容 |
|---|---|
| `ee73dba` | Study 018、50% per-FFID split、configurable formal scope を定義 |
| `251ccb8` | multiline staggered-source geometry と可変 model 条件 |
| `3917d7a` | temporal block FiLM |
| `45addba` | epoch target sampling と sampling provenance |
| `ae789e2` | target-coordinate masked-softmax neighbor gate |
| `e232cae` | identity-initialized neighbor-wise depthwise FIR |
| `3524037` | Stage 15 の正の model 変更を結合 |
| `ee3d9e5` | 50,000-step formal 条件を凍結 |

## Run provenance

Stage run は `config.resolved.yaml`、`inputs.lock.json`、seed、Git SHA、environment、metric、
scope audit、checkpoint を run directory に保存する。代表として、完了済み最良 Stage 09 と
formal architecture の短時間確認 Stage 15 の artifact hash を示す。

### Stage 09

- Run: [`20260829T064550Z_2d1e8d9_stage09_full_receptive_width256_10000_steps`](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T064550Z_2d1e8d9_stage09_full_receptive_width256_10000_steps/)
- Git commit: `2d1e8d9`
- 開始 / 終了: `2026-08-29T06:45:56Z` / `2026-08-29T07:02:12Z`
- device: NVIDIA H100 NVL、`cuda:1`
- drawn / unique training targets: 960,000 / 651,295

| Artifact | SHA-256 |
|---|---|
| `config.resolved.yaml` | `bdc5f15ac3726a9ee0a45dfca9c3f086f1a799fd7fadc7b9fc5742b6c48e9471` |
| `inputs.lock.json` | `f683e96bdf81b1c3f7701b0dec253668f018a7ed8a342be20172ab4b4d84280a` |
| `metrics.json` | `3fa36143e0379552f1d57bfdf967d814c4ad398c5397d707dae6c350e6de2b77` |
| `run.json` | `ab86eb9c645b88d66c91a23638f59b88e6d3a79452f1c651af7c69349b8d6abb` |
| `artifacts/best.pt` | `2ad30d56872806a3d8b155556bcf41ea00c48af1ab42ad5f8c7d0dce2f8a8456` |

### Stage 15

- Run: [`20260829T074408Z_3524037_stage15_combined_width384_gate_fir31`](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T074408Z_3524037_stage15_combined_width384_gate_fir31/)
- Git commit: `3524037dd631067a97846e88f9166b572362dc97`
- 開始 / 終了: `2026-08-29T07:44:16Z` / `2026-08-29T07:52:37Z`
- model: K274、width 384、9,210,121 parameters

| Artifact | SHA-256 |
|---|---|
| `config.resolved.yaml` | `4d87df8a46ccc17e84acb8c1e5204d49078c2ce329406a77b568dd5130c4413b` |
| `inputs.lock.json` | `e428eda4ff99d4a66ef02d36496b3ba9682a946c3a75d5e5f003a44ecce7e0ae` |
| `metrics.json` | `3613b3e0eb86af49132156fc404683ad159d0b80dba17c2e54058fd6c95b5176` |
| `run.json` | `603be999750b0286dd04e42e844b7d693f5c773e2fb8e95368335a0a6ee3b8c4` |
| `artifacts/best.pt` | `ac43b986292fc296801a759aaaf848edb638e3f672d5bddc9efad80813db266b` |

### Formal 50,000-step run

- Run: [`20260829T075432Z_ee3d9e5_formal_50000_steps`](../runs/study_018_all_ffid_50pct_neighbor_inpainter/20260829T075432Z_ee3d9e5_formal_50000_steps/)
- Git commit: `ee3d9e5d5fce73e3ce0450b3471fe3284af616a1`
- 開始 / 終了: `2026-08-29T07:54:41Z` / `2026-08-29T09:20:59Z`（1 時間 26 分 18 秒）
- device: NVIDIA H100 NVL、`cuda:1`
- drawn / unique training targets: 4,800,000 / 1,134,025（effective train の 98.46266185420033%、
  未抽出 17,706）
- peak CUDA allocated / reserved: 12,434,698,240 / 21,934,112,768 bytes
- process max RSS: 13,426,828 KiB

| Artifact | SHA-256 |
|---|---|
| `config.resolved.yaml` | `f4be4bb4e173e8ec1a4a9ac284439bab19908496fb93b5872f686547cd8da2b9` |
| `inputs.lock.json` | `35af5da1d7fddc1c0b560980bb2c4668d08e8921ca6b3f0652945da3fe3cb4c4` |
| `metrics.json` | `c71ddf5a44d7e8e0aa4683747e2c4f175fe74fe1fec847c2c021bec75e3481b1` |
| `run.json` | `a87a13bfacbf7bf31ebd3107bacb518b1041eb6b5ffd3b3925c261f286089c29` |
| `artifacts/best.pt` | `ae2b3d784c2a02a13169db798e149d7ed10171675cf278c2e376199da947121f` |

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

### Repository quality gates

正式結果と Study 完了状態を反映した 2026-08-29 の worktree で、次を全て
実行した。

```bash
ruff check .
ruff format --check .
pytest
python -m seis_interp.cli doctor
```

| Gate | 実測結果 |
|---|---|
| `ruff check .` | pass、`All checks passed!` |
| `ruff format --check .` | pass、`192 files already formatted` |
| `pytest` | pass、`1099 passed in 37.77s` |
| `python -m seis_interp.cli doctor` | exit 0 |

doctor は Python 3.10.12、PyTorch `2.5.0a0+b465a5843b.nv24.09`、CUDA available、
NVIDIA H100 NVL 2 台、segyio 1.9.14、および設定済み data root の exists /
readable がいずれも true であることを確認した。ホスト固有の絶対 path は記録しない。

## 制約と解釈上の注意

- Stage 01–15 の短時間 diagnostic 最良は 18.24842244059429 dB で閾値未満だったが、
  凍結後の formal 50,000-step run は 20.460355529598864 dB で成功した。
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
  本報告の artifact SHA-256 は run 終了後に計算した。
- checkpoint は model と評価に必要な情報が中心で、optimizer、scheduler、RNG state を
  含まないため、完全な training resume 用ではない。
- 外部 SEG C3 NA の利用権は source manifest どおり利用前に別途確認が必要である。
- flat crossline aperture は悪化したが、crossline geometry 自体が不要とは断定できない。
  moveout-aware alignment や anisotropic gating は未検証である。
- train-only lag と ensemble の probe は独立した immutable artifact を持たず、診断用途に限る。
- Stage 07 と Stage 08 は複数因子を同時に変更したため、個別因子の因果効果には分解できない。

## 最終判断欄

**SUCCESS** — 正式 run は `oracle_per_trace_unit_rms_global_snr_db =
20.460355529598864 dB` を記録し、厳密な `> 20.0 dB` 条件を
`+0.460355529598864 dB` 上回った。checkpoint 再評価と formal scope checks は
全て通過し、`metric_success=true`、`scope_success=true`、`success=true` である。
