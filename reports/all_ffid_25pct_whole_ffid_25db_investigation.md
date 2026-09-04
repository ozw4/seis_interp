# FFIDを25%選択する条件で25 dBを目指した段階実験レポート

> 対象 study は
> [`study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter`](../studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/README.md)
> である。

## 結論

- 対象: SEG C3 Narrow-Azimuth、全amplitude-eligible FFID
- train条件: FFIDを丸ごと1,195 / 4,780個選択（正確に25%）
- 正式成功条件: `oracle_per_trace_unit_rms_global_snr_db > 25.0`
- 最良結果: **`9.0998 dB`**（Stage 07、K1374、6,030 step）
- 閾値までの不足: **`15.9002 dB`**
- 判定: **未達**（`metric_success=false`、`scope_success=true`、`success=false`）

ここでの「FFIDを25%選ぶ」は、amplitude-eligible FFID集合から25%のFFIDを選び、そのFFIDに
属するeligible traceをすべてtrainへ入れる条件を指す。各FFID内でtraceを25%ずつ分ける条件は
[`study_019_all_ffid_25pct_neighbor_inpainter`](../studies/study_019_all_ffid_25pct_neighbor_inpainter/README.md)
が扱う別scopeである。
近傍被覆、crossline、source-y範囲、shot-bracketing referenceを切り分けた後、sampler、
学習量、whole-shotモデル、source表現、受容野、容量、receiver条件付け、距離重み、lossを
単独要因として継続評価し、dynamic source重みと昇格要因の組合せまで確認した。Stage 07は
Stage 03から`+0.3798 dB`改善したが、
25 dBには誤差energyをさらに約38.9分の1へ減らす必要がある。

最良runは
[`20260831T072719Z_7343bb0_stage07_full_train_sweep_k1374_6030_steps`](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T072719Z_7343bb0_stage07_full_train_sweep_k1374_6030_steps/)
である。

## 契約(split・評価)

split 契約と acceptance criteria の正本は
[`study_020` README](../studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/README.md)、
processed 入力契約は
[`inputs.yaml`](../studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/inputs.yaml)
である。本報告が依拠する要点:

- `split_scope: whole_ffid`、seed 42。train / validation / test = 1,195 / 896 / 2,689 FFID、
  重複 0、各 FFID の所属 split は最大 1。
- effective train / validation / test traces = 578,685 / 437,087 / 1,287,693
  (入力 2,304,024 のうち 544 を amplitude QC で除外、duplicate physical cell 15 個から 15 行除去)。
  残存 duplicate、train geometry collision、train-validation 物理座標 overlap はいずれも 0。
- train trace と validation target をそれぞれ自身の RMS で unit-RMS 化し、予測は後処理で
  再正規化せず raw model output と比較する。

```text
success
  iff oracle_per_trace_unit_rms_global_snr_db > 25.0
  and all formal scope/leakage checks are true
```

比較は厳密な `>` であり、25.0 dB ちょうどは失敗である。validation を checkpoint 選択と
切り分け選択に使い、test target は事前契約どおり参照しない。

## 段階実験の設計

すべてfresh initialization、同じseedと評価domainを使う。原則2,500 updateとし、budgetを
単独要因にするStage 07/19/22だけを延長する。Stage 01から03は有限apertureの被覆だけを段階的に
変える。Stage 04はStage 01のK274へ戻し、source方向に整合したprediction referenceだけを
追加する。

| Stage | 単独で変える条件 | local K | validation zero-neighbor |
|---:|---|---:|---:|
| 01 | Study 018 architectureをwhole-FFID条件へ移植 | 274 | 132,336（30.2768%） |
| 02 | source-x line radius `0 -> 1` | 714 | 15,560（3.5599%） |
| 03 | source-y half-shot radius `4 -> 8` | 1,374 | 0 |
| 04 | K274 + exact-receiver shot-bracketing reference | 274 + reference 1 | localはStage 01と同じ |
| 05 | promoted K1374 + shot-bracketing reference | 1,374 + reference 1 | 0 |

geometry-onlyのbracketing監査では、train 578,685行のうち524,285行、validation
437,087行のうち397,535行が両側bracketを持った。残り54,400 / 39,552行は片側nearestで、
未解決、non-train source、target-FFID source、same-source-y sourceはすべて0だった。
比較診断ではnearest-shotコピーが4.3999 dB、線形bracketing単体が
5.4785 dBだった。これは独立した正式runではなく、Stage 04を選ぶための診断値である。

Stage 05は、Stage 03とStage 04がそれぞれStage 01比`+0.20 dB`以上かつ全scope check合格の
場合だけ実行する。両者は有限aperture coverageとlong-range shot referenceという異なる
要因なので、独立効果を確認してから組み合わせる。

## 段階実験の結果

Stage 01から05は全full-scope check、checkpoint再評価、test/excluded非参照監査に合格し、
best checkpointはいずれもstep 2,500だった。

| Stage | local K / width / steps | Validation S/N | 主比較差 | Stage 01差 |
|---:|---:|---:|---:|---:|
| [01](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T044414Z_f234880_stage01_k274_whole_ffid_2500_steps/metrics.json) | 274 / 384 / 2,500 | 4.4312 dB | baseline | 基準 |
| [02](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T045444Z_31f81ca_stage02_crossline_k714_2500_steps/metrics.json) | 714 / 384 / 2,500 | 7.7835 dB | Stage 01比 +3.3523 dB | +3.3523 dB |
| [03](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T050923Z_31f81ca_stage03_crossline_k1374_2500_steps/metrics.json) | 1,374 / 384 / 2,500 | **8.72 dB** | Stage 02比 +0.9364 dB | **+4.2887 dB** |
| [04](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T052831Z_610c307_stage04_k274_source_bracketing_residual_2500_steps/metrics.json) | 274 + ref 1 / 384 / 2,500 | 8.5133 dB | Stage 01比 +4.0821 dB | +4.0821 dB |
| [05](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T053909Z_f0a67c8_stage05_crossline_k1374_source_bracketing_residual_2500_steps/metrics.json) | 1,374 + ref 1 / 384 / 2,500 | 8.596 dB | Stage 03比 -0.124 dB | +4.1647 dB |

K274からcrossline K714への変更はvalidation zero-neighborを30.28%から3.56%へ減らし、
3.35 dBの大きな改善を得た。K1374はzero-neighborを0にし、さらに0.94 dB改善した。
したがってwhole-FFID条件ではcrosslineを含むsource coverageが主要因の一つである。一方、
被覆を完全化しても25 dBとの差は16.28 dB残り、有限aperture拡大だけでは
目標へ届かなかった。

Stage 04はbracketing単体に近いstep 1の5.4811 dBから、2,500 stepで
8.5133 dBまで改善した。K274との比較では明確に有効だが、K1374単独より
0.2066 dB低かった。Stage 05はStage 04比では0.0827 dB改善したが、
Stage 03比では0.124 dB悪化した。完全被覆contextと固定bracketing referenceの
単純な組合せに正の相乗効果はなかった。

## 継続実験の設計と結果

Stage 06以降もsplit、primary metric、success ruleを変更していない。Stage 06--08は最良の
K1374 trace model周辺、Stage 09以降はwhole-shot欠損に合わせたjoint 8 x 68 receiver-grid
model周辺を単独要因で比較した。全完了runで`scope_success=true`、checkpoint全validation
再計算一致、`success=false`だった。

| Stage | 単独条件 | Parameters / steps | Validation S/N | 対応baseline差 |
|---:|---|---:|---:|---:|
| [06](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T070900Z_7343bb0_stage06_epoch_sampling_k1374_2500_steps/metrics.json) | Stage 03 + epoch sampler | 21,921,721 / 2,500 | 8.7156 dB | Stage 03比 -0.0044 dB |
| [07](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T072719Z_7343bb0_stage07_full_train_sweep_k1374_6030_steps/metrics.json) | Stage 06 + TRAIN一巡 | 21,921,721 / 6,030 | **9.0998 dB** | Stage 03比 +0.3798 dB |
| [08](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T070900Z_7343bb0_stage08_crossline_k1374_bracketing_channels_2500_steps/metrics.json) | lower/upper bracket別channel | 21,944,833 / 2,500 | 8.4743 dB | Stage 03比 -0.2456 dB |
| [09](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T073639Z_b43afe9_stage09_joint_shot_gather_k8_2500_steps/metrics.json) | joint shot K8 control | 29,409 / 2,500 | 6.7827 dB | control |
| [10](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T074426Z_ba9c33f_stage10_joint_shot_gather_receiver_y_dilation_2500_steps/metrics.json) | receiver-y dilation | 29,409 / 2,500 | 6.7757 dB | Stage 09比 -0.007 dB |
| [11](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T075054Z_1ba324b_stage11_joint_shot_gather_ordered_raw_k8_2500_steps/metrics.json) | ordered raw K8 | 37,025 / 2,500 | 6.8 dB | Stage 09比 +0.0173 dB |
| [12](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T075600Z_3b7b84d_stage12_joint_shot_gather_width128_2500_steps/metrics.json) | width 32→128 | 387,969 / 2,500 | 7.0106 dB | Stage 09比 +0.2279 dB |
| [13](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T075623Z_d05e89a_stage13_joint_shot_gather_full_temporal_field_2500_steps/metrics.json) | temporal RF 51→767 | 52,321 / 2,500 | 6.8193 dB | Stage 09比 +0.0367 dB |
| [14](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T080638Z_b6b516e_stage14_joint_shot_gather_width128_receiver_film_2500_steps/metrics.json) | Stage 12 + cell FiLM | 1,362,817 / 2,500 | 7.0281 dB | Stage 12比 +0.0176 dB |
| [16](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T081158Z_79904e3_stage16_joint_shot_gather_distance_power2_2500_steps/metrics.json) | distance power 1→2 | 29,409 / 2,500 | 6.9982 dB | Stage 09比 +0.2155 dB |
| [17](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T080638Z_a9476bb_stage17_joint_shot_gather_primary_mse_2500_steps/metrics.json) | derivative weight 0.1→0 | 29,409 / 2,500 | 6.7794 dB | Stage 09比 -0.0033 dB |
| [18](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260901T011553Z_a9476bb_stage18_joint_shot_gather_no_neighbor_dropout_2500_steps/metrics.json) | neighbor dropout 0.05→0 | 29,409 / 2,500 | 6.7828 dB | Stage 09比 +0.0001 dB |
| [19](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260901T005815Z_ddb31dc_stage19_joint_shot_gather_width128_five_sweeps_6000_steps/metrics.json) | Stage 12 + TRAIN五巡 | 387,969 / 6,000 | 7.4092 dB | Stage 12比 +0.3986 dB |
| [20](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260901T005815Z_9eae71a_stage20_joint_shot_gather_width128_distance_power2_2500_steps/metrics.json) | Stage 12 + distance power 2 | 387,969 / 2,500 | 7.2845 dB | Stage 12比 +0.2739 dB |
| [21](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260901T013424Z_7a5cd93_stage21_joint_shot_gather_dynamic_attention_2500_steps/metrics.json) | dynamic source attention | 29,746 / 2,500 | 7.0227 dB | Stage 09比 +0.24 dB |
| [22](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260901T013424Z_7a5cd93_stage22_joint_shot_gather_width128_distance_power2_five_sweeps_6000_steps/metrics.json) | Stage 20 + TRAIN五巡 | 387,969 / 6,000 | 7.7007 dB | Stage 20比 +0.4162 dB |

Stage 07はstep 3,015で8.9709 dB、step 6,030で9.0998 dBだった。
後半3,015 stepの利得は0.1289 dBまで縮小した。training auditは
11.9772 dBで、validationとの差も残る。単純なsampler変更だけのStage 06は
Stage 03と同等であり、利得は全TRAIN targetを実際に一巡したbudgetから得た。

joint-shot controlのstep 0 TRAIN-only IDWは6.2263 dB、学習後は
6.7827 dBだった。Stage 09--20では幅128と距離二乗だけがmatched control比
`+0.20 dB`を超えた。
receiver-y範囲、source保持、全時間RF、receiver-cell FiLM、純MSEはいずれも単独の主要因では
なく、dropout除去も+0.0001 dBだった。Stage 20では幅128と距離二乗の組合せが
各単独条件を上回り、Stage 12比+0.2739 dBの正の相互作用を確認した。Stage 19は
3,000 stepの7.1443 dBから6,000 stepの7.4092 dBへさらに
+0.2648 dB改善したため、距離二乗との同budget組合せをStage 22へ昇格する。
Stage 15 K16はformal実行前に、後述の全validation geometry診断でK8より悪いと確認して
棄却したため欠番を維持した。

Stage 21のdynamic attentionも単独昇格基準を超えたが、Stage 20へその利得を全加算する
楽観予測は7.5245 dBで、現在bestを1.5753 dB下回る。観測済みの
width×distance interactionを加えても逆転しないため、attentionをさらに組み合わせる
Stage 23は実行しない。

Stage 22はstep 3,000の7.4101 dBからstep 6,000の
7.7007 dBへ+0.2906 dB改善し、Stage 20比では
+0.4162 dBだった。joint-shot系列の最良だがStage 07を1.3991 dB
下回り、25 dBとの差は17.2993 dB残った。

## Train-only・上限診断

正式runを増やす前に、予測器の構築にvalidation振幅を使わないgeometry/reference診断と、
到達可能性を判定するtarget-derived oracle診断を分けて実施した。

| TRAIN-only reference（全validation） | S/N |
|---|---:|
| same-line linear | 5.479 dB |
| 2D Delaunay linear | 5.496 dB |
| K8 inverse-distance power 1 | 6.2263 dB |
| K8 inverse-distance power 2 | **6.4434 dB** |
| K16 best | 6.4116 dB |
| K32 best | 6.3439 dB |

source gridは160 m間隔の50 line、line内は原則80 m、隣接lineはsource-yが40 m staggerする。
K8でvalidation全receiver cellが少なくとも1 TRAIN sourceに被覆され、97.8%はsource-y両側、
93.3%はsource-x両側を持つ。したがってK16/K32追加は被覆不足の解消ではなく遠い波形の混入に
なり、Stage 15を棄却した。

| K8 power-2の時間帯 | S/N | signal比 | error比 |
|---|---:|---:|---:|
| 0--1 s | 10.12 dB | 47.87% | 20.53% |
| 1--2 s | 5.97 dB | 44.42% | 49.51% |
| 2--3 s | 0.65 dB | 5.64% | 21.42% |
| 3--4 s | 0.31 dB | 1.41% | 5.78% |
| 4--5 s | 0.26 dB | 0.67% | 2.76% |

2--5 sはsignalの7.72%に対してerrorの29.96%を占めた。relative receiver-yのfar / middle /
near帯はそれぞれ5.17 / 6.11 / 8.38 dBで、far側23 cellがerrorの40.09%を占めた。
receiver位置への非定常性は明確だが、Stage 10/14から単純な空間RFや固定cell FiLMだけでは
解消しない。

同一絶対receiverへ揃えたraw IDWは代表50,000 traceで約-3.2 dB、exact-CMP K4 IDWは
-1.69 dB、source-time / 4D POCSは約4.7--5.0 dB、MSSA low-rank投影もbaselineを悪化させた。
FFT/STFT phase interpolation、global delay、train-only KRR/PCA coefficient regressionも
5--7 dB台で棄却した。target-oracle shift+gainをK8へ与えても改善は約0.38 dB、best shiftは
median/p90が0 sample、p95が1 sampleで、単一static shiftが主要因ではなかった。

target-derived local linear spanはK64 / K128 / K256で約14.18 / 18.09 / 23.11 dB、利用可能な
384--512近傍まで増やしたsample ceilingは約23.36 dBだった。これは非線形whole-shotモデルの
数学的上限ではないが、target自身で係数を最適化する有利な診断でも25 dB未満である。
現在bestのerror MSE 0.123を25 dB相当0.0032未満へ下げるには、
error energyをさらに約38.9分の1へ減らす必要がある。

## 初回budget停止と限定的な再開

同系統のStudy 018では、同じformal architectureが2,500 stepの
16.8037 dBから50,000 stepの20.4604 dBへ改善し、実測利得は
3.6567 dBだった。この利得をそのまま加えて25 dBへ届くための2,500-step
昇格基準は`21.3433 dB`である。

当時の最良Stage 03はこの基準を12.6234 dB下回った。別条件のStudy 019で観測した
width 384→512の利得も2,500 stepで0.2157 dBに留まったため、trace modelを
10,000 / 50,000 stepへ機械的に伸ばす判断は停止した。その後、samplerと実際のTRAIN一巡を
切り分けるためStage 07だけを6,030 stepへ延長し、Stage 03比+0.3798 dBを確認した。
しかし後半半周の利得は+0.1289 dBに縮小し、最良でも昇格基準を
12.2435 dB下回った。そこで長期化を一般化せず、matched control比+0.20 dBを
超えたjoint-shot幅128だけをStage 19で五巡相当まで確認する限定的な再開とした。さらに
幅128、距離二乗、五巡budgetをStage 22で組み合わせ、その最大実測値を確定した。

## 最終昇格・停止監査

追加runは、有限metric、全scope check、checkpoint再計算一致に加え、matched baseline比
`+0.20 dB`を最低条件とした。Stage 21はこの条件を満たしたが、Stage 20へ全利得を足す
楽観予測でも7.5245 dBで現在bestを越えないため、組合せを停止した。

Stage 22はStage 20比+0.4162 dB、後半+0.2906 dBだったが、長期昇格
基準21.3433 dBを13.6426 dB下回った。同系統Study 018で得た最大の
2,500→50,000-step利得3.6567 dBをそのまま加える楽観値も
11.3574 dBに留まる。target自身で係数を最適化する384--512近傍linear-span診断
さえ約23.36 dBで、25 dB許容誤差energyの約1.459倍を残す。したがってsplit、raw metric、
dataを維持したまま昇格できる追加stageはない。

## Formal scope・漏洩監査

完了した全formal runで次を確認した。

- FFID数train/validation/test=`1,195 / 896 / 2,689`、overlap 0、各FFIDの最大split数1
- effective trace数=`578,685 / 437,087 / 1,287,693`
- target-FFID neighbor entry、target center、train-validation座標overlap、train collision、
  canonical duplicate残存がすべて0
- neighbor amplitude sourceはtrainだけ。test/excluded振幅値は未materialize
- validation targetだけをcheckpoint選択とmetricに使用
- 保存checkpointのraw metricと全validation再計算値が完全一致
- Stage 04/05のbracketing sourceは全件train。未解決、target-FFID、same-source-y、
  non-train参照がすべて0
- Stage 09以降は1 FFIDを8 x 68 receiver-cellのwhole shotとして構成し、入力近傍source、
  normalization、training targetはいずれもtrain splitだけから構築
- joint-shot runもstep 0をcheckpoint候補に含め、保存checkpointを全validationで再評価

各runで`scope_success=true`だった。主指標は25 dB未満のため
`metric_success=false`、総合`success=false`である。

## 最良Stage 07の監査

| 項目 | 結果 |
|---|---|
| Git commit | `7343bb0031a7713c55a19a691f47ef1d8b57e0ad` |
| model | crossline K1374、width 384、21,921,721 parameters |
| best step | 6,030 |
| primary metric | `9.0998 dB` |
| training audit | `11.9772 dB`、10,000 traces、seed 44 |
| checkpoint revalidation | 保存値 = 再計算値、`revalidation_matches=true` |
| clean validation traces | 437,087 |
| validation signal / error energy | 273,179,375.0032627 / 33,609,934.5625 |
| validation neighbor | mean 259.4439、min 9、zero 0 |
| runtime | 1,652秒（27分32秒） |
| peak CUDA allocated / reserved | 15,804,441,088 / 22,779,265,024 bytes |
| metric / scope / overall | `false` / `true` / `false` |

## Reproducibility

修正splitの生成:

```bash
python -m seis_interp.cli data prepare-baseline \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_whole_ffid_25pct_train_amplitude_qc \
  --config studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/config.yaml \
  --json
```

最良Stage 07の実行:

```bash
python -m seis_interp.cli train neighbor-inpainter \
  --config studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/variants/stage07_full_train_sweep_k1374.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_whole_ffid_25pct_train_amplitude_qc \
  --output runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/<run-id> \
  --device cuda:1 \
  --json
```

## 品質ゲート

`ruff check .`、`ruff format --check .`、`pytest`、`python -m seis_interp.cli doctor` は全て pass。

## 制約

- validation splitをcheckpoint選択とStage選択の両方に使うため、model-selection optimismが
  あり得る。test targetは契約どおり未参照であり、最終test generalizationは未評価である。
- primary metricはtarget自身のRMSを使うoracle waveform指標で、未知gainの実運用復元とは
  異なる。
- seed 42、単一survey、単一splitのPOCであり、別seed / surveyは未検証である。
- `cudnn_benchmark=true`、`cudnn_deterministic=false`なのでbitwise再現は保証しない。
- nearest / bracketing単体の値はgeometryルール選択用の診断で、独立したformal run artifactを
  持たない。

## 最終判断

**THRESHOLD NOT REACHED** — 全scope / leakage / checkpoint監査を通過した最良runは
`oracle_per_trace_unit_rms_global_snr_db = 9.0998 dB`であり、厳密な
`> 25.0 dB`条件を満たさなかった。crosslineを含む完全被覆とTRAIN一巡は改善したが、固定
bracketing、whole-shot化、受容野、容量、receiver条件付け、距離重み、dynamic attention、
loss、五巡budgetを含む21本のformal runでも残り15.9002 dBを説明できなかった。
長期昇格gateとtarget-derived local span診断も25 dB未満であり、同一scopeの追加実験は
empirically blockedと判断した。
