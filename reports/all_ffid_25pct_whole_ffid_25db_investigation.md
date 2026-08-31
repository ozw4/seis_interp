# FFIDを25%選択する条件で25 dBを目指した段階実験レポート

## 結論

- 対象: SEG C3 Narrow-Azimuth、全amplitude-eligible FFID
- train条件: FFIDを丸ごと1,195 / 4,780個選択（正確に25%）
- 正式成功条件: `oracle_per_trace_unit_rms_global_snr_db > 25.0`
- 最良結果: **`8.719953365995504 dB`**（Stage 03、K1374、2,500 step）
- 閾値までの不足: **`16.280046634004496 dB`**
- 判定: **未達**（`metric_success=false`、`scope_success=true`、`success=false`）

旧Study 019の「各FFID内のtraceを25%」という解釈を修正し、FFID集合の25%を丸ごと
trainへ割り当てる実装、processed data、Study 020、5本のimmutable runを新規作成した。
近傍被覆、crossline、source-y範囲、shot-bracketing reference、両者の組合せを順に
切り分けた。crossline K1374はK274から`+4.288703991241179 dB`改善したが、25 dBへの
実測ベースの長期budget昇格基準を大幅に下回ったため、追加budgetは実行しなかった。

最良runは
[`20260831T050923Z_31f81ca_stage03_crossline_k1374_2500_steps`](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T050923Z_31f81ca_stage03_crossline_k1374_2500_steps/)
である。

## 条件変更

今回の「FFIDを25%選ぶ」は、各FFID内のtraceを25%ずつtrainへ入れる意味ではなく、
amplitude-eligible FFID集合から25%のFFIDを選び、そのFFIDに属するeligible traceを
すべてtrainへ入れる意味とした。旧解釈のStudy 019とrunは実行履歴として変更せず、
修正条件を独立した
[`Study 020`](../studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/README.md)
として実装・実行した。

| 項目 | 旧Study 019 | 修正Study 020 |
|---|---|---|
| split scope | 各FFID内のtrace単位 | FFID丸ごと（`whole_ffid`） |
| train | 全eligible FFIDから各25% trace | eligible FFIDの25%を選択 |
| 1 FFIDが属するsplit数 | 3 | 1 |
| processed data / run | 履歴として保持 | 新規生成 |

## データ・split契約

対象はmanifestでchecksumを固定したSEG C3 Narrow-Azimuth全4ファイルである。seed 42で
4,780個のamplitude-eligible FFIDを一度だけ並べ替え、1,195 FFIDをtrain、896 FFIDを
validation、2,689 FFIDをtestへ割り当てた。trainは正確に`1,195 / 4,780 = 25%`で、
3集合の重複は0、各FFIDの所属split数は最大1である。

| 項目 | 条件 |
|---|---|
| split scope | `whole_ffid` |
| seed | 42 |
| train / validation / test FFID | 1,195 / 896 / 2,689 |
| amplitude QC | all-zero除外、`max_abs_amplitude <= 10000` |
| fully excluded FFID | 1746 |
| duplicate policy | 全split横断で最低`array_row`を保持。split/amplitude非参照 |
| samples / trace | 625 |

| split | prepared traces | duplicate除去 | effective traces |
|---|---:|---:|---:|
| train | 578,688 | 3 | **578,685** |
| validation | 437,088 | 1 | **437,087** |
| test | 1,287,704 | 11 | **1,287,693** |

入力2,304,024 traceのうち544 traceをamplitude QCで除外した。15個のduplicate physical
cellから15行を除去し、残存duplicate、train geometry collision、train-validation物理座標
overlapはいずれも0である。processed contractは
[`inputs.yaml`](../studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/inputs.yaml)
に固定した。

## 実装した内容

- `sampling.split_scope: whole_ffid`と`sampling.random_ffid_holdout_fraction`を追加し、
  amplitude QC後のFFID単位で決定論的にsplitする処理を実装した。
- `preparation.json`へsplit別FFID数を保存し、configの要求値と実測値を検証するようにした。
- whole-FFID時は各FFIDが1 splitだけに属すること、3 splitのFFID集合が重複しないこと、
  unionが全eligible FFIDになることをformal scopeへ追加した。
- train targetでも同じtarget FFIDの全neighborをexact FFID IDでmaskし、FFIDが丸ごと未知な
  validation条件と学習contextを一致させた。
- neighbor振幅はtrain FFIDだけ、validation targetはcheckpoint選択とmetricだけに使用し、
  test/excluded振幅値はtraining runでmaterializeしない監査を追加した。
- CLIでtrace単位とFFID単位のsplitを跨いで上書きするときだけholdout率の明示を必須にし、
  従来の`global`と`per_ffid`間の上書き互換性を維持した。
- 同一source-x line・同一relative receiverのstrict lower/upper train shotをsource-y距離で
  線形補間するprediction referenceを追加した。片側だけならnearestを使い、target FFID、
  same source-y、non-train sourceを禁止する。referenceはdropoutせず、CNN residual headを
  zero初期化する。
- split、neighbor、bracketing、checkpoint round-trip、設定契約をunit/integration testsで
  固定した。

## 評価契約

train traceとvalidation targetはそれぞれ自身のRMSでunit-RMS化する。予測値を後処理で
再正規化せず、raw model outputとoracle unit-RMS validation targetのpoint-weighted global
S/Nを測る。

```text
success
  iff oracle_per_trace_unit_rms_global_snr_db > 25.0
  and all formal scope/leakage checks are true
```

比較は厳密な`>`であり、25.0 dBちょうどは失敗である。validationをcheckpoint選択と
切り分け選択に使い、test targetは事前契約どおり参照しない。

## 段階実験の設計

すべてfresh initialization、2,500 update、同じseedと評価domainを使う。Stage 01から03は
有限apertureの被覆だけを段階的に変える。Stage 04はStage 01のK274へ戻し、source方向に
整合したprediction referenceだけを追加する。

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
比較診断ではnearest-shotコピーが4.3999236270 dB、線形bracketing単体が
5.4785273632 dBだった。これは独立した正式runではなく、Stage 04を選ぶための診断値である。

Stage 05は、Stage 03とStage 04がそれぞれStage 01比`+0.20 dB`以上かつ全scope check合格の
場合だけ実行する。両者は有限aperture coverageとlong-range shot referenceという異なる
要因なので、独立効果を確認してから組み合わせる。

## 段階実験の結果

Stage 01から05は全full-scope check、checkpoint再評価、test/excluded非参照監査に合格し、
best checkpointはいずれもstep 2,500だった。

| Stage | local K / width / steps | Validation S/N | 主比較差 | Stage 01差 |
|---:|---:|---:|---:|---:|
| [01](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T044414Z_f234880_stage01_k274_whole_ffid_2500_steps/metrics.json) | 274 / 384 / 2,500 | 4.431249374754326 dB | baseline | 基準 |
| [02](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T045444Z_31f81ca_stage02_crossline_k714_2500_steps/metrics.json) | 714 / 384 / 2,500 | 7.783543855019937 dB | Stage 01比 +3.352294480265612 dB | +3.352294480265612 dB |
| [03](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T050923Z_31f81ca_stage03_crossline_k1374_2500_steps/metrics.json) | 1,374 / 384 / 2,500 | **8.719953365995504 dB** | Stage 02比 +0.936409510975567 dB | **+4.288703991241179 dB** |
| [04](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T052831Z_610c307_stage04_k274_source_bracketing_residual_2500_steps/metrics.json) | 274 + ref 1 / 384 / 2,500 | 8.51333997509688 dB | Stage 01比 +4.082090600342554 dB | +4.082090600342554 dB |
| [05](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T053909Z_f0a67c8_stage05_crossline_k1374_source_bracketing_residual_2500_steps/metrics.json) | 1,374 + ref 1 / 384 / 2,500 | 8.595997409114656 dB | Stage 03比 -0.123955956880849 dB | +4.16474803436033 dB |

K274からcrossline K714への変更はvalidation zero-neighborを30.28%から3.56%へ減らし、
3.35 dBの大きな改善を得た。K1374はzero-neighborを0にし、さらに0.94 dB改善した。
したがってwhole-FFID条件ではcrosslineを含むsource coverageが主要因の一つである。一方、
被覆を完全化しても25 dBとの差は16.280046634004496 dB残り、有限aperture拡大だけでは
目標へ届かなかった。

Stage 04はbracketing単体に近いstep 1の5.481098917181287 dBから、2,500 stepで
8.51333997509688 dBまで改善した。K274との比較では明確に有効だが、K1374単独より
0.206613390898625 dB低かった。Stage 05はStage 04比では0.082657434017776 dB改善したが、
Stage 03比では0.123955956880849 dB悪化した。完全被覆contextと固定bracketing referenceの
単純な組合せに正の相乗効果はなかった。

## Budget昇格を停止した根拠

同系統のStudy 018では、同じformal architectureが2,500 stepの
16.80368309012617 dBから50,000 stepの20.460355529598864 dBへ改善し、実測利得は
3.656672439472694 dBだった。この利得をそのまま加えて25 dBへ届くための2,500-step
昇格基準は`21.343327560527307 dB`である。

最良Stage 03はこの基準を12.623374194531802 dB下回った。別条件のStudy 019で観測した
width 384→512の利得も2,500 stepで0.215653618119529 dBに留まる。したがってcapacityまたは
budgetだけを増やして残り16.28 dBを埋める根拠がなく、10,000 / 50,000-step runを停止した。

## Formal scope・漏洩監査

5 runすべてで次を確認した。

- FFID数train/validation/test=`1,195 / 896 / 2,689`、overlap 0、各FFIDの最大split数1
- effective trace数=`578,685 / 437,087 / 1,287,693`
- target-FFID neighbor entry、target center、train-validation座標overlap、train collision、
  canonical duplicate残存がすべて0
- neighbor amplitude sourceはtrainだけ。test/excluded振幅値は未materialize
- validation targetだけをcheckpoint選択とmetricに使用
- 保存checkpointのraw metricと全validation再計算値が完全一致
- Stage 04/05のbracketing sourceは全件train。未解決、target-FFID、same-source-y、
  non-train参照がすべて0

各runで`scope_success=true`だった。主指標は25 dB未満のため
`metric_success=false`、総合`success=false`である。

## 最良Stage 03の監査

| 項目 | 結果 |
|---|---|
| Git commit | `31f81ca16995f506d5c2443d236e8d6652f4333c` |
| model | crossline K1374、width 384、21,921,721 parameters |
| best step | 2,500 |
| primary metric | `8.719953365995504 dB` |
| training audit | `10.3349836646672 dB`、10,000 traces、seed 44 |
| checkpoint revalidation | 保存値 = 再計算値、`revalidation_matches=true` |
| clean validation traces | 437,087 |
| validation signal / error energy | 273,179,375.0032627 / 36,681,963.17026253 |
| validation neighbor | mean 259.4438864574、min 9、zero 0 |
| runtime | 1,089秒（18分9秒） |
| peak CUDA allocated / reserved | 15,800,184,832 / 22,779,265,024 bytes |
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

最良Stage 03の実行:

```bash
python -m seis_interp.cli train neighbor-inpainter \
  --config studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/variants/stage03_crossline_k1374.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_whole_ffid_25pct_train_amplitude_qc \
  --output runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/<run-id> \
  --device cuda:1 \
  --json
```

主要SHA-256:

| Artifact | SHA-256 |
|---|---|
| Stage 03 `metrics.json` | `98cf65c425352cf59b577af776361b0a240d89a3696154ee396155c8eb1cd5aa` |
| Stage 03 `artifacts/best.pt` | `b0e0b018c948ac32417d89e5e09ad37f5877a46da4da79e23df1eff13607fd94` |
| Stage 03 `config.resolved.yaml` | `f037f699b5dafc0b153be0a5a6df8ab451a1dbae14860e72c7ee577bc66e05c3` |
| Stage 03 `inputs.lock.json` | `98f2bc2811a232bea0fd6c0437b2c348618373d885dfbb60a7932a6e28e6a2dc` |
| Stage 03 `run.json` | `53b69d3524ebf8eb7b29b4df3271afa44ea85cdcb7807ad4725f69f3b4d79fb7` |
| processed `preparation.json` | `2ca69ca22af9149ac8183dbd67937a32d8ce769d3b84c316b83ab5b62cf588cf` |
| processed `normalization.json` | `540e1c8f79e2b14f61cfa287e91bf648fbbc9831fc25dda138be40610d7c26b1` |
| processed `trace_split.parquet` | `7987c94f9b716b9f6f6ca507a13e28166af8f779664b6ffc516f5d54438e3312` |

Stage別`metrics.json`のSHA-256:

| Stage | SHA-256 |
|---:|---|
| 01 | `025b8e5c760b56163734bf1f1cf9ab72683b9728ee7bc8a31efc1b07bb1fdac9` |
| 02 | `d4e758cedf570aea88f5d02af6f3a72f62618c693bdfb15b0b03cdde7e49010c` |
| 03 | `98cf65c425352cf59b577af776361b0a240d89a3696154ee396155c8eb1cd5aa` |
| 04 | `1c0d323adcb178a1d7c9daf270bb3daf5f035811cf6ca7f166ec8cd75e8c5691` |
| 05 | `5409b9a174dbad929b89a5e82f5446fa1503899834a209e41df3e38f5678fb5c` |

## Repository quality gates

2026-08-31に最終状態で次を実行し、すべて合格した。

| Command | Result |
|---|---|
| `ruff check .` | `All checks passed!` |
| `ruff format --check .` | `204 files already formatted` |
| `pytest` | `1178 passed in 42.55s` |
| `python -m seis_interp.cli doctor` | exit 0。Python 3.10.12、PyTorch 2.5.0a0、CUDA有効、H100 NVL 2台、data root readable |

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
`oracle_per_trace_unit_rms_global_snr_db = 8.719953365995504 dB`であり、厳密な
`> 25.0 dB`条件を満たさなかった。crosslineを含む完全被覆は大きく改善したが、固定
bracketing referenceとの組合せに相乗効果はなく、実測tailに基づく長期budget昇格基準にも
大きく届かなかった。
