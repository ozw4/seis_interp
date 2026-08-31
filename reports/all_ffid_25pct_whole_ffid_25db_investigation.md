# FFIDを25%選択する条件で25 dBを目指した段階実験レポート

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

geometry-onlyのbracketing監査では、train 578,685行のうち524,285行、validation
437,087行のうち397,535行が両側bracketを持った。残り54,400 / 39,552行は片側nearestで、
未解決、non-train source、target-FFID source、same-source-y sourceはすべて0だった。
比較診断ではnearest-shotコピーが4.3999236270 dB、線形bracketing単体が
5.4785273632 dBだった。これは独立した正式runではなく、Stage 04を選ぶための診断値である。

## Finite-aperture切り分けの結果

Stage 01から03は全full-scope check、checkpoint再評価、test/excluded非参照監査に合格し、
best checkpointはいずれもstep 2,500だった。

| Stage | local K / width / steps | Validation S/N | 直前Stage差 | Stage 01差 |
|---:|---:|---:|---:|---:|
| [01](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T044414Z_f234880_stage01_k274_whole_ffid_2500_steps/metrics.json) | 274 / 384 / 2,500 | 4.431249374754326 dB | — | 基準 |
| [02](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T045444Z_31f81ca_stage02_crossline_k714_2500_steps/metrics.json) | 714 / 384 / 2,500 | 7.783543855019937 dB | +3.352294480265611 dB | +3.352294480265611 dB |
| [03](../runs/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/20260831T050923Z_31f81ca_stage03_crossline_k1374_2500_steps/metrics.json) | 1,374 / 384 / 2,500 | **8.719953365995504 dB** | **+0.936409510975567 dB** | **+4.288703991241178 dB** |

K274からcrossline K714への変更はvalidation zero-neighborを30.28%から3.56%へ減らし、
3.35 dBの大きな改善を得た。K1374はzero-neighborを0にし、さらに0.94 dB改善した。
したがってwhole-FFID条件ではcrosslineを含むsource coverageが主要因の一つである。一方、
被覆を完全化しても25 dBとの差は16.280046634004496 dB残り、有限aperture拡大だけでは
目標へ届かなかった。

