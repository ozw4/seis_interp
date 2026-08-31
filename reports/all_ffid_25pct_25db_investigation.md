# 全 FFID・train 25% 条件で 25 dB を目指した段階実験レポート

## 結論

- 対象: SEG C3 Narrow-Azimuth、全 amplitude-eligible FFID
- train ratio: 各 FFID 内で 25%
- 正式成功条件: `oracle_per_trace_unit_rms_global_snr_db > 25.0`
- 最良結果: **`16.348938587526803 dB`**（Stage 07、10,000 step）
- 閾値との差: **`-8.651061412473197 dB`**
- 判定: **未達**（`metric_success=false`、`scope_success=true`、`success=false`）

段階実験では、既存 Study 018 モデルの 25% density baseline から始め、明示的
neighbor reference、shared attention、aperture 拡大、deterministic moveout alignment、
model width、training budget を順に切り分けた。width 384→512 は 2,500 step で
`+0.215653618119529 dB` の小さい利得を示し、fresh 10,000-step run へ昇格した。
最良 run は 10,000 step まで単調に改善したが、最後の 2,500 step の増分は
`+0.215778544578125 dB` まで縮小した。

Study 018 の同系統モデルでは 10,000→50,000 step の利得が
`+1.7497346480585065 dB` だった。この実測に基づく 50,000-step 昇格基準
`23.250265351941493 dB` に Stage 07 は `6.901326764414691 dB` 届かなかったため、
budget-only formal 延長は実行せず停止した。

最良の immutable run は
[`20260831T030617Z_05dc7d9_stage07_width512_k274_10000_steps`](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T030617Z_05dc7d9_stage07_width512_k274_10000_steps/)
である。

## データ・split 契約

「train 25% FFID」は whole FFID の 25% を選ぶ意味ではなく、各 eligible FFID の
amplitude-eligible trace を whole-trace 単位で 25% train に割り当てる意味とした。
同一 trace の time sample が複数 split に跨ることはない。

| 項目 | 条件 |
|---|---|
| dataset | SEG C3 Narrow-Azimuth、manifest で固定した 4 SEG-Y source |
| split scope | `per_ffid` |
| seed | 42 |
| train | 25% |
| validation | holdout 75% の 25% = 全体の 18.75% |
| test | holdout 75% の 75% = 全体の 56.25% |
| amplitude QC | all-zero を除外、`max_abs_amplitude <= 10000` |
| duplicate policy | 全 split の前に物理座標ごとの最低 `array_row` を保持 |
| eligible FFID | 4,780 |
| fully excluded FFID | 1746 |
| samples / trace | 625 |

| split | prepared rows | duplicate 除去 | effective rows |
|---|---:|---:|---:|
| train | 575,870 | 6 | **575,864** |
| validation | 431,890 | 3 | **431,887** |
| test | 1,295,720 | 6 | **1,295,714** |

15 個の duplicate physical cell から 15 行を除去し、残存 collision は 0 である。
全 4,780 eligible FFID が train / validation / test のすべてに寄与する。processed
contract は
[`inputs.yaml`](../studies/study_019_all_ffid_25pct_neighbor_inpainter/inputs.yaml)
に固定した。

## 評価契約

モデル入力の振幅は train trace のみから取得する。validation 振幅は checkpoint 選択と
metric 計算だけに使い、test / excluded 振幅は materialize しない。target offset は
neighbor aperture から除外する。予測値の後処理正規化は行わず、raw model output と
oracle per-trace unit-RMS validation target の point-weighted global S/N を測る。

```text
success
  iff oracle_per_trace_unit_rms_global_snr_db > 25.0
  and all formal scope/leakage checks are true
```

比較は厳密な `>` であり、25.0 dB ちょうどは失敗である。

## 事前比較と昇格規則

50% train の Study 018 formal architecture は、2,500 step で
`16.80368309012617 dB`、50,000 step で `20.460355529598864 dB` だった。同じ
architecture を 25% train に移した Stage 01 は `14.222844294958843 dB` で、density
低下だけで `-2.580838795167327 dB` となった。

2,500-step ablation は Stage 01 比 `+0.20 dB` 以上かつ best が最終 step の場合だけ
10,000 step へ昇格することとした。10,000-step winner は、Study 018 の実測 tail gain
を足しても 25 dB に届くために `23.250265351941493 dB` 以上を必要条件とした。

## 段階実験の結果

full-scope Stage の差分は Stage 01 比である。全 full-scope run で
`scope_success=true`、checkpoint 再評価一致、best checkpoint は最終 step だった。

| Stage | 切り分け条件 | K / width / steps | Validation dB | Stage 01 差 | 採否 |
|---:|---|---|---:|---:|---|
| [01](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T010022Z_8e6b250_stage01_study018_formal_2500_steps/metrics.json) | Study 018 formal architecture を 25% split へ移植 | 274 / 384 / 2,500 | 14.222844294958843 | 基準 | baseline |
| [02](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T012028Z_fda4b9d_stage02_residual_neighbor_reference/metrics.json) | aligned-neighbor reference + zero-init residual | 274 / 384 / 2,500 | 14.22890961173312 | +0.006065316774277 | 不採用 |
| [03](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T015611Z_71e2e93_stage03_shared_offset_attention/metrics.json) | shared temporal encoder + single masked attention fusion | 274 / 384 / 2,500 | 9.819645233036228 | -4.403199061922615 | 不採用 |
| [04](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T020919Z_71e2e93_stage04_same_line_k734/metrics.json) | legacy CNN の aperture だけ K734 へ拡大 | 734 / 384 / 2,500 | 14.089875885195529 | -0.132968409763315 | 不採用 |
| [05](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T023909Z_d361c40_stage05_legacy_coarse_shift_k274/metrics.json) | `shift = 3 * dry` の zero-padded coarse alignment | 274 / 384 / 2,500 | 14.204319211934315 | -0.018525083024528 | 不採用 |
| [06](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T025138Z_4283dfd_stage06_width512_k274/metrics.json) | unshifted baseline の width だけ 384→512 | 274 / 512 / 2,500 | 14.438497913078372 | +0.215653618119529 | 10k へ昇格 |
| [07](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T030617Z_05dc7d9_stage07_width512_k274_10000_steps/metrics.json) | Stage 06 を fresh 10,000-step cosine horizon へ延長 | 274 / 512 / 10,000 | **16.348938587526803** | **+2.126094292567959** | study best、50k は停止 |

Stage 03 の前には FFID 2348–2363、100 step の GPU smoke を実行した。
[`Stage 03a`](../runs/study_019_all_ffid_25pct_neighbor_inpainter/20260831T015518Z_71e2e93_stage03a_shared_attention_ffid2348_2363_smoke/metrics.json)
は `5.580651159639629 dB` で有限・単調な学習と checkpoint 再現を確認したが、16 FFID
だけのため `scope_success=false` であり、正式比較値には使っていない。

## 各切り分けから分かったこと

### Neighbor density

25% split の K274 validation availability は平均 54.788、p01 / median / max は
22 / 56 / 98 本だった。Stage 04 の K734 は平均 132.690、p01 / median / max は
56 / 131 / 231 本まで増えたが、S/N は `-0.1330 dB` 悪化した。遠い neighbor を単に
追加することは、近い観測の density 低下を補わなかった。

### Fusion

Stage 03 は K274 を shared encoder 後に少数 feature へ attention fusion してから
decoder へ渡した。train audit `9.822922017404714 dB` と validation
`9.819645233036228 dB` が近いため、主因は過学習ではなく、複数 event を単一 fusion
へ早く圧縮する表現 bottleneck と解釈した。

train-only で距離 prior を選んだ補助 probe では、scale 0.25 の aligned fixed baseline が
FFID 2348–2363 validation で約 5.111 dB、scale 0.1 は約 4.887 dB だった。prior 調整の
期待利得は約 0.22 dB に留まり、Stage 03 の 4.40 dB の損失を救えないため full run は
行わなかった。この probe は独立 run artifact を持たない診断値である。

### Alignment

train-only cross-correlation から `3 * relative_receiver_y_index` samples の coarse shift を
導出した。Stage 05 では source index を `output_index - shift` とし、circular wrap を
禁止し、端部 availability を sample 単位で伝えた。結果は Stage 01 比
`-0.018525083024528 dB` で、既存 full-receptive-field CNN と FIR に対する追加利得は
なかった。

### Capacity と budget

width 512 は 2,500 step で唯一事前昇格基準を超えたが、利得は `+0.2157 dB` だった。
10,000 step への fresh 延長は Stage 06 から `+1.910440674448431 dB`、Stage 01 から
`+2.126094292567959 dB` 改善した。budget は最も大きな正の要因だが、25 dB との差を
埋める傾きではなかった。

## Stage 07 学習曲線

| Step | Validation S/N (dB) | 直前評価からの増分 |
|---:|---:|---:|
| 1 | -1.525307243102649 | — |
| 2,500 | 14.558682274477732 | — |
| 5,000 | 15.557861355021528 | +0.999179080543796 |
| 7,500 | 16.133160042948678 | +0.575298687927150 |
| 10,000 | **16.348938587526803** | +0.215778544578125 |

後半 2,500-step ごとの増分は連続して縮小した。単に 5 倍の更新数を許可する根拠には
ならず、事前の absolute promotion gate にも大きく届かなかった。

## 実装した内容

- 25% per-FFID whole-trace split と exact effective count の study contract / tests
- optional `masked_aligned_neighbor_mean` prediction reference
- exact geometry offset、zero-padded coarse shift、masked attention を持つ
  `SharedOffsetAttentionInpainter`
- legacy `NeighborTraceInpainter` の optional deterministic coarse alignment
- coarse alignment の exact offset / derived shift checkpoint 保存と破損検知
- 既定 shift 0 で旧 initialization、output、state dict、checkpoint の厳密互換を維持
- Stage 01–07 の resolved YAML variants と Study 019 integration contract tests
- model permutation、all-unavailable、padding mask、2-step gradient、checkpoint round-trip、
  derived-buffer corruption rejection の unit / integration tests

研究条件と履歴は
[`README.md`](../studies/study_019_all_ffid_25pct_neighbor_inpainter/README.md)、
[`config.yaml`](../studies/study_019_all_ffid_25pct_neighbor_inpainter/config.yaml)、
[`decisions.md`](../studies/study_019_all_ffid_25pct_neighbor_inpainter/decisions.md)
に記録した。

## 最良 run の監査

| 項目 | 結果 |
|---|---|
| Git commit | `05dc7d9bbd913aec9c0caf33d9559273a5a61333` |
| model | unshifted K274、width 512、14,898,313 parameters |
| best step | 10,000 |
| primary metric | `16.348938587526803 dB` |
| training audit | `16.592268072774452 dB`、10,000 traces、seed 44 |
| checkpoint revalidation | 保存値 = 再計算値、`revalidation_matches=true` |
| clean validation traces | 431,887 |
| validation signal / error energy | 269,929,375.0038355 / 6,256,857.879580375 |
| peak CUDA allocated / reserved | 13.308 / 20.805 GiB |
| runtime | 2,048 s |
| metric / scope / overall | `false` / `true` / `false` |

全 formal scope check は true だった。

- effective split counts と eligible FFID count が一致
- target center offset が 0 本
- canonical duplicate physical cell、train geometry collision、train-validation overlap が 0
- test / excluded amplitude value rows は非 materialize
- selected raw metric と checkpoint 再計算値が一致

## Reproducibility

processed split の生成例:

```bash
python -m seis_interp.cli data prepare-baseline \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_per_ffid_25pct_train_amplitude_qc \
  --config studies/study_019_all_ffid_25pct_neighbor_inpainter/config.yaml \
  --json
```

最良 run の実行:

```bash
python -m seis_interp.cli train neighbor-inpainter \
  --config studies/study_019_all_ffid_25pct_neighbor_inpainter/variants/stage07_width512_k274_10000_steps.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_per_ffid_25pct_train_amplitude_qc \
  --output runs/study_019_all_ffid_25pct_neighbor_inpainter/<run-id> \
  --device cuda:1 \
  --json
```

主要 SHA-256:

| Artifact | SHA-256 |
|---|---|
| Stage 07 `metrics.json` | `a788d014e2c930247bca01ba64260cfd1a12b9947992b96703e76f7341ca023c` |
| Stage 07 `artifacts/best.pt` | `883c1646d69b80ec69a73802a0680736d3170749e3c981364f60748d9ac0d2e7` |
| Stage 07 `config.resolved.yaml` | `f0f5060c74a85c18a679666d948544e17d7b2074a9ef0d99a3ace46116526356` |
| Stage 07 `inputs.lock.json` | `16c401cb30acc9d836031bcc7201082fd8110448464abe24fcb0e410f92948d9` |
| processed `trace_split.parquet` | `1454cde6402b7e6ad3ffd4d9bbca77ca6f2e2ff8034ddf90c0baa2f021ac553f` |
| processed `preparation.json` | `6e72f572e20a3bcadcf16c45601cc0636b187009ea369987d79e3fe57d6d3f90` |

## Repository quality gates

| Gate | 結果 |
|---|---|
| `ruff check .` | pass、`All checks passed!` |
| `ruff format --check .` | pass、`197 files already formatted` |
| `pytest -q` | pass、`1150 passed in 41.72s` |
| `python -m seis_interp.cli doctor` | exit 0 |

doctor は Python 3.10.12、PyTorch CUDA available、NVIDIA H100 NVL 2 台、numpy、
PyYAML、segyio、pandas、pyarrow、matplotlib、Codex、Claude Code、GitHub CLI、data root
の exists / readable を確認した。

## 制約と次の研究課題

- 同じ validation split を Stage 選択と checkpoint 選択に使っており、model-selection
  optimism があり得る。
- test target は評価していない。これは事前契約どおりだが、未観測 test の最終
  generalization 証拠ではない。
- primary metric は target 自身の RMS を使う oracle waveform 指標であり、実運用の
  unknown gain 復元は別課題である。
- seed 42、単一 survey、単一 split の POC であり、別 seed / survey の再現性は未検証。
- `cudnn_benchmark=true`、`cudnn_deterministic=false` のため bitwise 再現は保証しない。
- 25 dB を再度狙う場合は、neighbor を 1 本へ早期圧縮しない multi-head signed spatial
  basis、factorized spatial convolution、local low-rank reconstruction など、複数の
  coherent event を保持する新しい融合機構を先に検証すべきである。

## 最終判断

**THRESHOLD NOT REACHED** — 全 scope / leakage / checkpoint 監査を通過した最良 run は
`oracle_per_trace_unit_rms_global_snr_db = 16.348938587526803 dB` であり、厳密な
`> 25.0 dB` 条件を満たさなかった。段階実験により capacity と budget の正の効果は
確認したが、実測 tail と事前停止基準から 50,000-step の budget-only 延長は支持されない。
