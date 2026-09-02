# `all_ffid_siren` 15 dB 達成に向けた段階実験レポート

- 実施日: 2026-08-28
- 対象データ: SEG C3 Narrow-Azimuth
- 当初の対象: `study_016_all_ffid_siren`
- 正式な成功条件: `oracle_per_trace_unit_rms_global_snr_db > 15.0`
- 採用した後継条件: `study_017_all_ffid_neighbor_inpainter`
- 正式 run: `20260828T194620Z_edb2561_all_ffids`

## 結論

指定指標での全 FFID 成功を達成した。正式 run の
`oracle_per_trace_unit_rms_global_snr_db` は **18.111870025656728 dB** で、15 dB の
閾値を **3.111870025656728 dB** 上回った。run に記録された `metric_success`、
`scope_success`、`success` はすべて `true` である。

ただし、成功したモデルは座標だけを入力する SIREN ではない。coordinate-only SIREN
のリーク安全な最良値は 10.680487521794015 dB であり、15 dB には届かなかった。
その切り分け結果を受け、train split の物理近傍波形だけを条件とする temporal CNN
へ研究条件を変更した。このため、結果は「SIREN 自体の成功」ではなく、
「`all_ffid_siren` から開始した切り分けにより、指定波形指標を満たす補間条件を発見し、
全 FFID の正式 run で検証した成功」と解釈する。

正式な研究契約と結果は
[`study_017_all_ffid_neighbor_inpainter`](../studies/study_017_all_ffid_neighbor_inpainter/README.md)、
機械可読な正本は
[`metrics.json`](../runs/study_017_all_ffid_neighbor_inpainter/20260828T194620Z_edb2561_all_ffids/metrics.json)
にある。

## 成功条件と評価領域

評価対象は、各 validation trace をその trace 自身の RMS で除した oracle unit-RMS
波形である。checkpoint の選択と最終判定には、モデルの raw prediction を用いた。
prediction を再度 unit-RMS 化した値は診断値であり、選択には使っていない。

```text
oracle_per_trace_unit_rms_global_snr_db
  = 10 log10(
      sum(target_unit^2)
      / sum((target_unit - prediction_raw)^2)
    )

success ⇔ metric > 15.0 dB AND formal scope checks are all true
```

比較は厳密な `>` であり、15.0 dB ちょうどは失敗とするテストを追加した。

## データとリーク監査

準備済み split は seed 42 の FFID 内 trace split で、時間 sample を別 split へ分けて
いない。振幅品質フィルタ後の元の eligible trace 数は 2,303,480 だった。

全 survey を監査したところ、同一の物理座標
`[source_x_m, source_y_m, receiver_x_m, receiver_y_m]` を持つセルが15個あり、30行が
重複していた。この中には train-validation 間の同一物理セルが2組含まれていた。
target-center を近傍から外すだけでは、同一座標・同一波形の twin trace を防げないため、
split、FFID 範囲、振幅を参照する前に、各物理セルで最低 `array_row` の行だけを残した。

| 項目 | 正規化前 | 正規化後 | 除去数 |
|---|---:|---:|---:|
| train | 1,842,102 | 1,842,090 | 12 |
| validation | 114,492 | 114,490 | 2 |
| test | 346,886 | 346,885 | 1 |
| eligible 合計 | 2,303,480 | 2,303,465 | 15 |

観測された FFID 2–4782 のうち、FFID 1746 の544 tracesは既存の振幅品質ルールにより
全て `excluded` である。それ以外の全 4,780 eligible FFID と625 samplesは、
canonicalization後も保持した。

正式 run では次をすべて検査し、合格した。

- 残存する重複物理セル: 0
- train geometry の座標 collision: 0
- train-validation の物理座標 overlap: 0
- train-validation の完全一致 unit-amplitude overlap: 0
- neighbor offset 内の target center: 0
- neighbor amplitude の供給元: train split のみ
- test/excluded amplitude value の materialization: なし
- source-x shot line をまたぐ neighbor lookup: なし

`amplitudes.npy` 全体の byte 列は入力固定のため SHA-256 を計算したが、test/excluded
の値を数値 tensor として読み込んでいない。この違いも run の `amplitude_access` に
明示した。

## 段階1: SIREN の学習経路を切り分け

まず、point sampling、FFID batch、complete-trace batch、幅、座標表現、学習率 schedule、
時間座標 scale、周波数、深い層、dense skip を順に切り分けた。以下は完全な
`metrics.json` を持つ immutable run で、値は同じ oracle per-trace unit-RMS 領域の
`best_validation_global_snr_db` である。

| Stage | 主な条件 | FFID 範囲 | Best dB | 判定 |
|---:|---|---|---:|---|
| 1 | random points、width 512、batch 300k | 全 survey | -0.000041 | 後の重複監査基準では不採用 |
| 2 | full-FFID、correlation 0.3、width 512 | 全 survey | -0.001481 | 後の重複監査基準では不採用 |
| 3 | complete-trace batch、legacy 6D、width 256 | 2348 | 1.479477 | リーク安全な subset |
| 4 | Stage 3 の範囲拡張 | 2348–2351 | 4.593618 | リーク安全な subset |
| 5 | width 256 | 2348–2363 | 5.138709 | リーク安全な subset |
| 6 | width 512 | 2348–2363 | 4.493958 | 改善なし |
| 7 | Cartesian CMP + half-offset、width 256 | 2348–2363 | 10.230463 | 大幅改善 |
| 8 | Stage 7 + cosine LR | 2348–2363 | 10.324141 | 小幅改善 |
| 9 | Cartesian + offset radius | 2348–2363 | 9.844906 | 悪化 |
| 10 | Cartesian、width 512、cosine LR | 2348–2363 | 10.397155 | 容量増の効果は小さい |
| 11 | Cartesian、time scale 4 | 2348–2363 | 10.473193 | 小幅改善 |
| 12 | Cartesian、`omega_0=90`、width 256 | 2348–2363 | **10.680488** | SIREN 最良 |
| 15 | layer omega 90→30、4層 | 2348–2363 | 10.444614 | Stage 12 未満 |

Stage 12 の正本は
[`metrics.json`](../runs/study_all_ffid_temp/20260828T175604Z_f7c0ea2_stage12_trace_batch_ffid2348_2363_cartesian_omega90_w256_cosine/metrics.json)
にある。Stage 3–12/15 の FFID 範囲には重複物理セルがなく、trace split としては
リーク安全だが、全 FFID の正式結果ではない。

次の条件も実行したが、checkpoint だけが残り、完全な run metadata を持たないため
正式証拠には採用しなかった。

| Stage | 条件 | Checkpoint best dB | 結論 |
|---:|---|---:|---|
| 13 | exponential omega 5→50 | 0.000696 | collapse |
| 14 | exponential omega 30→90 | 8.940770 | Stage 12 未満 |
| 16 | dense skip、omega 90→30 | 9.804371 | 改善なし |
| 17 | fixed omega 30、12層 | 0.010177 | collapse |
| 18 | dense skip、12層 | 0.000344 | collapse |

この段階から、単純な計算量・幅・深さの追加では15 dBまでの約4.32 dB差を埋めにくく、
座標だけから各波形を生成する条件そのものが主な制約と判断した。

## 段階2: 表現と古典的補間の切り分け

SIREN の局所 optimum だけを見て判断しないため、profile 出力、Fourier-ReLU、
masked tensor completion、moveout、低ランク/POCS、非局所 retrieval、複数 expert の
混合を一時診断した。

| 系統 | 範囲 | Validation dB | 扱い |
|---|---|---:|---|
| Profile direct-conv、latent 64、5 Fourier bands | 2348–2363 | 10.024054 | リーク安全 proxy |
| Profile direct-conv | 2314–2382 | 10.345880 | リーク安全 proxy |
| ISR Fourier-ReLU、K=10 | 2348–2363 | 4.522785 | リーク安全 proxy |
| Masked 3D gather completion | 2348–2363 | 7.730520 | リーク安全 proxy |
| Source-relative shared coordinate proxy | 2348–2363 | 10.275549 | リーク安全 proxy |
| Cartesian global coordinate proxy | 2348–2363 | 10.140656 | リーク安全 proxy |
| Time mixture of 5 SIREN experts | 2348–2363 | 約11.0311 | 一時診断 |
| Leakage-safe moveout 系の最良 | 2348–2363 | 約10.8098 | 一時診断 |
| Target を参照する32-neighbor least squares | subset | 約15.157 | **リークにより無効** |

`/tmp` に JSON が残った proxy もあるが、Git SHA、入力 lock、完全な run metadata を
備えた immutable run ではないため、上表は方向決定の根拠に限定した。特に15 dBを
超えた non-local least squares は target trace 自体を係数推定に使うため、成功候補から
除外した。リーク安全な古典/ensemble 系の組合せも15 dBには届かなかった。

この切り分けにより、近隣 trace の波形情報は必要だが、validation target を直接参照せず
train trace だけから条件付けするモデルが必要だと判断した。

## 段階3: Train-only neighbor temporal CNN

### モデル

各 target に対し、自然な acquisition lattice 上の104位置を固定順序で問い合わせる。

- relative receiver-x: index ±1、40 m間隔
- source shot: index ±2、80 m間隔
- relative receiver-y: index ±3、40 m間隔
- target offset `(0, 0, 0)` は除外
- source-x line は固定し、別 shot line へ移動しない
- train 以外または欠損位置は振幅0とavailability `false`

入力は104本のtrain-only unit-RMS波形、104 availability channel、train geometryだけで
fitした3 target coordinates、正規化時間である。モデルはkernel 15・width 128のstem、
kernel 7のgated depthwise residual block 11個、scalar trace headで構成し、dilationは
`[1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1]`、parameter数は983,041である。

### 学習条件

| 項目 | 値 |
|---|---:|
| seed | 42 |
| optimizer | AdamW |
| updates | 2,500 |
| batch size | 96 traces |
| learning rate | `5e-4` |
| minimum learning rate | `1.5e-5` |
| schedule | cosine |
| weight decay | `1e-5` |
| neighbor dropout | 0.05 |
| loss | MSE + 0.1 × first-difference MSE |
| gradient clip norm | 1.0 |
| CUDA precision | bfloat16 autocast |
| energy accumulation | float64 |
| validation interval | step 1、500ごと、最終step |

### 段階的な規模拡張

| 段階 | 範囲 | Validation dB | 扱い |
|---|---|---:|---|
| Neighbor proxy 1 | FFID 2348–2363 | 16.182384937077 | リーク安全 proxy、閾値超過 |
| Neighbor proxy 2 | FFID 2314–2382 | 16.513428561703 | 凍結条件の replication |
| 初回 all-survey proxy | 全 eligible survey | 18.060750804746 | 重複物理セルのため正式不採用 |
| 初回 proxy から重複2 validation行だけ除外 | 114,490 traces | 18.060753714290 | 方向性確認のみ |
| Canonicalized formal run | 全4,780 eligible FFID | **18.111870025657** | **正式成功** |

初回 all-survey proxy は、重複 validation 2行を評価から除外しても18 dBを維持したため、
結果が twin trace だけに依存していないことは確認できた。しかし学習とcheckpoint選択時には
重複が存在したので、成功証拠にはせず、global canonicalization 後にseed 42からfreshに
学習し直した。

## 正式 run の結果

正式 run は Git commit `edb2561ffa731f02e2c87325ba340ebce9671104`、seed 42、
NVIDIA H100 NVL (`cuda:1`) で実行した。開始は `2026-08-28T19:46:29Z`、終了は
`2026-08-28T19:48:52Z` だった。

| Step | Validation global S/N (dB) |
|---:|---:|
| 1 | 0.622755976 |
| 500 | 15.800804348 |
| 1,000 | 16.921890913 |
| 1,500 | 17.620158303 |
| 2,000 | 17.991833245 |
| 2,500 | **18.111870026** |

主要な最終値は次のとおりである。

| 項目 | 値 |
|---|---:|
| `oracle_per_trace_unit_rms_global_snr_db` | **18.111870025656728** |
| 閾値との差 | +3.111870025656728 dB |
| best step | 2,500 |
| validation trace数 | 114,490 |
| validation point数 | 71,556,250 |
| signal energy | 71,556,250.00323877 |
| error energy | 1,105,250.118569839 |
| error mean square | 0.015445892127799305 |
| prediction self-normalized診断 | 18.103393576670555 dB |
| train leave-one-out audit | 18.104445832103483 dB |
| CUDA peak allocated | 9,177,508,352 bytes |
| CUDA peak reserved | 11,316,232,192 bytes |

保存後にcheckpointをstrict loadし直し、保存値と再計算値がともに
18.111870025656728 dBであることを確認した。checkpoint は104 neighbors、width 128、
983,041 parametersとして復元できた。

## 実装した内容

### SIREN 切り分け用の再利用機能

- complete-trace survey batch
- Cartesian half-offset座標
- optional offset radius
- cosine learning-rate schedule
- temporal coordinate scale
- layer-wise exponential omega schedule
- dense SIREN skip connection

### Neighbor inpainter の再利用実装

- [`neighbor_trace_inpainter.py`](../src/seis_interp/models/neighbor_trace_inpainter.py):
  temporal CNN model
- [`neighbor_geometry.py`](../src/seis_interp/processing/neighbor_geometry.py):
  物理格子と104 neighborのindex
- [`neighbor_inpainter_trainer.py`](../src/seis_interp/training/neighbor_inpainter_trainer.py):
  optimizer、schedule、loss、validation、checkpoint選択
- [`neighbor_inpainter_checkpoints.py`](../src/seis_interp/training/neighbor_inpainter_checkpoints.py):
  strict save/load
- [`train_neighbor_inpainter.py`](../src/seis_interp/pipelines/train_neighbor_inpainter.py):
  canonicalization、train-only振幅routing、正式scope監査、run出力
- [`interim_trace_dataset.py`](../src/seis_interp/data/interim_trace_dataset.py):
  leakage-sensitive pipeline向けの選択行value validation
- [`cli.py`](../src/seis_interp/cli.py):
  `train neighbor-inpainter` command

主要コミットは次のとおりである。

| Commit | 内容 |
|---|---|
| `0b32f20` | mixed complete-trace survey batch |
| `434f752` / `5ba3b12` | Cartesian half-offset / radius |
| `2d4e082` / `813d85a` | cosine schedule / time scale |
| `61941b2` / `ea0e042` | layer omega schedule / dense skip |
| `819c3de` | neighbor trace model |
| `3049907` | physical neighbor geometry |
| `615fafe` | neighbor trainerとcheckpoint |
| `75301ed` | 正式pipeline、CLI、integration tests |
| `edb2561` | CUDA memory counterのportable reset |
| `bd31e6f` | 正式成功をStudy 017へ記録 |

最初の正式起動では、このPyTorch/CUDA buildが
`torch.cuda.reset_peak_memory_stats(device)` の明示device引数を拒否したため、学習更新前に
停止した。current-device context内の引数なし呼び出しへ修正し、cuda:0/1の両方でsmoke
test後に新しいrun IDで再実行した。失敗した起動はrun成果物を作成していない。

## 正式 run の成果物

Run directory:
[`runs/study_017_all_ffid_neighbor_inpainter/20260828T194620Z_edb2561_all_ffids`](../runs/study_017_all_ffid_neighbor_inpainter/20260828T194620Z_edb2561_all_ffids)

| ファイル | SHA-256 |
|---|---|
| `config.resolved.yaml` | `f83629ff591896fbfb4a5b5500d1e8b80113dac97f746de2a01774173592b718` |
| `inputs.lock.json` | `4c28c8a1a2a19c94c9bcdcc72662d05af7824b697412f6f3575b9f2dcf2243c2` |
| `metrics.json` | `8913960274405c663300946deb1b838f209ccf7b2de35e00667f2b436fed88c6` |
| `run.json` | `b9258b999a5995fa29956482ee3fc3cae12853149caaf93191838e6278e14603` |
| `artifacts/best.pt` | `7ccd3c1c013927793b4cd25ad741b0100d73559ad9ec9a175a2f8097ad2af53b` |

## テストと品質ゲート

正式結果の記録後、リポジトリ指定の品質ゲートを実行した。

| Gate | 結果 |
|---|---|
| `ruff check .` | pass |
| `ruff format --check .` | pass、183 files formatted |
| `pytest` | **1,036 passed in 43.01 s** |
| `python -m seis_interp.cli doctor` | exit 0 |

`doctor` は Python 3.10.12、PyTorch `2.5.0a0+b465a5843b.nv24.09`、CUDA利用可、
NVIDIA H100 NVL 2台、NumPy/Pandas/PyArrow/segyio、Codex、Claude Code、data rootを確認した。

追加したテストは、model shape/parameter数、neighbor順序、same-source-x制約、center除外、
train-only lookup、checkpoint strict load、loss/schedule、cross-split duplicateの
canonicalization、scope drift拒否、15 dBのstrict比較、test/excluded NaN値を参照しないこと、
immutable outputs、input hash、Git SHA、seed、CLIを対象とする。

## 再実行方法

準備済みデータがある場合は、repository rootから次を実行する。

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_all_ffids"

python -m seis_interp.cli train neighbor-inpainter \
  --config studies/study_017_all_ffid_neighbor_inpainter/config.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_per_ffid_random_split_amplitude_qc \
  --output "runs/study_017_all_ffid_neighbor_inpainter/$RUN_ID" \
  --device cuda:0
```

入力のsource/version/SHA-256、正規化前後のsplit件数、model/training条件は
[`inputs.yaml`](../studies/study_017_all_ffid_neighbor_inpainter/inputs.yaml) と
[`config.yaml`](../studies/study_017_all_ffid_neighbor_inpainter/config.yaml) に固定している。

## 制約

- primary metric はvalidation target自身のRMSを使うoracle waveform指標である。未知traceの
  物理振幅gainは別のtrain-only gain modelで推定する必要がある。
- 成功モデルはneighbor waveformを条件とし、coordinate-only implicit fieldではない。
- 正式runはSEG C3 NA、seed 42、単一model条件での検証であり、別survey・別seedへの
  generalizationは未検証である。
- validationはcheckpoint選択に使用し、test targetは評価していない。次の研究では固定済み
  test splitの最終評価条件を別途定義する必要がある。
- `inputs.yaml` のsource SHAは今回の実データと照合済みでrunにも記録したが、pipelineの
  formal scope判定は主として件数・sample数・excluded FFID・リーク監査を強制している。

## 最終判断

coordinate-only SIRENの改善だけでは15 dBを達成できなかった。一方、targetを除外した
train-only物理近傍波形をtemporal CNNへ与える条件は、16 FFID、69 FFID、全surveyの順に
再現して15 dBを超えた。重複物理セルを全splitの前段でcanonicalizeしたfresh formal runでも
18.111870025656728 dBを得て、scope、リーク、checkpoint再評価、品質gateをすべて通過した。
したがって、指定されたoracle波形指標に対するPOCは成功と判断する。
