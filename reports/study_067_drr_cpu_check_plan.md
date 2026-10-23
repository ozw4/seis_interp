# study_067 DRR 本番 backend の CPU 受入計画

この文書は実データ計算の計画であり、Sprint 1 では実行しない。既存 run と scratch の `quick_run.py` / `results.jsonl` は読み取り専用とし、結果は新規の pilot experiment に保存する。scratch の所在は承認設計を提示したユーザーの指示に従う。

本番の `run_c3_region_method` / `run_forge_c3_method` は、実効 overlap 0 のとき旧 `interpolate_drr_nonoverlapping`、overlap > 0 のときだけ `interpolate_drr_volume_batched` を通す。backend / boundary_rule は resolved config と import identity に含める。FORGE は観測だけを oracle RMS で model-space 化し、共通の 500 m/s 整列 canvas 上で補間、unalign と RMS 復元後に validation の clean_target を採点する。C3 は整列と trace RMS を使わない。test の waveform は読まない。

| 受入対象 | dataset | rank / window | 反復 | overlap | validation mean trace SNR の基準 |
|---|---|---|---:|---|---:|
| D5a・旧 backend 照合 | C3 | 12 / [4,8,4,8] | 30 | [0,0,0,0] | 14.004714184959756 dB |
| D5b | C3 | 12 / [4,8,4,8] | 30 | [0,4,0,4] | 14.761245437471583 dB |
| D5b | FORGE | 12 / [4,8,4,8] | 30 | [0,4,0,4] | 1.0894010827668017 dB |
| D7 | C3 | 12 / [4,8,4,8] | 60 | [0,4,0,4] | 16.087339661975765 dB |
| D7 | FORGE | 12 / [4,8,4,8] | 60 | [0,4,0,4] | scratch 完了時の results.jsonl の D7 値を読み取り、固定して照合 |

受入誤差は各ペアで 0.01 dB 以内。D-1 自体の採否は本番再現とは別に、FORGE ≥ 0.656、C3 ≥ 15.125 の登録済み T2 を使う。D7 FORGE の結果は Sprint 1 時点で未確認であり、予測値で補わない。

手順は以下の通り。

1. 同じ S1 partition を使う新規 pilot study を、既存の C3 / FORGE 入力から prepare する。`drr_grid.yaml` で rank 12、window [4,8,4,8]、damping 2、全帯域、上表の反復数を指定する。overlap ありは `algorithm.overlap_policy: point_half`、なしは `explicit` / `[0,0,0,0]`。backend / boundary_rule は実効 overlap から導出し、overlap ありは `drr_batched` / `clip_and_shift`、なしは `drr_nonoverlapping` / `unknown_padding` が resolved config に記録されることを確認する。元の study_067 config は変更しない。
2. `execution.workers: 1`、`cpu_threads: 1`、`batch_matrix_limit: 2000000` で開始する。この limit は各 worker の SVD call の Hankel 要素数を制限する（1枚の行列が limit を超えるときだけ1枚が最小単位）。2049 bins を一括処理しない。RSS を記録する。
3. 下記の本番 pilot 入口で `a02_b02` を実行する。`pilot` は validation のみを採点し、barrier 後の seed / test 評価には進まない。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_c3_region_experiment.py pilot \
  --experiment <NEW_C3_DRR_PILOT> --method drr_grid --candidate a02_b02
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_forge_grid_experiment.py pilot \
  --experiment <NEW_FORGE_DRR_PILOT> --method drr_grid --candidate a02_b02
```

4. manifest の `test_scoring_calls: 0` と、C3 の `metrics.validation.mean_trace_snr_db` / FORGE の `metrics.validation.clean_target.mean_trace_snr_db` を確認する。同じ入力の予測、unalign、scoring を経たことを確認し、scratch 値との差を保存する。
5. 周波数 chunk の数値不変性は、同じ cell を小さい limit と 2,000,000 の2条件で比べる。overlap > 0 で window が割り切れない C3 / FORGE の軸では、window を軸長へ切り詰め、最後を volume 内へずらす。非空 window を正の interior-Hann 重みで合成し、観測を厳密に復元する。overlap 0 は未知 padding の旧規則を使う。非整除 window でも overlap 0 の grid adapter が旧 backend と bitwise 一致することを tiny array で検証する。
6. D-1 採用後の D-2 は rank 30 × [8,16,8,16] および [8,12,8,12] を両 dataset で1–2反復、worker 1、RSS 上限付きで測る。FORGE [8,16,8,16] の60反復は候補あたり数十時間、worker 1 の D-2 自体も数時間以上になり得る。「CPU で数十分」を計画値に使わず、この測定で全コストと並列度を決める。RSS cap はホストの空き RAM を確認して開始前に固定し、既存 process に signal を送らず、専用 job 自身の資源制限で適用する。測定から全60反復、S1、seed のコストと同時 worker 数を決める。72 h / cell または FORGE GPU critical path 超過は開始前に報告する。

CPU unit tests は、周波数分割の各 SVD 呼出しの上限、overlap 0 の旧 backend 一致、非整除 shape の clip-and-shift と Hann 合成を検証する。上記の実データ計算を実施した証拠にはしない。
