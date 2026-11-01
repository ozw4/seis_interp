# 感度 run の準備（2026-11-01、未起動）

本番 launch-frozen の YAML・既存 production code・`runs/` は変更しない。clean commit の作成・tag・通常 prepare・起動は operator が行う。本作業では GPU を隠し、公式 `prepare --check-only` だけを使用した。費用は外部負荷なしの事前登録値で、壁時計の保証ではない。

| spec | 固定 cell / 出力 | production method YAML との差 | GPU-h | minimum / margin 込み GiB |
|---|---|---|---:|---:|
| [sa_forge_mask010](sa_forge_mask010/config.yaml) | `a03_b02`、S1 validation のみ | `patches.inner_mask_fraction: 0.25 → 0.1` | 10–14 | 17 / 19 |
| [sb_forge_fp32](sb_forge_fp32/config.yaml) | `a03_b02`、S1 validation のみ | `training.mixed_precision` と `prediction.mixed_precision`: `bf16 → 'off'` | 15–25 | 32 / 34 |
| [sa_c3_mask010_seeds](sa_c3_mask010_seeds/config.yaml) | `a05_b02`、7321–7323、validation + test | `patches.inner_mask_fraction: 0.25 → 0.1` | 各約8、計約24 | 32 / 34 |

S-a の method YAML は置換した1行以外 byte-identical。S-b は**精度という1因子を、実装上の2 leaf key で変更**する。片方だけの変更では全経路の FP32 にならない。その他は byte-identical。`inputs.yaml` は対応 production と byte-identical、local grid は共通 grid の CCNet 25候補を ID・順序ごと保持する。config の変更は pilot routing（CCNet のみ、local grid、`common_grid` を外す、`method_execution: {}`、device を `cuda:0` だけにする、import 元）と S-b 当該 cell の予算に限る。C3 の `fixed_cell_selection` は旧 winner 選択ファイルの path / SHA256 を固定する実行用 binding である。数学条件・seed・holdout は保持する。S-b の他 cell の予算は FP32 用ではないため、通常の `run` / `worker` で25候補を実行してはならない。

実際の method 差分は次のとおり。

```diff
# sa_forge_mask010 と sa_c3_mask010_seeds（各1行）
-  inner_mask_fraction: 0.25
+  inner_mask_fraction: 0.1
# sb_forge_fp32（training と prediction の各1行）
-  mixed_precision: bf16
+  mixed_precision: 'off'
-  mixed_precision: bf16
+  mixed_precision: 'off'
```

`mixed_precision_dtype('off', ...)` は autocast dtype を返さず、学習・予測は既存の FP32 path を通る。`_fit_ccnet` → `train_ccnet5d_observed_steps` と `predict_ccnet5d_tiled` はともに off をサポートする。bf16 専用 model はない。既存の `allow_tf32 = True` は維持し、登録上の FP32 は FP32/TF32 を意味する。厳密な IEEE FP32 へ勝手に変えない。

C-0 の bf16 a03/d6 の train / prediction peak は 15.0 / 14.5 GiB。保存済み C-0 end-to-end probe は bf16 a01/a05 であり、今回の a03/d2 FP32 と一致する実測はない。S-b は設計の約30 GiBを暫定 peak として `max(17, ceil(30) + 2) = 32` GiB、別の admission margin 2 GiBを加えて34 GiBとする。これは未測定の推定であり、FP32 実測として報告しない。S-a FORGE は既存17 GiB、C3 は既存32 GiBを保持する。

## C3 の固定 cell と S1 import

study_064 `20261012T235609226847Z_05c92e79ce80_traces/workers/ccnet5d_grid_selection.json`（SHA256 `0206fbdb0544c0b2eeb891a33aad01d1f705d61ae4105dc2e3843803cb527c58`）の validation relMSE winner は **a05_b02**、patch `[384,8,16,8,16]`、time dilation 2、tk3 / hidden53 / intermediate63。Sprint 3 の受入 cell と一致する。旧 run の19.073 dBは既存の登録比較値で、新 seed の値として扱わない。

旧 resolved YAML と production C3 `a05_b02` の文字どおりの差は mask だけではない。旧側には `execution.reuse_patch: false` / `execution.cache_patch_gradients: false`、`execution.cudnn_benchmark: true`、shadow EMA の明示がない。前2者は現在の fitter の absent default も false、後2者は identity 除外だが、**前2者の absent と明示 false を現行 hash は同一視しない**。旧 hash は `8af014bd27b29655285e72a81e17a76eeab2f24fd8279882194789cb8012b490`、今回の mask 0.1 は下表の `e470...`。公式 check-only の C3 S1 import は **0/25**、新評価 seed も0/3。入力 fingerprint は旧 S1 と一致しても、identity import は不可である。さらに旧 frozen fitter は placement / mask に同じ7301を渡し、現在は `SeedSequence(7301).spawn(3)` を使う。旧 S1 を同一実行として認める identity 変更や互換処理は加えない。今回の新 seed は現行 production と同じ RNG path、mask だけを変える。

既存 production worker は25候補を必須とし、pilot は seed を実行しない。新しい [run_seeds.py](sa_c3_mask010_seeds/run_seeds.py) は frozen `source/src` の専用 [pipeline](../../../src/seis_interp/pipelines/run_c3_fixed_cell_sensitivity.py) を呼ぶ。準備された入力全体と本番入力の byte hash、振幅入力 binding、7321–7323、mask 以外の method identity、旧 winner 選択ファイルの hash、本番の全手法 S1 barrier を検証する。本番 barrier は read-only で確認する。感度 experiment 側に、外部固定 cell と本番 barrier の provenance を明記した test-access marker を作り、**本番の公開 `execute_c3_region_run` を変更せず** evaluation を逐次3回実行する。25候補を選択したことにはせず、S1 manifest / selection JSON / rehearsal を作らない。通常の deadline、retry、checkpoint、admission、partition loader、target reader、scorer、per-trace parquet と seed summary をそのまま使う。S1 の追加訓練コストは0で、歴史的 S1 値の import とも主張しない。

CPU synthetic test では同一 seed の partition、採点行、指標、per-trace parquet の列・dtype・行順・値を production 関数と比較する。入力不一致、seed 不一致、model 変更、旧選択 hash 不一致、未通過 barrier、rehearsal、同一出力先は評価前に拒否する。完了後の再実行では test を再採点しない。本番側への書込みがないことも確認する。

## Check-only identity

`prepare --check-only` の一時 preparation 上で production と同じ観測数・alignment を適用し、`method_config_hash` を計算した。device は既存の execution-only 除外。全4 seed で各 method hash は同じで、production との差は上記の mask または精度だけ。各 seed の入力 fingerprint は production と一致し、C3 は `input/` 全 artifact hash も一致した。temporary preparation は終了時に削除する。

| spec / cell | method config hash |
|---|---|
| production FORGE `a03_b02` | `ffb7f05a8659bb2b0f2d4556107cec6ffb68598acaf1a360bb13c814dcd30a5b` |
| S-a FORGE `a03_b02` | `8e6810e4ba400ccebb56db2a0df36f4c794f0f5e84a03c0ccec2b11a4b46adc3` |
| S-b FORGE `a03_b02` | `5ea2caa71786f21030968bf9e2640b05ece23b51a8fe7b8ababde92020accfd0` |
| production C3 `a05_b02` | `6a55abbe6eb630d8127f3defa311cb0aa134255afe5495002af37e45596c53da` |
| S-a C3 `a05_b02` | `e470fc0b5130f9f84b2c94b990324980c9b045a21897b3f3dd99f446e7558301` |

| dataset / seed | input fingerprint（production と共通） |
|---|---|
| FORGE 600702 (S1) | `2ad487443904e5007f3a59569a2d14937d1f063de1728101ef88e0b8ed674238` |
| FORGE 7321 | `8c5224e3c34af6acd80e8b7ca923ead29a7781d90d8138ff92759496a986b982` |
| FORGE 7322 | `f148676d2f868d079c3a2d07643cd41042a2cccc746be15747a1d531688aac4c` |
| FORGE 7323 | `2fa1ec4daaec9e08b29449fe335eafe18151a8e19f4040c017d18b7c68daab8e` |
| C3 620702 (S1) | `1ce5bd665bfd4d113c3fe2d8fa71766aef154d81359ccd99fafd33eaa46eef51` |
| C3 7321 | `6c347edd585e2fb3312c4f778cf63510241c955679166a52aa57fdadbcd965be` |
| C3 7322 | `dd75568d0a905f8117194d15ca3f67eebf8970f6cbf1e7b24da3bbe38fe63382` |
| C3 7323 | `f390dce1f4e1c4996a8c02686a9ada207e6991d36f9dab0ddbebeb4531ae1657` |

FORGE 両 arm の import は source を空にして0/25（新規S1）。C3 は study_064 だけを source に指定して0/25。coverage の25は grid 定義上の数で、起動数は上表の1 / 1 / 3。

## GPU0 pinning と admission

新 launcher は既存 `launch_study067_pilots.start` を使い、physical GPU0 の完全な UUID による `CUDA_VISIBLE_DEVICES`、frozen `PYTHONPATH`、BLAS threads=1、`nice -n 19` で detached process を作る。親コマンドの `CUDA_VISIBLE_DEVICES=0` は通常 CUDA ordinal 0 のみを可視にするが、physical index との順序一致が前提となる。launcher は起動時だけ `nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader` で physical GPU0 の完全 UUID を得て子へ設定するため、CUDA enumeration 順序が異なっても physical GPU1 は不可視となる。dry-run は照会せず数値0を表示する。config の許可 device も `cuda:0` だけにする。`auto` は admission が返した `cuda:0` に置換され、CUDA は可視集合外の GPU1 に context / allocation を作れない。親から可視性は子へ継承する。launch は全て今後の operator 操作であり、この準備では行わない。

`gpu_memory_info` は NVML の UUID / free / total を取得し、数値 selector 0 を physical enumeration index 0へ写す。CUDA primary context は作らない。本番の生存 admission slot は `device: cuda:1`、当該 PID の `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` は未設定だった。したがって今回の GPU0 pilot の `cuda:0` と無制限本番の `cuda:0` は同じ namespace になる。共有先は両方 `/workspace/runs/.gpu_admission/`、startup lock は同じ `admission.lock`。他 job の `required - current_reserved` の正の差を同じ device key で free memory から引き、90%到達または60秒で startup lock を外しても再成長予約を保持する。GPU1 slot の予約は GPU0 の free から差し引かない。legacy PID exclusion も `cuda:0` の意味が一致する。

**一般形の制約**：slot は UUID ではなく論理 `cuda:N` を保存する。`CUDA_VISIBLE_DEVICES=1`、`1,0`、GPU1 の UUID pinning と無制限本番の混在では同じ物理 GPU の slot key が食い違う。将来の修正は free memory・slot・候補予約・exclusion・startup の比較 key を全て physical UUID へ揃え、torch 操作用に論理 index を別保持すること。稼働中の production code は変更しない。数値0の CUDA ordinal と NVML index が同じであることが現行コードの前提でもある。異機種 / enumeration が違う環境では数字だけで保証せず、operator が GPU0 の UUID と CUDA ordinal 0 の対応を確認する。今回 GPU0 以外への reordering は使わない。

GPU0 の宣言容量93.08 GiB、context allowance 2 GiBに対し、3種類各1 run の要求は **19 + 34 + 34 = 87 GiB**。本番や外部負荷のない GPU0 なら最大3 process（FORGE2 + C3 seed1）を相乗り可能。C3 専用入口は3 seedを逐次にするので同時 seed は1。C3 seedを3個同時にすると102 GiBで不可。予約と実 free によって admission が待機させるため、3起動が即時開始を保証しない。S-b の未測定メモリも考慮し、operator が保守的に逐次実行してもよい。

## Operator commands（この作業では実行しない）

以下は operator が作る commit の clean checkout `/workspace` で実行する。`progress.sh` はローカル除外のままでよい。通常 prepare は新しい感度 run だけを `runs/` に作る。既存2 production run には書き込まない。

```bash
cd /workspace
PY=/workspace/.venv/bin/python
STUDY=studies/study_067_forge_c3_common_grid
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
test -z "$(git status --porcelain)"
git rev-parse HEAD

SA_FG=$(CUDA_VISIBLE_DEVICES='' nice -n 19 "$PY" scripts/run_forge_grid_experiment.py \
  prepare --repo /workspace --study "$STUDY/pilots/sa_forge_mask010")
SB_FG=$(CUDA_VISIBLE_DEVICES='' nice -n 19 "$PY" scripts/run_forge_grid_experiment.py \
  prepare --repo /workspace --study "$STUDY/pilots/sb_forge_fp32")
SA_C3=$(CUDA_VISIBLE_DEVICES='' nice -n 19 "$PY" scripts/run_c3_region_experiment.py \
  prepare --repo /workspace --study "$STUDY/pilots/sa_c3_mask010_seeds")
C3_PRODUCTION=/workspace/runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces

CUDA_VISIBLE_DEVICES=0 "$PY" scripts/launch_study067_sensitivity.py sa_forge_mask010 --experiment "$SA_FG"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/launch_study067_sensitivity.py sb_forge_fp32 --experiment "$SB_FG"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/launch_study067_sensitivity.py sa_c3_mask010_seeds \
  --experiment "$SA_C3" --production "$C3_PRODUCTION"
```

launcher は FORGE では `$EXPERIMENT/source/scripts/run_forge_grid_experiment.py pilot --method ccnet5d_grid --candidate a03_b02`、C3 では `$EXPERIMENT/study/run_seeds.py --production ...` を使う。双方とも frozen `$EXPERIMENT/source/src` のみを import する。`--dry-run` を上記 launch command に加えれば GPU 照会・process 起動・log 作成をせず command / environment を表示する。再開は同じ実験と別の `--log sensitivity_resume1.log` で行う。元のログを上書きしない。

監視は以下。GPU状況を照会する最後のコマンドも operator 向けで、準備中は未実行。

```bash
tail -n 30 "$SA_FG/sensitivity.log" "$SB_FG/sensitivity.log" "$SA_C3/sensitivity.log"
CUDA_VISIBLE_DEVICES='' "$PY" "$SA_FG/source/scripts/run_forge_grid_experiment.py" status --experiment "$SA_FG"
CUDA_VISIBLE_DEVICES='' "$PY" "$SB_FG/source/scripts/run_forge_grid_experiment.py" status --experiment "$SB_FG"
CUDA_VISIBLE_DEVICES='' "$PY" "$SA_C3/source/scripts/run_c3_region_experiment.py" status --experiment "$SA_C3"
cat "$SA_FG/method_runs/ccnet5d_grid/S1/a03_b02/manifest.json"
cat "$SB_FG/method_runs/ccnet5d_grid/S1/a03_b02/manifest.json"
cat "$SA_C3/workers/ccnet5d_grid.json"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

check-only の再確認は通常 prepare block の代わりに次を使う。

```bash
CUDA_VISIBLE_DEVICES='' nice -n 19 "$PY" scripts/run_forge_grid_experiment.py prepare --repo /workspace --study "$STUDY/pilots/sa_forge_mask010" --check-only
CUDA_VISIBLE_DEVICES='' nice -n 19 "$PY" scripts/run_forge_grid_experiment.py prepare --repo /workspace --study "$STUDY/pilots/sb_forge_fp32" --check-only
CUDA_VISIBLE_DEVICES='' nice -n 19 "$PY" scripts/run_c3_region_experiment.py prepare --repo /workspace --study "$STUDY/pilots/sa_c3_mask010_seeds" --check-only
```
