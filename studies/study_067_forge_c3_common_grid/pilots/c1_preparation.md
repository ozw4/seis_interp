# C-1 実験と identity（2026-10-23）

CPU の公式 `prepare` で以下を凍結し、4実験とも完了。候補 ID は共通 grid のものを保持する。T1 不成立により両 dataset で tk3 を維持する。完了 manifest の SHA256、指標、runtime、shadow EMA と import 検証は [c1 artifact](../round2_outcomes.json)、判定は [decisions.md](../decisions.md) を参照。この記録作業では実験を起動しない。

| arm | cell | parameters | prepared directory（`runs/study_067_forge_c3_common_grid/` 配下） |
|---|---|---:|---|
| c1_c3_tk3 | `a05_b02` | 363,292 | [20261022T202519313159Z_05c92e79ce80_traces](../../../runs/study_067_forge_c3_common_grid/20261022T202519313159Z_05c92e79ce80_traces) |
| c1_c3_tk9 | `a05_b02` | 363,227 | [20261022T202537703605Z_05c92e79ce80_traces](../../../runs/study_067_forge_c3_common_grid/20261022T202537703605Z_05c92e79ce80_traces) |
| c1_forge_tk3 | `a03_b02` | 363,292 | [20261022T202533191991Z_05c92e79ce80_forge](../../../runs/study_067_forge_c3_common_grid/20261022T202533191991Z_05c92e79ce80_forge) |
| c1_forge_tk9 | `a03_b02` | 363,227 | [20261022T202604622878Z_05c92e79ce80_forge](../../../runs/study_067_forge_c3_common_grid/20261022T202604622878Z_05c92e79ce80_forge) |

C3 は FP32・32 GiB/arm、FORGE は bf16・17 GiB/arm。mask 0.25、fresh 更新、cuDNN benchmark on、主 EMA 0.999 と shadow 0.999 / 0.9995 / 0.9998。

現行 S1 を一時領域へ公式 prepare し、validation domain の保存入力 fingerprint と resolved config identity を照合した。tk3 は現行設定、tk9 は S1 の model 3 key（time kernel 9 / hidden 36 / intermediate 37）を採用した場合に一致する。現在は両 dataset とも tk3 のまま。

| dataset | 共通 input fingerprint（tk3 / tk9 / 対応 S1 で一致） |
|---|---|
| c3 | `1ce5bd665bfd4d113c3fe2d8fa71766aef154d81359ccd99fafd33eaa46eef51` |
| forge | `2ad487443904e5007f3a59569a2d14937d1f063de1728101ef88e0b8ed674238` |

| arm | 対応構造の S1 と一致した method config hash |
|---|---|
| c1_c3_tk3 | `6a55abbe6eb630d8127f3defa311cb0aa134255afe5495002af37e45596c53da` |
| c1_c3_tk9 | `21bce2dc295d42648304c89156a69bdb8c5ee17bdccd6a5ecaba5390232cbee4` |
| c1_forge_tk3 | `ffb7f05a8659bb2b0f2d4556107cec6ffb68598acaf1a360bb13c814dcd30a5b` |
| c1_forge_tk9 | `c1e52408cd3ad8dc0f8a0ffc4389fbe91fb6dbb237157d902661c1207995926f` |

両 arm を `import_from` に宣言したまま、各 source 単独の `find_prediction_import` を CPU で検証した。完了 tk3 は C3 `a05_b02` / FORGE `a03_b02` の S1 として再利用できる。両 tk9 は input fingerprint が一致しても time kernel / hidden / intermediate の identity 不一致で拒否される。本番 CCNet YAML と `import_from` は変更していない。

起動用コマンド（この作業では未実行）。`GPU_UUID_C3` / `GPU_UUID_FORGE` へ完全な physical GPU UUID を設定して使う。末尾に `--dry-run` を加えると process 起動・GPU 照会・ログ作成を行わず4コマンドを表示する。

```bash
.venv/bin/python scripts/launch_study067_pilots.py c1 \
  --gpu-uuid-c3 "${GPU_UUID_C3:?set full C3 GPU UUID}" \
  --gpu-uuid-forge "${GPU_UUID_FORGE:?set full FORGE GPU UUID}"
```

launcher は source snapshot 内の公式 runner を `pilot --method ccnet5d_grid` で実行し、validation のみを採点する。各実験の `c1.log` へ出力し、detached PID を表示する。expandable_segments を有効にする。2 arm 同時の空きメモリ要件は margin 込みで C3 68 GiB、FORGE 38 GiB。N-1 と同様に launcher で physical UUID・legacy PID の所在を確認し、pilot の global admission は off とする。
