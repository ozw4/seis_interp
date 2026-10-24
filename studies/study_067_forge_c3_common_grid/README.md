# study_067: C3 / FORGE 共通 25 候補と 3 seed 比較

## 評価（Sprint 3）

評価の正本は [evaluation.yaml](evaluation.yaml)。S1 の validation relMSE 最小（同値は candidate ID、diverged は最下位）で選び、各 seed 自身の test 集合を使う。source-cell bootstrap の重みは seed・手法・対比・指標で共有し、SNR 8 対比と relMSE 8 対比を別々の Holm 族とする。全 seed 終了後、repository root から次を実行する。保存済み run は読み取り専用で、結果は指定した新規出力ディレクトリへ保存する。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nice -n 19 .venv/bin/python scripts/evaluate_study067.py evaluate \
  --c3 runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces \
  --forge runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge \
  --output runs/study_067_forge_c3_common_grid/analysis_final
```

`contrasts.csv` に seed 別差・percentile CI・両側 p・Holm・df=2 の t 区間、`contrasts_validation_free.csv` に S1 validation の和集合を除いた感度を保存する。`seed_summary.csv` は3 seed の等重み平均・標本 SD・各 seed 値、`per_seed_metrics.csv` は validation/test の副次指標、`ema_shadow.csv` は保存済み shadow の値と欠測を表す。`selection.csv`、`expected_best_of_k.csv`、`winners_curse.csv`、`cost.csv` と、dataset/手法ごとの heatmap・曲線・派生 per-trace parquet も出力する。`manifest.json` に解析コード一式・評価設定・grid・各設定と run manifest の SHA256、Git commit/tag を残す。

diverged 評価 seed があれば3 seed 平均と該当 CI は算出不能とし、残り seed だけで代用しない。欠けた対比を含む8対比の族では Holm も欠測とする。winner's curse は S1 の source-cell bootstrap による in-bag 選択と out-of-bag 評価の差を記述する。expected-best-of-k は復元抽出で、diverged だけを引く確率を保持し、有限候補を含む条件付き曲線を併記する。追加 pilot の入力欄は空であり、S-a / S-b、古典手法の dataset 別上限、Q3(b) は含めない。

ReGSI（提案法）、CCNet5D、profile-wise NeRSI、POCS、DRR を、各手法につき両 dataset で同じ二軸の探索により比較する。現在は実行仕様を準備した段階であり、本 study の比較結果は未確定。POCS は第2弾 P-1 の K1 を両 dataset に採用し、ReGSI は R-1 を通過した V3 を維持する。

ReGSI P2（k=2、dilations `[1,2]`）の validation mean trace SNR は baseline / V1 / V2 / V3 の順に FORGE 1.8990 / 2.2110 / 2.7100 / 3.0070 dB、C3 18.8600 / 19.1400 / 19.1400 / 19.0700 dB。V3 は FORGE +1.1080 dB、C3 +0.2100 dB で、事前規則（FORGE ΔSNR ≥ +0.2 dB、C3 ΔSNR ≥ −0.2 dB を満たすうち FORGE 改善最大）により採用する。全 25 候補は 346,885 parameters（363,269 比 −4.5102%、既存の ±5% tolerance 内）を共有する。数値は提示された丸め値である。

NeRSI P3 の C3 validation mean trace SNR は jitter 0.5 で 12.6600 dB（jitter なし baseline 13.5100 dB）。事前規則の C3 ≥ 10.5 dB を満たす。FORGE P3 も完了し、validation mean trace SNR は −0.6900 dB（設計書の比較基準 −0.689 dB）、relMSE は 1.2140 で床未達。N-1 / N-2 の登録済み規則を適用した結果は下記のとおり。

第2弾の確定結果：R-1 は C3 19.9140 dB（relMSE 0.012633）≥ 19.8120 dB で V3 を維持。P-1 は C3 13.7920 dB（Δ +2.9150）、FORGE 1.1650 dB（Δ +0.2700）で T2 を満たし、K1 を採用する。P-1 は **screening 由来（事後的、開示）**。C-0 の cuDNN benchmark は1.34倍で Q1(i) の1.3倍を超え、CCNet の両 dataset で on を採用。FORGE CCNet S1 は約406 GPU-h（報告閾値450未満）。実測値と run 証拠は [decisions.md](decisions.md) を参照。

G0：[end-to-end probe](../../runs/study_067_forge_c3_common_grid/20261022T192704131837Z_05c92e79ce80_ccnet_memory_probe/memory_probe.json) の fresh 1000 更新で **両 cell PASS**。更新 901–1000 の L0 / 平均 loss 比 / 出力 RMS 比は、`a01_b02`: 0.46086556168940895 / 0.918636 / 0.284084、`a05_b02`: 0.46086556169270365 / 0.842538 / 0.394574。両方とも全値有限で、[設計 gate](../../reports/study_067_design_round2.md)（loss 比 ≤ 0.95、RMS 比 ≥ 0.2）を満たす。

N-1 / N-2：**両 dataset の本番 YAML に upsampling を適用済み（FORGE s=3、C3 s=0）、support mask は不採用、jitter と primary EMA を保持**。[採用 arm の manifest](../../runs/study_067_forge_c3_common_grid/20261022T183518698062Z_05c92e79ce80_forge/method_runs/nersi_profile/S1/a02_b04/manifest.json) は -0.036867405 dB / relMSE 1.045480641。全 arm と規則 (i)–(iv) の適用は [decisions](decisions.md) / [artifact](round2_outcomes.json) に記録。shadow EMA の改善は +0.008805205 dB < [N-2 閾値 +0.15 dB](../../reports/study_067_design_round2.md) で不採用、C3 の追加 run は不要。床救済は終了し、FORGE の床0.4 dB未達を予想し、追加調整しない。ユーザー承認により両 `nersi_profile.yaml` に `canvas_time_upsampling: true` を適用した。FORGE は採用 N-1 arm の `a02_b04` を identity / input fingerprint の一致確認後に import し、upsampling のない P3 は identity で拒否する。C3 は s=0 のため構造と import identity が変わらず、NeRSI coverage は 1 / 24 のまま。

D-1：**両 dataset に window 由来 Hann overlap を採用（screening 由来、事後的、開示）**。[再現 artifact](round2_outcomes.json) の FORGE D7 / C3 D7 / FORGE D5b はそれぞれ 1.097869356 / 16.087339662 / 1.089401083 dB、全て差0.0 dBで check-d1 PASS。[T2 閾値](../../reports/study_067_design_round2.md) FORGE ≥ 0.656、C3 ≥ 15.125 dB を満たす。旧 watchdog で停止した最初の FORGE D7 は証拠から除く。D7 の両 `a02_b02` だけを一致確認後に import 登録し、D5b と study_063 の DRR は identity 不一致で再利用しない。

D-2：FORGE の [b05](../../runs/study_067_forge_c3_common_grid/20261022T183601578711Z_05c92e79ce80_forge/d2_measurements/a05_b05/result.json) / [b04](../../runs/study_067_forge_c3_common_grid/20261022T183601578711Z_05c92e79ce80_forge/d2_measurements/a05_b04/result.json) はともに完了（rank 30、2反復、worker 1）。b05 は wall 31526.9712 s、9 blocks、3502.4879 s/block、peak RSS 4.9397 GiB、60反復外挿29.1874 h/block。b04 は wall 10971.3912 s、16 blocks、685.2647 s/block、peak RSS 3.7954 GiB、60反復外挿5.7105 h/block。b04 は従来の5.5867 h/block 比 +2.2176%。**測定 overlap は両方 `[0,4,0,4]`** で、採用 b04 `[0,6,0,6]` / 25 blocks と b05 `[0,8,0,8]` の実測ではない。採用 b04 の代理推定は142.7655 CPU-h / 17.1336 h wall（12 workers）。依頼時の前提とは異なり、C3 の [b04](../../runs/study_067_forge_c3_common_grid/20261022T183612495378Z_05c92e79ce80_traces/d2_measurements/a05_b04/result.json) / [b05](../../runs/study_067_forge_c3_common_grid/20261022T183612495378Z_05c92e79ce80_traces/d2_measurements/a05_b05/result.json) も completed で存在し、60反復外挿は1.5066 / 8.2045 h/block。同じ overlap の代理測定として記録する。

[計画推定 artifact](round2_outcomes.json) は FORGE / C3 の DRR S1 残り + seed を2973.0 / 1613.7 CPU-h、workers 12・S1 同時候補4 / 2で wall 121.8 / 95.8 h と見積もる。FORGE の更新幅は +15.4896 CPU-h / +0.7448 h wall で、並列計画は維持する。最大候補 wall は29.1887 / 16.4097 hで72 h規則の開始前報告は不要。各 dataset の重い2窓は対応する実測 per-block 時間、他窓は同 dataset の b05 からの行列寸法比を使う。採用 overlap への代理適用と worker 並列性は未検証であり、競合と RSS 複製の保証ではない。CPU 配分は DRR 48 + 24、POCS 2 + 2、GPU 側予約20 = 96 cores 以下とし、起動時に BLAS 各1 thread と GPU 側の同時起動調整を必要条件にする。詳細は [decisions](decisions.md)。

探索の正本は一つの [grid.yaml](grid.yaml)。dataset 別の正本は [C3 config](c3/config.yaml) / [inputs](c3/inputs.yaml)、[FORGE config](forge/config.yaml) / [inputs](forge/inputs.yaml) と各 method YAML。固定値の例外と理由は [dataset_differences.yaml](dataset_differences.yaml)、採用理由と未決事項は [decisions.md](decisions.md) に記録する。prepare は共通 grid の SHA256 と固定値の例外を検証する。CCNet の grid は空間幅だけを上書きし、時間方向は各 dataset の full trace を保持する。

各 dataset は 5 手法 ×（S1 の 25 候補 + 評価 3 seed）= 140 run、合計 280 run を論理的な予算とする。S1 は固定 partition の validation のみを予測・採点し、test waveform は method へ渡さない。FORGE は clean_target の masked_trace_relative_mse、C3 は全対象の同指標を最小化し、同値なら configuration ID の辞書順で選ぶ。S1 を終えた手法は、選択した設定で新しい observed / validation / test assignment の seed 学習を barrier 前に開始してよい（Q1=A の承認事項、実装は Sprint 2）。test の採点は全手法の S1 完了 barrier 後に限る。model seed は固定し、partition seed を変える。validation と test の seed 別指標、および seed を等重みとした平均 ± 標本 SD を保存する。FORGE は clean_target と all_eligible の両方を報告する。

床の基準は各手法の S1 validation mean trace SNR の best-of-25 が C3 で 10 dB 超、FORGE で 0.4 dB 超となること。relMSE で選んだ cell の3-seed test も報告する。ReGSI 最良は研究目標であり、barrier やベースライン採否の条件にはしない。比較結果は未確定。

FORGE S1 は study_066 の reference input と observed / validation / test cell ID を照合する。評価 seed でも outer namespace は `study_052_forge_m1_mvp`、validation namespace は `study_060_forge_fair_grid` を維持し、それぞれの既存 SHA256 generator に評価 seed を渡す。natural-missing cell は geometry、観測、ラベル、採点から除外する。4 ms anti-alias decimation、trace_oracle の正規化・復元境界、および 500 m/s 整列は reference の sealed physical waveform / RMS / alignment を検証して再利用する。RMS scalar 以外の test amplitude は S1 で読み出さない。

再利用元は config の `import_from` に列挙した study_063 / study_064 の C3 run（DRR は対象外）、C3 POCS forge-matched 16 / 32 反復 run、study_066 の FORGE hand-made grid / single-config run、および study_067 の P1b rx4x8 / P2 ReGSI 全 variant / P3 C3 NeRSI / N-1 FORGE NeRSI 採用 arm / R-1 C3 / P-1 K1 / D-1 D7 pilot。R-1 は現行 V3 と一致し、K1 は採用した固定値と一致する完了 run を再利用する。C-1 の完了4実験のうち、採用構造に一致する両 tk3 だけを再利用し、両 tk9 は identity で拒否する。FORGE CCNet の `20261021T231630621271Z_05c92e79ce80_ccnet_bf16_spatial_inference_restart` は checkpoint と prediction がなく、`a01_b02` も fresh training が必要である。完了した prediction のない run は再利用に数えない。

CPU `prepare --check-only` の import coverage（DRR overlap / N-1 upsampling 適用、C-1 完了後、各欄は **imported / to-run**）。実行コマンドと元出力は [coverage artifact](round2_outcomes.json) に保存した。

| 手法 | C3 S1 imported / to-run | FORGE S1 imported / to-run | C3 評価 seed 最大 imported / to-run | FORGE 評価 seed 最大 imported / to-run |
|---|---:|---:|---:|---:|
| ReGSI | 2 / 23 | 1 / 24 | 0 / 3 | 0 / 3 |
| CCNet | 1 / 24 | 1 / 24 | 0 / 3 | 0 / 3 |
| NeRSI | 1 / 24 | 1 / 24 | 0 / 3 | 0 / 3 |
| POCS | 25 / 0 | 25 / 0 | 0 / 3 | 0 / 3 |
| DRR | 1 / 24 | 1 / 24 | 0 / 3 | 0 / 3 |
| 合計 | 30 / 95 | 29 / 96 | 0 / 15 | 0 / 15 |

POCS は両 dataset とも全候補が K1 に一致する。DRR は D7 の採用 overlap と一致する cell だけを import し、study_063 の旧 overlap からは import しない。CCNet は C-1 tk3 の C3 `a05_b02` / FORGE `a03_b02` を各1件 import する。両 tk9 は入力 fingerprint が一致しても model identity で拒否し、`import_from` は変更していない。NeRSI は C3 の P3 と FORGE の N-1 採用 arm（ともに `a02_b04`）を各1件 import する。FORGE P3 は `import_from` に残したまま、upsampling の有効性が異なるため identity で拒否する。評価 seed は S1 winner に依存する。件数と source path は上記 coverage artifact を参照。

import は宣言した最小限の execution-only key を除く resolved config、保存済みの全 partition ID、観測波形、geometry、時間軸、実際に使用する RMS scalar の一致を要求する。BF16、training batch、patch/core shape、有効な jitter は除外しない。NeRSI の jitter null は absent と同一視し、canvas から再導出する `model.profile_shape[0]` は identity から除く。FORGE の実行時 profile 時間長は 2176（canvas 2172 を time-only 段込みの64倍数へ切上げ）。profile 時間長の正規化とは別に、有効な `canvas_time_upsampling` は identity に残るため、upsampling のない P3 は import できない。prediction の hash を検証し、現在の scoring code で再採点する。manifest には source path、config / input hash、`source/` snapshot hash を記録する。古い NeRSI は jitter が一致しないため再利用しない。seed の再利用は S1 winner と一致する場合に限る。

FORGE CCNet は C3 と同じく、毎更新 fresh patch と現在重みの forward / backward を1回ずつ使う（`reuse_patch: false`、`cache_patch_gradients: false`）。精度の差は FORGE bf16 / C3 FP32（TF32 有効）。inner mask は両 dataset とも 0.25。placement / mask / supervision は `SeedSequence(7301).spawn(3)` の独立系列。manifest に patch 数、教師 trace 数、patch ごとの実効 inner 比率、would-clip 率を記録する。`execution.cudnn_benchmark` は実行専用で、C-0 の1.34倍を根拠に両 dataset で on を採用した。精度設定は保持する。on は bitwise 再現性を保証しない。FORGE CCNet S1 の見積りが 450 GPU-h を超える場合は開始前に報告する。

`prepare --check-only` は CPU で一時ディレクトリへ prepare し、成功した場合に import coverage を表示して削除する。`runs/` は変更しない。GPU 要求量は明示 margin を含め、許可 device の total memory から config の `context_allowance_gib` を引いた最大容量と prepare / admission の両入口で照合する。FORGE CCNet の予算は fresh training / prediction の C-0 実測により 17 / 17 / 17 / 30 / 64 GiB を維持する。coverage の評価 seed 件数は候補別に示し、S1 winner 未確定時の最大件数は保証ではない。

## 本番 launch（operator）

2026-10-24 にユーザーが本番開始と2 shard・重い候補優先の scheduling を承認した。
以下は最終差分を operator が commit / tag した**clean checkout の launch commit**で行う。
`study-067-launch-v4`（`11ebcc7`）は rehearsal の production code の基準であり、今回の
scheduling 設定を含む最終 launch commit は operator が確定する。学習・予測の production
code は v4 と同一で、今回の `src/` 変更は rehearsal checker だけである。

`excluded_devices_while_pids_alive` の legacy PID **3958956 と381291 は、2026-10-24 の
read-only `/proc` 確認で両方とも消滅済み**。登録した start ticks と一致する生存 process は
なく、この2 entry は cuda:0 を除外しない。entry は残し、PID reuse は start ticks で区別する。
両 GPU を見せるため、CPU 検証時の空の `CUDA_VISIBLE_DEVICES` を引き継がない。

```bash
cd /workspace
set -euo pipefail
test -z "$(git status --porcelain)"
export LAUNCH_COMMIT=$(git rev-parse HEAD)
export PY=/workspace/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES
C3=$(PYTHONPATH=/workspace/src "$PY" scripts/run_c3_region_experiment.py \
  prepare --repo /workspace --study studies/study_067_forge_c3_common_grid/c3)
FG=$(PYTHONPATH=/workspace/src "$PY" scripts/run_forge_grid_experiment.py \
  prepare --repo /workspace --study studies/study_067_forge_c3_common_grid/forge)
export C3 FG
printf 'launch=%s\nC3=%s\nFORGE=%s\n' "$LAUNCH_COMMIT" "$C3" "$FG"
```

FORGE CCNet は2 shard。変更した priority は全25 IDを一度ずつ列挙する。
下表の行内は b 昇順、列内は a 昇順。C3 NeRSI・POCS と両 DRR priority は変更しない。

| dataset / method | priority |
|---|---|
| FORGE CCNet | `a01_b02`（G0 gate）、a05 行、a04、a03、a02、a01 の残り |
| C3 CCNet | a05 行、a04、a03、a02、a01 |
| C3 ReGSI | b05（k6）列、b04、b03、b02、b01 |
| FORGE ReGSI | `a01_b05`、b05 列の残り、b04、b03、b02、b01 |
| FORGE NeRSI | a03（λ 0.01）行、a04（λ 0.03）、a01、a02、a05 |

`candidate_priority` を適用してから `i/n` の位置剰余で分割する。FORGE CCNet の先頭は、
`0/2`: `a01_b02, a05_b02, a05_b04, a04_b01, ...`、
`1/2`: `a05_b01, a05_b03, a05_b05, a04_b02, ...`。
各 shard 内で、import / 完了候補を先に処理し、残りの優先順から現在メモリに収まる最初の候補を
共通 admission lock 内で予約する。a05 が収まらない間に軽い候補へ進むことは承認済みの例外。
shard 間には全候補の厳密な開始順を強制する barrier はなく、G0 は shard 0 内の先頭である。
両 shard に a05 が先頭群として配分されるが、scheduler の process 起動順は決定的ではない。

**大きな shard を先に確保する段階は本番にも必要**。supervisor は method 順に worker を
起動し、method 間の重さ順は保証しない。ReGSI などが先に相乗りすると、両 a05 の同時 admission
を妨げる。次は FORGE CCNet だけを先に開始し、gate を経た後の両 a05 が別 GPU を確保するまで
待つ。gate は本番50,000更新なので rehearsal の5分期限を流用しない。空き不足なら log と
admission state を確認し、別 process に signal を送らない。

```bash
mkdir -p "$FG/workers"
W() {
  local method=$1 shard=$2 worker=$3
  local niceness=()
  if [[ "$method" == pocs_grid || "$method" == drr_grid ]]; then niceness=(nice -n 10); fi
  setsid nohup env PYTHONPATH="$FG/source/src" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${niceness[@]}" "$PY" "$FG/source/scripts/run_forge_grid_experiment.py" worker \
    --repo /workspace --experiment "$FG" --method "$method" --shard "$shard" \
    >> "$FG/workers/$worker.log" 2>&1 </dev/null &
}
W ccnet5d_grid 0/2 ccnet5d_grid
W ccnet5d_grid 1/2 ccnet5d_grid_shard_1_of_2
PYTHONPATH="$FG/source/src" "$PY" - <<'PY'
import json, os, time
from pathlib import Path
from seis_interp.training.worker_processes import process_is_alive
root = Path(os.environ['FG']) / 'workers'
while True:
    states = []
    for name in ('ccnet5d_grid', 'ccnet5d_grid_shard_1_of_2'):
        try:
            state = json.loads((root / f'{name}.json').read_text())
        except (OSError, ValueError):
            break
        if state.get('status') in ('failed', 'retry_pending', 'completed'):
            raise SystemExit(f'{name}: inspect startup state/log before proceeding')
        if not process_is_alive(state.get('pid', -1), state.get('process_start_time')):
            raise SystemExit(f'{name}: worker is no longer alive')
        states.append(state)
    if len(states) == 2 and all(
        state.get('phase') == 'S1'
        and state.get('last_admission', {}).get('run', '').startswith('a05_')
        for state in states
    ) and {state['last_admission']['device'] for state in states} == {'cuda:0', 'cuda:1'}:
        break
    time.sleep(5)
PY
for method in regsi_real nersi_profile pocs_grid drr_grid; do W "$method" 0/1 "$method"; done
PYTHONPATH="$FG/source/src" "$PY" - <<'PY'
import json, os, time
from pathlib import Path
from seis_interp.training.worker_processes import process_is_alive
root = Path(os.environ['FG']) / 'workers'
deadline = time.monotonic() + 60
while True:
    ready = True
    for method in ('regsi_real', 'nersi_profile', 'pocs_grid', 'drr_grid'):
        try:
            state = json.loads((root / f'{method}.json').read_text())
        except (OSError, ValueError):
            ready = False
            continue
        if state.get('status') in ('failed', 'retry_pending'):
            raise SystemExit(f'{method}: inspect startup state/log before proceeding')
        ready &= state.get('status') == 'completed' or process_is_alive(
            state.get('pid', -1), state.get('process_start_time'))
    if ready:
        break
    if time.monotonic() >= deadline:
        raise SystemExit('FORGE worker startup precondition was not met')
    time.sleep(1)
PY
# 全 FORGE worker を先に開始したため、初回の missing-worker retry を消費せず adoption する。
"$PY" scripts/launch_grid_experiment.py forge --repo /workspace --experiment "$FG" --only-dead-workers
"$PY" scripts/launch_grid_experiment.py c3 --repo /workspace --experiment "$C3"
```

`launch_grid_experiment.py` は experiment 内の frozen CLI と frozen `source/src` を使って、
`nohup`・新 session・stdin `/dev/null`・append `supervisor.out` で supervisor を起動する。
両 supervisor が worker に明示する環境は `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
`MKL_NUM_THREADS=1`、`PYTHONUNBUFFERED=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、
`PYTHONPATH=<experiment>/source/src` の6つで、rehearsal の W と同じ。追加の worker 専用環境は
不要。`CUDA_VISIBLE_DEVICES` は親から継承し、supervisor は解除しない。POCS / DRR は nice 10。

進捗・待機理由・retry は frozen status と worker log、全体完了は supervisor state で確認する。
復旧は supervisor が不在になったことを確認後に `--only-dead-workers` で行い、live worker を
PID / start ticks で採用し、完了 worker と retry budget を保持する。live supervisor との同時起動は
lock で拒否する。通常の `run` は live worker があると拒否する。

```bash
PYTHONPATH="$C3/source/src" "$PY" "$C3/source/scripts/run_c3_region_experiment.py" status --experiment "$C3"
PYTHONPATH="$FG/source/src" "$PY" "$FG/source/scripts/run_forge_grid_experiment.py" status --experiment "$FG"
tail -n 50 "$C3/supervisor.out" "$FG/supervisor.out" "$FG/workers/ccnet5d_grid.log"
# supervisor 不在時の復旧（必要な dataset だけ実行）
"$PY" scripts/launch_grid_experiment.py c3 --repo /workspace --experiment "$C3" --only-dead-workers
"$PY" scripts/launch_grid_experiment.py forge --repo /workspace --experiment "$FG" --only-dead-workers
```

結果は experiment 直下の `summary.csv` / `seed_metrics.csv`、`workers/<method>_summary.csv`、各 run の `manifest.json` / prediction / per-trace parquet に保存する。FORGE は直下にも `<method>_summary.csv` を保存する。全 run manifest は runtime、run ごとにリセットした RSS sampler の peak（sampling interval も記録）、run ごとにリセットした CUDA allocated / reserved peak、allocator 設定と admission cap を記録する。import run の計測値は再採点処理の資源量を表す。


memory probe の入口は [probe_forge_ccnet_memory.py](../../scripts/probe_forge_ccnet_memory.py)。`--plan-only` は CPU で引数と grid を確認し、run を作らない。GPU 実行は legacy worker が同じ GPU で動作していないことを確認してから行う。

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/probe_forge_ccnet_memory.py --device cuda:0 --minimum-free-memory-gib 17 --reserve-gib 8
```

probe は現行 resolved config / S1 観測入力を使う。`--mode memory` は各 patch の1更新と最大 halo の full-time prediction forward を測る。`--mode timing` は10回の untimed warm-up 更新に続けて `--updates N` 回を測り、同期した更新ごとの時間と中央値を報告する。setup / autotune は比較に含めない。`--cudnn-benchmark both` は off,on,on,off の順に独立 trial を実行し、両順序の中央値・速度比と、全 timed update の中央値による速度比を保存する。`--mode end-to-end` は予測・整列解除後に validation だけを主 EMA / shadow EMA で採点し manifest を保存する。memory / timing は validation を採点せず、全 mode で test を採点しない。

patch 間で cache を解放して空きを再計測し、`max(0, free − reserve_gib × 2**30) / total` を allocator fraction に設定する。reserve 後の空きが閾値未満なら `skipped / insufficient_free_memory`、cap 下の CUDA OOM は `exceeds_cap` として次へ進む。cap 下の OOM は必要 peak の下限であり、成功した実測値として予算に使わない。C-0 の fresh 更新と予測の実測に `max(現行値, ceil(最大 reserved GiB) + 2)` を適用し、予算 17 / 17 / 17 / 30 / 64 GiB を維持した。

## C-1（完了・T1 不成立、tk3 維持）

C3 `a05_b02`（FP32）と FORGE `a03_b02`（bf16）の tk3 / tk9 計4実験は完了。**T1：FORGE ΔSNR ≥ +0.2 dB かつ C3 ΔSNR ≥ −0.2 dB** を登録どおり適用した。Δ は tk9 − tk3、S1 validation mean trace SNR、FORGE は `clean_target`、primary EMA は0.999。

| dataset / cell | tk3 SNR (dB) | tk9 SNR (dB) | ΔSNR (dB) | tk3 relMSE | tk9 relMSE | tk3 / tk9 runtime (h) |
|---|---:|---:|---:|---:|---:|---:|
| C3 `a05_b02` | 18.8033 | 18.7228 | -0.0805 | 0.0167 | 0.0170 | 10.3941 / 16.5190 |
| FORGE `a03_b02` clean_target | 1.5923 | 1.6616 | +0.0693 | 0.7376 | 0.7277 | 13.7093 / 15.9392 |

FORGE が +0.2 dB に届かず **T1 不成立、tk3 を維持**する。両 production CCNet YAML は time kernel 3 / hidden 53 / intermediate 63 のまま byte 不変。manifest の値はユーザー提示の6桁丸めと一致する。各 arm 単独の identity check で両 tk3 の S1 import と両 tk9 の拒否を確認した。全 manifest の path / SHA256、full-precision 値、差分、runtime、診断専用 shadow EMA 0.999 / 0.9995 / 0.9998 は [c1 artifact](round2_outcomes.json)、shadow の表は [decisions](decisions.md) に保存した。準備先と identity hash は [C-1 記録](pilots/c1_preparation.md)。

## 第2弾 pilot の切替

NeRSI の `canvas_time_upsampling: true` は、`ceil(T_profile / 2^(3+s)) <= 48` となる最小 s の time-only ×2 段（channels 28、kernel 3）を加える。FORGE は s=3、latent 34×4、341,485 parameters。C3 は s=0 で計算も import identity も変わらない。`training.loss_support_mask: true` は整列前の記録範囲だけを loss に使い、FORGE は各 trace 1,001 samples、C3 は全 samples のまま。`training.ema_decay` は N-2 の 0.9998 にも切替可能。N-1 は upsampling 採用・support mask 不採用・jitter 保持、N-2 は EMA 変更を不採用と判断済み（[結果 artifact](round2_outcomes.json)）。ユーザー承認済みの N-1 upsampling を両 S1 YAML に適用し、jitter 0.5 と全 canvas loss（`loss_support_mask` は absent）を保持する。FORGE N-1 全4 arm の最大 CUDA reserved は 1,046,478,848 bytes（0.9746 GiB）、最大 allocated は 914,378,752 bytes（0.8516 GiB）で、宣言済み NeRSI 4 GiB 以内。

N-1 の4 arm は `forge/nersi_n1_{jitter,no_jitter}_{canvas,support}.yaml`、P-1 の K1 は両 dataset の `pocs_k1.yaml` を使う。これらを新しい pilot study の該当 method YAML として凍結して prepare し、`pilot --candidate a02_b04`（N-1）または `pilot --candidate all`（P-1）を本番 runner で実行する。既存 experiment の設定を書き換えない。pilot study は共通 grid を local `grid.yaml` へ複製し、`common_grid` binding を外した独立した設定として prepare する（S1 正本の binding は保持する）。各 template は pilot study 内で元の method 名（`nersi_profile.yaml` / `pocs_grid.yaml`）へ配置する。pilot 入口は S1 validation のみを採点し、seed の評価へ進まない。

DRR は実効 overlap 0 のとき `drr_nonoverlapping` と `unknown_padding` を使い、軸長で割り切れない窓も旧計算を維持する。overlap > 0 のときだけ `drr_batched` と `clip_and_shift`（window を軸長へ切り詰め、最後を内側へずらし、正の interior-Hann 重みで合成）を使う。backend と境界規則は resolved config に記録して import identity に含める。D-1 採用後の共通 grid は point 軸50% / line 軸0 の overlap を各 window に明示する（[grid artifact](grid.yaml)）。`point_half` の導出にも対応するが、import 元 D7 と同じ `explicit` を使う。周波数は `execution.batch_matrix_limit` で分割し、worker 数とこの分割上限は import identity から除く。両 key の変更による正の overlap の bitwise 出力不変は `test_overlapping_grid_execution_settings_preserve_bits_and_import_identity` で検証する。study_063 の overlap 0 は設定 identity で拒否する。実データの受入条件は [CPU check plan](../../reports/study_067_drr_cpu_check_plan.md)、D-2 の実測・代理推定と実行条件は上記の結果および [decisions](decisions.md) を参照。

新規ニューラル run は診断専用の shadow EMA 0.999 / 0.9995 / 0.9998 を持ち、主予測は EMA 0.999。shadow key は import identity から除く。validation / test の指標だけを manifest に保存する。shadow の状態は厳密再開 checkpoint に含め、予測配列は barrier 待機中だけ一時保存する。過去の import 元に shadow がなければ診断表では欠測とし、そのための再学習はしない。

## 長時間 run の運用

両 runner のニューラル run は、2,500 更新または30分ごとに `resume.pt` を原子的に保存する。最終更新では同じファイルを `before_prediction` 状態にし、予測失敗後は optimizer 更新を追加せず再開する。OOM・基盤障害は最大2回再試行し、失敗試行は上限後の試行も含め `attempts/<n>/` に残す。loss・予測・数値指標の非有限値は `diverged` として S1 の最下位に数え、再試行せず他の候補へ進む。全候補が発散した場合も同順位として ID 順を適用する。再開時の AdamW step と EMA snapshot は CPU に保持する。学習前に最大 halo のゼロ振幅予測を行い、run 中は CUDA cache を解放しない。CCNet の CPU thread 先読みは RNG の消費順を保つ。

`worker --shard i/n` は0始まりで候補を分割する。同一手法の全 shard が終わってから shard 0 が25候補を選択する。`run` の supervisor は `method_execution.<method>.worker_shards`（既定1）個の worker を起動する。完了・発散済み・import 対象の S1 cell は GPU メモリ待ちを通さない。残る候補は事前の優先度順のうち、現在の空きメモリに収まる最初のものを選び、admission と共通の lock 内で予約する。選択後の seed 学習・予測は最大2 process で先行でき、test target の読み出しと採点は全手法の S1 barrier 後に限る。barrier 通過は永続化し、その後の worker 再起動で取り消さない。3 seed が完了した場合だけ平均・標本 SD を計算し、欠測・発散 seed と実際の seed_count を明示する。

admission は空きディスク5 GiB以上を確認し、不足時と GPU 待機は試行回数を消費せず待つ（`cuda_execution.minimum_free_disk_gib` で指定可能）。基盤障害は run ごとの失敗回数を保存し、5秒・10秒の指数 backoff 後に再試行する。sticky CUDA error は fresh process へ移す。並列 run は独立した process を使い、watchdog は停滞した run の process だけを終了する。ニューラル run の閾値は30分以上の下限と、初期予定間隔・直近256報告間隔（checkpoint・予測を含む）の95%点の3倍の大きい方とする。POCS / DRR は進捗 watchdog を使わず、壁時計上限も既定では設けない。supervisor 再起動時に pid と開始時刻が一致する live worker または取得済み worker lock があれば、重複起動を拒否し state・再試行枠を変更しない。manifest に nvidia-smi の GPU 利用率・使用量・空き量を残し、CPU worker は nice 10 にする。他 process への signal は使わない。勾配診断は device 上で集計し、更新ごとの追加同期を行わない。

## 手法別 protocol

全ニューラル手法は AdamW、lr 0.001 一定、wd 0、50,000 更新、loss は masked_trace_mse、loss trace 数64、model seed 7301、EMA 0.999（shadow 0.999 / 0.9995 / 0.9998）。

| 手法 | inner mask（名目 / 実効） | 再利用 / 勾配 | clip | 精度 | parameters（C3 / FORGE） |
|---|---|---|---|---|---|
| ReGSI | 0.25 / C3 5888÷23593、FORGE 3264÷13271 | 同一 mask で92 / 51更新、毎回 fresh gradient | norm 1.0、発動率を manifest に記録 | FP32、TF32 有効 | 346885 / 346885 |
| CCNet | 0.25 / patch ごとに round(観測数×0.25)÷観測数を manifest に記録 | 毎更新 fresh patch、fresh forward / backward | 適用せず would-clip 率を記録 | C3 FP32（TF32 有効）/ FORGE bf16 | 363292 / 363292 |
| NeRSI | 対象外 | 観測 trace に fitting、fresh gradient | 適用せず would-clip 率を記録 | FP32、TF32 有効 | 363269 / 341485（N-1 適用済み、[証拠](round2_outcomes.json)） |

## 公平性の宣言（承認設計の原文）

以下を decisions.md、README、論文の補遺に載せます。

1. **手法ごとの protocol 表**
   - 項目：optimizer、学習率、wd、更新数、loss trace 数、inner mask（名目値と実効値）、データの再利用、clip（発動率つき）、EMA（shadow を含む）、精度（TF32 を含む）、勾配の鮮度、dataset ごとの params。
   - データの再利用は次のとおりです。ReGSI は同じ mask で C3 は 92 更新、FORGE は 51 更新を続けます（毎回新しい勾配）。CCNet は毎更新 patch を引き直し、隠した trace のうち 64 本だけを教師にします。NeRSI は観測した全 trace を当てはめます。
   - 共通でも宣言済みでもない項目があれば、`prepare --check-only` が失敗するようにします。
2. inner mask は入力を条件とする手法（ReGSI、CCNet）だけに適用します。support mask を採用した場合、それは NeRSI だけに適用します。座標ネットは、offset で位置が動く padding の段差を自分で合成しなければならないためです。
3. FORGE の CCNet の bf16 はメモリ上必要です。ReGSI が FP32 のままでいられるのは、厳密な省メモリ機能（checkpointing、pruning、fused aggregation）があるためです。この非対称を宣言し、大きさは S-b で測ります。
4. clip は ReGSI だけに適用し、発動率を示します。他の手法では would-clip 率（clip していたら発動した率）を記録します。
5. **パラメータ数の規則は1つ**にします。C3 ではニューラル3手法とも 363,269 ±5% に入ります。FORGE では C3 と同じ構造を使い、宣言した canvas 適応だけを例外にします（NeRSI の s=3 で 341,485、−5.9967%）。FORGE NeRSI は C3 の ±5% band（345,106–381,432 parameters）より小さい。FORGE の宣言済み canvas 適応例外として報告し、幅と tolerance は変えない。
6. grid の値は sample 単位です。FORGE は 4 ms、C3 は 8 ms なので、dilation 6 は FORGE で 24 ms、C3 で 48 ms に当たります。
7. FORGE の loss は 2,172 sample の canvas 全体にかかります。有効な sample は 46% です（NeRSI は N-1 の結果に従います）。
8. **格子と実座標の差**
   - grid 系の手法は cell 中心の座標で、ReGSI と NeRSI は実座標で予測します。
   - binned offset から計算した遅延は、実 offset から計算した遅延と中央値 13 sample（p90 32）ずれます。ただし隣り合う trace 同士のずれは中央値 1 sample です。
   - 隣接 trace のゼロ lag 相関は、格子と実座標でほぼ同じです（震源点方向 0.285 対 0.285、受振点方向 0.135 対 0.151）。
   - study_060 では regsi_grid と regsi_real の差は約 0.07–0.1 dB でした。
9. FORGE の SNR は全手法で、trace ごとの oracle test RMS を使います。oracle を使わない IDW 版と scale 不変の指標は副次として報告します。
10. model seed は 7301 に固定します。study_060 の FORGE では model seed による差は約 0.03 dB でした（ReGSI、CCNet）。したがって SD は partition とデータ順の変動を表します。
11. **調整量の表と、過去の勝ち設定の扱いの表**
    - 調整量の表には過去の target-informed な調整も含めます（C3：ReGSI は study_044 の28 run、CCNet は study_043 の6 run、NeRSI は study_040/046 の76+28 run。C3 の窓自体も study_041 で target を見て選ばれています）。
    - studies 057–066 で各手法が勝った設定を一覧にし、「第2弾で試した／ユーザー規則で不可／証拠により不採用」のどれかを付けます。
12. **pilot の記録**：規則を登録した時点で既に見ていた結果を開示します。対象は POCS の P1b grid、K1 の screening、C3 DRR の S1（study_063）、D7 の C3 と D5b、ReGSI の P2 a01_b01、NeRSI の P3 です。
13. **負の結果の段落**：(2) の不採用項目を載せます。
14. **床の基準**（2026-10-22）：S1 validation で best-of-25 とします。選ばれた cell の3 seed test も併せて報告します。
15. **選択規則**
    - validation の relMSE が最小の候補を選び、同値なら candidate ID 順にします。
    - diverged の run は最下位として数に含めます。
    - OOM の後は同じ yaml を再開します。
    - early stopping はしません。途中の結果を見て run を止めたり、優先順位を変えたりしません（メモリによる並べ替えだけは可）。
16. **S1 barrier**：確認するのは完了と床だけで、ReGSI が最良かどうかは確認しません。S1 を見た後は固定値も grid も変えません。seed は自動で進めます。ユーザーが止めて変更した場合は「post-S1」と明記します。
17. 最初の test 採点の後は、何も変えません。未達の目標は未達として報告します。
18. grid.yaml、各 config、各手法の yaml、評価 script と config の SHA256、さらに code の commit / tag を記録します。dirty worktree のままでは開始しません。

