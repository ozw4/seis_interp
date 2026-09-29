# study_forge_nersi_temp

FORGE NeRSIのパラメータを自由に変えて試すためのscratchワークスペース。
番号付きstudyではなく、正式な証拠には使わない。
結果を残したい条件は、番号付きstudyへ移してから正式な手順で実行する。

条件の正本は [config.yaml](config.yaml)、入力の準備runは [inputs.yaml](inputs.yaml)。
[config.default.yaml](config.default.yaml) はstudy_059で採用したNeRSIの既定値で、既定の準備runに対して
同じモデル（パラメータ数、座標変換）を再現する。テストがこのファイルを固定しているので編集しない。
`config.yaml` は最初は既定値と同じで、自由に編集してよい。既定値へ戻すときは `config.default.yaml` を写す。

## 使い方

まず `--check` で設定を検証する。学習もtestデータの読み込みもせず、ファイルも作らない。

```bash
.venv/bin/python scripts/run_study_forge_nersi_temp.py --check
.venv/bin/python scripts/run_study_forge_nersi_temp.py            # 学習・評価・図まで実行
.venv/bin/python scripts/run_study_forge_nersi_temp.py --summarize  # summary.csvを作り直して表示
```

最初に既定のconfigのまま1回実行し、比較の基準となるbaselineの行を作る。

条件を変えるときは [config.yaml](config.yaml) を直接編集するか、任意の場所にコピーして `--config` で渡す
（例: `studies/study_forge_nersi_temp/<name>.yaml`）。条件ごとに `name` を変える。
`name` はrunディレクトリ名に入り、同じ `name` のrunは一覧で時刻しか違わなくなる。
各runのディレクトリには、読み込んだconfigファイルそのもの（`config.requested.yaml`）と解決済みconfig
（`config.resolved.yaml`）が残るので、コピーしたファイルを保存しておく必要はない。
以前の条件をもう一度流すときは、そのrunの `config.requested.yaml` を `--config` で渡す。
ただし複数ヘッドなどのキーを追加する前に作ったrunの `config.requested.yaml` は、そのままでは
必須キーの欠落でエラーになる。次の6キーを無効の値で書き足してから渡す:
`model.head_slowness_ms_per_m: null`、`model.static_shift_samples: 0.0`、
`model.residual_slowness_ms_per_m: 0.0`、`model.output_band_hz: null`、`model.output_window_mask: false`、
`training.weight_decay_after: null`。
`config.resolved.yaml` は固定値と導出値を含むので `--config` には渡せない。
`--device` はconfigの `device` を上書きする。上書き後の値は `run.json` の `requested_config` に記録され、
`config.requested.yaml` は元のファイルのまま残る。GPUごとに別の条件を並列に流せる。

```bash
.venv/bin/python scripts/run_study_forge_nersi_temp.py --config a.yaml --device cuda:0 &
.venv/bin/python scripts/run_study_forge_nersi_temp.py --config b.yaml --device cuda:1 &
```

configは全キー必須である。キーの欠落、未知のキー、同じキーの重複、コードや準備runで固定される値は、
データを読む前にキー名つきでまとめてエラーになる。未知のキーには近い候補名を示す。
YAMLの構文エラー（インデントのずれ、タブ）はファイル名つきの `ERROR:` 行になる。
準備runに依存する検査（batch_sizeと観測トレース数、デバイス、canvas長でのモデル構築）も、
runディレクトリを作る前に行う。不正なconfigではrunディレクトリは作られない。

PyYAMLは指数表記を、小数点と符号つきの指数が両方あるときだけ数として読む。
`1e-3`・`1e+3`・`1.0e3` はどれも文字列になる。`0.001` または `1.0e-3` と書く。
整数のキーには `20000` のように10進の数字だけを書く。`07301` は8進数、`20:00` は60進数として読まれるので
エラーにする。

## パラメータ

| キー | 意味 |
|---|---|
| `name` | runの名前。runディレクトリ名の末尾になる。条件ごとに変える。小文字snake_case。`latest`・`temp` などは使えない |
| `device` | `cpu` または `cuda:<番号>`。番号なしの `cuda` は不可 |
| `model_seed` | 重みの初期化と、ミニバッチで選ぶ観測トレースの順序を決める |
| `torch_threads` | このrunのCPUスレッド数。終了後に元へ戻す |
| `model.coordinate_frame` | 座標系。`midpoint_offset` または `source_receiver`（下表） |
| `model.polar_offset` | `true` でオフセット2軸を半径と方位角/πにする。`midpoint_offset` が必要 |
| `model.moveout_velocity_m_per_s` | `null` で無効。正の値Vのとき、準備runの整列に加えてデコーダ出力へ線形moveoutをかける（遅延が足し合わさる。下記）。`polar_offset: true` が必要。パラメータが増える |
| `model.fourier_components` | 軸あたりのFourier成分数F。角周波数はπ·base¹…π·base^F、特徴量は4·2·F |
| `model.frequency_base` | 1より大きいスカラー、または座標軸の順に4つの値 |
| `model.encoder_width` | エンコーダ幅W |
| `model.latent_channels` | 潜在チャネル数L |
| `model.decoder_channels` | 3段のデコーダのチャネル数 |
| `model.kernel_size` | デコーダ畳み込みの奇数カーネル長 |
| `model.head_slowness_ms_per_m` | `null` で従来の1ヘッド。正のslowness（ms/m、元のgatherでの見かけのslowness）のリストで、1つにつき1ヘッドのデコーダ出力を持つ（下記）。`midpoint_offset` と `polar_offset: true` が必要。`moveout_velocity_m_per_s` とは同時に使えない。scratchプロトタイプの1ヘッドの対照（`slowness: []` は2.0 ms/m、つまり整列速度500 m/sの1ヘッド）は、study_059の準備runで `[2.0]` と書くと再現する。`null` は従来のNersiTraceの構成で、paddingがなく（デコーダ長 = canvas）、パラメータ数も少し小さいので、同じ対照にはならない |
| `model.static_shift_samples` | `0` で無効。正の値Sのとき、sourceとreceiverごとの時間シフトを学習する。それぞれ ±S サンプル（tanhで制限、初期値0）。ヘッドが必要 |
| `model.residual_slowness_ms_per_m` | `0` で無効。正の値のとき、座標に依存するslownessの補正をヘッドごとに ±この値（ms/m）の範囲で学習する（初期値0）。ヘッドが必要 |
| `model.output_band_hz` | `null` で無効。`[lo, hi]`（Hz）で出力に固定のゼロ位相帯域テーパをかける。2 < lo < hi、hi+5はNyquist未満 |
| `model.output_window_mask` | `true` で、整列canvas上の各トレースの記録窓の外を0にする。`polar_offset: true` と整列済みの準備runが必要 |
| `training.learning_rate` | 一定の学習率 |
| `training.max_steps` | optimizer更新回数 |
| `training.batch_size` | 1更新で使う観測トレース数。重複なしで抽出するので観測トレース数以下 |
| `training.weight_decay` | AdamのL2 weight decay（結合型、AdamWではない） |
| `training.weight_decay_after` | `null` で一定。`{step: N, value: X}` で、N回目の更新（その更新の前）から全パラメータのweight decayをXにする。1 ≤ N < `max_steps`、X ≥ 0 |
| `training.ema_decay` | 最終重みのEMA減衰率。`null` でEMAなし |
| `training.report_interval` | ミニバッチlossを記録する更新間隔 |
| `prediction.batch_size` | 推論時のトレース数。学習結果には影響しない |

座標軸の順（`frequency_base` の4値もこの順に対応する）:

| `coordinate_frame` | `polar_offset` | 4軸 | 正規化 |
|---|---|---|---|
| `source_receiver` | `false` | source_line, source_point, receiver_line, receiver_point | 局所格子座標 / (shape−1) |
| `midpoint_offset` | `false` | cmp_x, cmp_y, half_offset_x, half_offset_y | 準備runの実座標で各軸min-max |
| `midpoint_offset` | `true` | cmp_x, cmp_y, radius, azimuth/π | CMPはmin-max、半径は最大半オフセットで割る |

`midpoint_offset` の変換行列は、準備runのgeometryから実行時に計算する（波形は使わない）。

パラメータ数はおおむね W·L·ceil(canvas/8) で決まる。canvasは準備runの整列後の時間長で、
Tではない。moveoutを有効にするとcanvasにpadding（ceil(samples)+1サンプル）が加わり、パラメータ数も増える。
`--check` はcanvas長で組んだモデルのパラメータ数、モジュール別内訳、準備runが宣言する予算
（更新回数・学習率・batch・パラメータ数の範囲）との一致を表示する。予算は比較表示のみで、強制はしない。

EMAは最初の更新時の重みから始まるため、更新回数に対して減衰率が大きいと初期の重みが最終重みに残る。
`--check` は最初の更新の重みが最終平均に残る割合 decay^(max_steps−1) を表示し、0.05を超えると警告する。
短いrunでは `ema_decay <= 1 - 3/max_steps` か `null` にする。

moveoutは準備runの整列（共通のcanvas）の上に追加でかかる。速度は準備runのgeometryから
「単位半径あたりのサンプル数」2·R_max/(V·dt) に換算する。R_maxは最大半オフセット。
Vは物理的な事象の速度ではなく、整列に上乗せする残差の速度である。準備runは速度v_align（既定の準備runでは
500 m/s）で整列済みで、moveoutは遠いオフセットほど遅らせる方向にしかかからないので、遅延（slowness）が足し合わさる。
デコーダ出力で平らな事象は、元のgatherでは 1/(1/v_align + 1/V) の速度になる。例えば V = 700 なら約292 m/s。
速度vの事象（v < v_align）を狙うときは V = 1/(1/v − 1/v_align) にする。
v_alignより速い事象はmoveoutでは平らにできない（Vを入れるとかえって傾きが大きくなる）。
整列速度そのものを変えるには別の準備runが要る。`--check` は合成後の速度も表示する。
moveoutの逆伝播はCUDA上で決定的とは限らない。

### 複数slownessヘッド・statics・残差slowness・帯域・窓

整列速度のcanvasで平らになるのは、そのslowness付近の事象だけである。`head_slowness_ms_per_m` に
K個のslowness p_k（ms/m、元のgatherでの値）を書くと、デコーダの出力がKチャネルになり、
チャネルkを自分の線形moveoutに沿って読んでから足し合わせる。各ヘッドにとってはslowness p_k の事象が
オフセット方向に平らになる。読み出し位置は `t + pad_left + (m_k + dm_k)·r + shift` で、
rは単位極座標半径（最大半オフセットR_maxで割った半オフセット）、
m_k = 1e-3·(p_align − p_k)·2·R_max/dt（p_alignは整列速度の逆数をms/mで表した値、整列なしなら0。pはms/m、dtは秒）。
p_k がp_alignより小さい（速い）ヘッドは正、大きい（遅い）ヘッドは負のシフトになる。
デコーダは左に `ceil(max(0, −min m_k) + 2S + M) + 1`、右に `ceil(max(0, max m_k) + 2S + M) + 1`
サンプル長くなり（下の例では 1487 + 2167、デコーダ長7997）、パラメータ数も増える
（ほぼ W·L·ceil(デコーダ長/8)）。`--check` はヘッド数、slownessの範囲、シフト、padとデコーダ長を表示する。

- `static_shift_samples` S: sourceとreceiverの格子（準備runのline×point）に1つずつ値を持ち、
  入力座標でbilinearに読んだ値から shift = S·(tanh(s_src) + tanh(s_rcv)) サンプルを全ヘッドに加える。
- `residual_slowness_ms_per_m`: 極座標から小さなMLPでヘッドごとの補正 dm_k = M·tanh(·) を出す。
  Mはms/mから 1e-3·値·2·R_max/dt サンプル/単位半径に換算する。
- `output_band_hz` [lo, hi]: ヘッドの和に、lo−2 Hz未満0、lo−2..loで上がり、hiまで1、hi..hi+5で下がり、
  それより上0のraised-cosine重みをrfft/irfftでかける。パラメータはない。
- `output_window_mask`: 最後に、整列canvas上の記録窓
  `[max_delay − rint(r·2·R_max/(v_align·dt)), +T)` の外を0にする。形状だけで決まる。
  半径はプロトタイプと同じくfloat32で計算するので、遅延がちょうど半サンプル付近のトレースでは
  窓の始まりが整列canvasの整数遅延と1サンプルずれ、記録されたサンプルが1つ0になることがある
  （study_059の準備runでは49,151トレース中7本）。

statics・残差slownessは0で始まるので、初期状態ではヘッドだけの出力を変えない。
帯域と窓は学習するパラメータのない固定のフィルタで、最初の更新からかかる。
すべて座標（と準備runのgeometry）だけに依存し、波形は使わない。

例（study_059の準備runで10ヘッドと上記すべてを使う条件。`weight_decay` は0から始め、2000更新目から2e-4）:

```yaml
model:
  coordinate_frame: midpoint_offset
  polar_offset: true
  moveout_velocity_m_per_s: null
  fourier_components: 16
  frequency_base: [1.05, 1.05, 1.1, 1.1]
  encoder_width: 1200
  latent_channels: 8
  decoder_channels: [32, 16, 8]
  kernel_size: 5
  head_slowness_ms_per_m: [0.3, 0.6, 0.9, 1.2, 1.5, 1.75, 2.0, 2.3, 2.7, 3.1]
  static_shift_samples: 6.0
  residual_slowness_ms_per_m: 0.2
  output_band_hz: [4, 25]
  output_window_mask: true
training:
  learning_rate: 0.001
  max_steps: 20000
  batch_size: 64
  weight_decay: 0.0
  weight_decay_after: {step: 2000, value: 2.0e-4}
  ema_decay: 0.9998
  report_interval: 100
```

この条件はデコーダ長7997、パラメータ数9,862,772になる（`--check` で確認できる）。
matched予算のパラメータ範囲からは外れるので、`--check` は警告を出す。
リポジトリ外のscratchプロトタイプ（同じ構成）では、1 seedでclean_targetの相対MSE 0.6824だった。
weight decayはtestの値を見て選んでいるので楽観的であり、このrunnerではまだ再現していない。
多ヘッドのモデルに最初から正のweight decayをかけると、出力が0のまま学習が進まないことがあった。

## 固定される条件

準備runで固定: 評価対象の幾何（real）、整列速度とcanvas、欠損マスク、振幅正規化（trace_oracle）、
トレース毎RMS、global RMS。変えるには別の準備runを [inputs.yaml](inputs.yaml) に指定する。
`inputs.yaml` は準備runのパスと `preparation_manifest.json` のsha256を持ち、実行前に照合する。
別の準備runのsha256は次で求める（不一致のエラーにも実際の値が出る）。

```bash
sha256sum runs/<study>/<preparation>/preparation_manifest.json
```

コードで固定: Adam（結合L2 weight decay）、一定学習率、fp32重み（CUDAではTF32の畳み込み）、
aligned canvas全体（zero paddingを含む）でのmasked_trace_relative_mse、勾配clippingなし、
入力4座標、2倍×3段のアップサンプル、GELU、線形出力。
これらをconfigに書くとエラーになる。`time_sample_count`・`coordinate_transform`・
`moveout_samples_per_radius`・`head_samples_per_radius`・`residual_slowness_samples_per_radius`・
`output_window`・`spatial_shape`・`sample_interval_s`・`checkpoint_rule` は上記のキーと準備runから導出し、
`config.resolved.yaml` に記録する。

## 出力

```text
runs/study_forge_nersi_temp/
├── summary.csv                       # 全runの一覧（上書きされる現在の出力）
└── <UTC>_<sha12>_<name>/             # runごと。完了後は変更しない
    ├── config.requested.yaml         # 読み込んだconfigファイルのバイト列そのもの（--configに渡せる。古いrunは上記の6キーを足す）
    ├── config.resolved.yaml          # run_methodへ渡した完全な設定（time_sample_countは契約のT）
    ├── inputs.lock.json              # 準備runとそのhash、入力binding
    ├── run.json                      # 状態、要求config（requested_config）、configのsha256、git commitとdirty、
    │                                 # 環境、ソースhash、時間、メモリ、予算比較
    ├── metrics.json, metrics_by_trace.parquet
    ├── training_log.parquet, training.log
    ├── artifacts/final.pt            # EMA有効時はEMA重み
    ├── artifacts/prediction.npy, artifacts/prediction_index.parquet
    └── figures/training_loss.png, common_source.png, common_receiver.png,
                common_source_errors.png, common_receiver_errors.png
```

`run.json` の状態は `running` → `completed`・`failed`・`interrupted` のいずれかで、
失敗時は `failed_stage` と理由を記録する。強制終了されたrunは `running` のまま残る。

学習の後、評価のために準備runの元SEG-Y（study_059では192ファイル、約5.4 GB）のhashを照合し、
testトレースを読む。この間は段階ごとの1行（saving、SEG-Yの照合と読み込み、evaluating、plotting）しか出ない。
キャッシュが効いていれば数十秒、効いていなければもっとかかる。止まったと判断して強制終了しない。

`summary.csv` はrunが終わるたびに全 `run.json` から作り直す。列はrun ID・名前・状態、
configの全パラメータ、準備runのconfigから変えたキー（`changed_keys`）とその値（`changes`、
例: `model.encoder_width: 150 -> 300`）、パラメータ数、予算との一致、
EMA初期重みの割合、完了runの指標（clean_targetとall_eligibleの相対MSE、global NMSE、相関の中央値、
最後のミニバッチloss）、CUDAのピークメモリ。CSVは丸めない。`--summarize` は完了runを
clean_targetの相対MSE順に小数4桁で、変えた値（`changes`）とともに表示し、その後に未完了のrunを状態つきで表示する。

study_059の準備runでは、1runの `prediction.npy` が約275 MBになる。古いrunは手で削除する。

## 注意

- 指標は元のtest集合を再利用しており、独立なtest性能の推定ではない。
- 予測はtestトレースの真のRMS（oracle）で振幅を戻してから評価する。
- 多数のrunから最良を選ぶと楽観的になる。採用する条件は別のマスクか別の準備runで確かめる。
- 学習曲線は学習中のミニバッチloss（EMAを適用しない重み、padding込みのcanvas）で、検証lossではない。
- GPUのrunは近似的にしか再現しない。CPUでスレッド数を固定したrunはビット単位で再現する。
- このrunnerは準備runの実装hash（`verify_mvp_implementation`）を検証しない。
  代わりに実際に使ったソースのhashを `run.json` の `source_hashes` に記録する。

番号付きstudyへ移すときは、runの `config.resolved.yaml` から `model`・`training`・`prediction` を写す。
`time_sample_count` はすでに契約のTになっている。`validate_mvp_configs` は `model_seed` が契約のseedと
一致することを要求する。
