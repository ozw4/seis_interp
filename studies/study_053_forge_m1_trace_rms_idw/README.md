# FORGE M1 トレースRMS正規化 + IDW振幅復元

study_052の結果を受け、ユーザー指定の振幅処理変更を行う追試。旧MVP v0.1の「trace-wise RMSを使わない」条件とは別実験として扱う。testを既に観察した後の設計変更であることを明記し、旧runを上書きしない。

## 固定条件

- 6手法、model seed 7301、構造・optimizer・学習step・inner mask規則・loss・checkpoint規則はstudy_052から維持。
- study_052のmaskファイルをhash照合してそのままコピー。39,321 test / 9,830 observed、mask hashも同一。mask生成namespaceはstudy_052のまま。
- 各observedの全4,001サンプルからfloat64でRMSを求め、各波形を自身のRMSで割る。AGC・demean・bandpass・波形空間内挿は行わない。
- IDWは全6手法共通の投影格子。各軸±2セルのChebyshev stencil内に存在するouter observedのみを使う。距離はsource/receiverの4軸格子差×candidate_grids.jsonの格子間隔（m）のユークリッド距離。重みは距離の逆二乗、RMSそのものの加重平均。品質によるradius/powerの探索なし。
- observedのスケールは実測RMS、testのスケールはIDW。自然欠測セルはdonorにも評価対象にも含まない。支持点のない欠測cell、ゼロまたは非有限RMSは停止。
- 正規化したobservedだけをモデルへ渡し、内部global scalarは1。元Global RMSは監査用に残すがモデルの振幅変換には使わない。
- inner targetの自身のRMSは教師波形の正規化にだけ使う。隠した波形・RMSを入力feature、近傍距離、inner contextへ渡さない。学習lossは単位RMS教師に対する既存masked_trace_relative_mse。inner段階では物理振幅へ戻さず、IDWを学習lossへ使わない。
- 推論出力にtestのIDW RMSを1回掛けて元単位へ戻す。予測自体のRMSを強制的に1にする処理は行わない。ゼロ予測はゼロのまま。
- ReGSI-real/gridは同一の振幅スケール表・初期重み・seedを使用。geometry以外の設定は同一。
- 全6手法が予測検査を通るまでtest正解振幅を読まない。評価は元単位、clean-target/all-eligibleと既存の共通指標を維持。

## 準備と実行

```bash
.venv/bin/python scripts/forge_m1_mvp.py prepare --study studies/study_053_forge_m1_trace_rms_idw
.venv/bin/python scripts/forge_m1_mvp.py preflight --preparation <準備run>
.venv/bin/python scripts/forge_m1_mvp.py run --preparation <準備run> --preflight <preflight run>
```

採用prepare/preflightはprepared_inputs.lock.jsonに固定する。実験成果物はruns/study_053_forge_m1_trace_rms_idw配下。本番runの開始はこの準備作業には含まない。
