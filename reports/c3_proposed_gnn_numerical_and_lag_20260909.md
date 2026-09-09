# C3 proposed GNN の数値再現性・FP32 予備測定・lag 診断 — 2026-09-09

**固定1,143 query の checkpoint 診断では、TF32 を無効化した2条件が CPU との固定許容値を満たした。観測 trace RMS を使う FP32 の予備測定は1更新を完了した。一方、学習する edge lag は固定訓練4本で21.1003から20.3853 dBへ悪化し、今回の本学習には含めない。** これらは別々の範囲を持つ診断であり、全欠測 target での GNN の10 dB達成やモデル採用を意味しない。

[global RMS・5,000更新の既報](c3_proposed_gnn_global_rms_5k_20260909.md)は、全58,999 target・22,655,616 samples の独立 frozen 予測で7.4002 dB、RMSE 4.2395だった。保存予測の全対象再採点は合格したが、訓練／独立予測間の誤差エネルギー照合と、CPU 復元の2基準は不合格だった。以下の部分診断によって、その[元の不合格記録](../runs/study_032_c3_proposed_gnn_10db/20260909T070636656122Z_e39df16d563c_global5k_completion_handoff/verification/result.json)を変更・取り消さない。

数値診断では、この global RMS モデルの final step5000 checkpoint を固定した。最初・中央・最後の native batch、512+512+119 = **1,143 query・438,912 samples** について、同じ保存済み入力 tensor と型付き graph を CPU と各 GPU process に渡した。観測入力の組み立てと graph の違いを除き、各条件は新規 process で順番に実行した。target 真値、test 波形は読まず、学習も物理品質の採点もしていない。[事前条件](../runs/study_032_c3_proposed_gnn_10db/20260909T074804795328Z_e39df16d563c_tf32_environment_diagnostic/request.json)。

下表の「環境」は `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE / NVIDIA_TF32_OVERRIDE` を表す。全条件で精度 setter は呼ばず、A/B は import 後の `high`・matmul TF32有効、C/D は `highest`・matmul TF32無効だった。cuDNN の Python property `allow_tf32` は全条件で true のままなので、この property だけを実効経路と同一視しない。

| CPU に対する GPU 条件 | 環境 | cuDNN benchmark | 正規化差 RMSE | 正規化差最大値 | physical relative L2 | 固定基準 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | 1 / 未設定 | true | 2.0540×10⁻⁴ | 7.8512×10⁻³ | 7.7893×10⁻⁴ | RMSE・最大値が不合格 |
| B | 1 / 未設定 | false | 1.3034×10⁻⁴ | 8.9693×10⁻³ | 4.9429×10⁻⁴ | RMSE・最大値が不合格 |
| C | 0 / 0 | true | 1.4861×10⁻⁷ | 6.0201×10⁻⁶ | 5.6448×10⁻⁷ | 3基準すべて合格 |
| D | 0 / 0 | false | 1.4499×10⁻⁷ | 6.0201×10⁻⁶ | 5.5073×10⁻⁷ | 3基準すべて合格 |

閾値は正規化差 RMSE ≤10⁻⁴、正規化差最大値 ≤10⁻³、physical relative L2 ≤10⁻³で、結果を見て変更していない。今回の正規化差は保存したモデル正規化出力同士を比較し、physical 差は global RMS を戻して float32 保存した出力同士を比較した。元の CPU 復元監査は physical 差を保存尺度で割る定義なので、同じ query 集合でも末尾桁までの一致は主張しない。[数値比較 CSV](../results/study_032_c3_proposed_gnn_10db/20260909T080729436879Z_e39df16d563c_numerical_and_lag_publication/numerical_comparisons.csv)。

A–B 間の physical relative L2 は7.2325×10⁻⁴、C–D 間は1.5004×10⁻⁷だった。この固定 checkpoint・入力・環境では、benchmark の切り替えだけでは CPU 差は解消せず、TF32 に関する環境変更を伴う C/D で大幅に縮小した。C/D も bitwise 一致ではない。環境変数2つを同時に変えているため、それぞれの寄与や個別 kernel の原因を分離した実験ではなく、過去の全 target の誤差エネルギー不一致を完全に説明したとはしない。[診断結果](../runs/study_032_c3_proposed_gnn_10db/20260909T074804795328Z_e39df16d563c_tf32_environment_diagnostic/result.json)。

PyTorch の公式説明では、`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` は TF32 を有効にし、matmul precision の設定より優先する。一方、`NVIDIA_TF32_OVERRIDE=0` は PyTorch 設定にかかわらず全 kernel の TF32 を無効化する。[公式環境変数仕様](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)。実験は PyTorch `2.5.0a0+b465a5843b.nv24.09`、CUDA12.6、cuDNN90400、H100 NVLで行われた。仕様の説明と、この実験で保存した実測 property・環境は区別して記録した。

続いて、`observed_trace_rms`・幅64・query batch128・学習率10⁻³・seed20260908の **fresh モデルと optimizer で1更新だけ**予備測定した。環境は0 / 0、import 後 `highest`、cuDNN benchmark trueで、初回 autotune を含む。QC train pool の最初の人工 mask episode から選んだ hidden 128本×384 samples をラベルとし、3,300本の observed support を入力として読んだ。validation target と test は読んでいない。

| FP32・1更新の予備測定 | 実測 |
| --- | ---: |
| graph/index 構築 | 1.4459 s |
| 観測・訓練ラベル読み出し | 0.4192 s |
| forward | 1.1596 s |
| backward・optimizer | 2.1234 s |
| worker 全体 / 起動側全体 | 9.2005 / 11.2383 s |
| GPU peak allocated / reserved | 8.2587 / 8.9632 GB |
| process lifetime peak RSS | 3.0092 GB |

GB は10⁹ bytes。forward 時間には初期重みを CPU に保存する1.4889 msも含む。初期重みは事前に seed から作った値と一致し、更新後の gradient はすべて有限だった。最初の更新で非ゼロ gradient を持った tensor は2個であり、既存の zero-head 初期化下での1更新の記録を、全層が十分学習した証拠とはしない。使用した重みと optimizer は本学習へ引き継がない。この測定は定常 throughput、5,000更新の所要時間・最大メモリ、品質の予測ではない。[予備測定 CSV](../results/study_032_c3_proposed_gnn_10db/20260909T080729436879Z_e39df16d563c_numerical_and_lag_publication/fp32_resource_pilot.csv)・[元記録](../runs/study_032_c3_proposed_gnn_10db/20260909T075825670757Z_e39df16d563c_observed_trace_rms_fp32_gpu_resource_audit/train_128.json)。

lag は別の CPU 診断である。[観測 RMS の固定訓練 fit](c3_proposed_gnn_amplitude_investigation_20260909.md)と同じ訓練4 query、126本の support、元の初期重み・乱数状態を使い、edge 幾何から時間ずれを学習する4個のゼロ初期係数だけを加えた。幅32、AdamW 学習率10⁻³、1,000更新、CPU 1 thread、seed20260908である。正の lag は sender を時間の進む側へ参照する符号とし、上限は生波形32 samples = 0.2560 sに制限した。固定 shear を事前に与える方法ではない。

| 同じ訓練4本・1,000更新（validation ではない） | 物理 SNR (dB) | 物理 RMSE |
| --- | ---: | ---: |
| observed trace RMS・lag なし | 21.1003 | 1.0110 |
| observed trace RMS・学習する edge lag | 20.3853 | 1.0977 |

差は **−0.7150 dB** で、この診断では改善しなかった。学習時間117.1028 s、全体118.1683 s。最終の全224 typed edge における時間ずれの絶対最大値は0.2574 sample = 2.0589 ms、飽和判定に達した edge は0だった。係数はゼロから変化し、重み・gradient・予測は有限、通常 checkpoint の CPU 復元は bitwise 一致した。入力とラベルは保存済みの同じ訓練配列で、validation/test 波形を使っていない。[固定訓練比較 CSV](../results/study_032_c3_proposed_gnn_10db/20260909T080729436879Z_e39df16d563c_numerical_and_lag_publication/fixed_training_lag_comparison.csv)・[監査結果](../runs/study_032_c3_proposed_gnn_10db/20260909T075344685724Z_e39df16d563c_learned_edge_lag_fixed_training_fit/verification.json)。この1 seed・4本の結果を受け、今回の本学習には lag を含めない。時間ずれを扱うモデル一般が不利だという結論ではない。

[集計 JSON](../results/study_032_c3_proposed_gnn_10db/20260909T080729436879Z_e39df16d563c_numerical_and_lag_publication/numerical_and_lag_summary.json)は3診断の範囲を分け、[manifest](../results/study_032_c3_proposed_gnn_10db/20260909T080729436879Z_e39df16d563c_numerical_and_lag_publication/manifest.json)に元ファイルと図表の SHA256 を記録した。本文は小数4桁を基本とし、小さい非ゼロの差は科学表記、機械可読値は丸めず保存している。新しい解析 run から compact JSON/CSV を byte-identical に採用し、既存 run・results・レポートは変更していない。集計自体では学習・forward・target 再採点・GPU 使用を行っていない。FP32 observed-trace RMS の5,000更新本学習の結果は含めず、GNN の全対象10 dB目標は未達として保持する。
