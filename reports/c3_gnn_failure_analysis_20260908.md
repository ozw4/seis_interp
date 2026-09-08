# C3 GNN初回pilotの失敗原因の考察（2026-09-08）

主因として最も強く支持されるのは、train内の極端な有限振幅に支配された全体RMSである。
通常の波形が約10^-34の尺度へ縮小され、入力情報の消失、学習の数値的不安定、
逆正規化による微小出力の巨大化が起こる条件になっていた。
今回の−567.8483 dBだけからGNNの構造的な適否を判断することはできない。

根拠は[初回報告](c3_first_results_20260908.md)と、今回追加したCPU監査である。
実データ・設定・checkpointの変更や、本モデルの追加学習は行っていない。

| 確認項目 | 値・意味 |
|---|---|
| 全trainの固定RMS | 4.1921×10^34 |
| RMSを支配する極端値 | abs > 1e30の710 sample／235 traceがtrain energyのほぼ全量 |
| validation targetの物理RMS | 9.9386 |
| 同RMSの正規化後 | 2.3708×10^-34 |
| その二乗 | 5.6207×10^-68 |
| float32の最小正subnormal | 1.4013×10^-45 |
| finalの物理誤差RMS | 2.4532×10^29 |
| 同誤差RMSを正規化尺度へ戻した値 | 5.8521×10^-6 |

したがって、正規化後の予測が10^-6程度に小さくなっていても、必要な信号尺度10^-34からは
約28桁離れている。小さいnormalized lossを通常振幅の復元成功とは解釈できない。
RMSの巨大さはnative記録、native前処理の再実行、独立float64集計の3通りで一致した。
極端値の元SEG-Yにおける由来や変換経路は未確認であり、破損・欠損符号・読み出し誤り等の
どれかへ原因を断定していない。

[encoderと出力headのCPU監査](../runs/study_028_c3_first_results/20260908T231055830085Z_edda0ae6aa06_gnn_encoder_scale_audit/audit.json)では、
物理振幅1・10・1000・1,000,000の合成定数波形を固定RMSで割ってencoderへ入力した。
4例すべてで、出力はゼロ波形を入れた場合とbitwiseで一致した。
[入力正規化](../src/seis_interp/data/masked_trace_source.py)の後に、
[encoder](../src/seis_interp/models/trace_codec.py)はbias付きConv1dとGroupNormを使う。
この尺度では通常波形の変動がbias等に比べ極小となり、数値表現に残らない経路が実証された。
この診断は合成定数波形の確認であり、全実波形について同じ比較を行ったものではない。

同じ保存checkpointの最終head biasは正規化単位で3.9230×10^-6、
[逆正規化](../src/seis_interp/training/relational_trace_graph_prediction.py)で物理単位へ換算すると
1.6446×10^29になる。これはbias項単独の換算値で、総予測には他の項も加わるが、
実際の誤差RMSと同じ桁である。weightがNaNやInfinityになる必要はない。
現在のGNNはdecoder出力を直接使う。共有codecのdocstringにあるresidualという語とは異なり、
物理波形のIDW baselineを加算する構造ではない。初期headはweight・biasともゼロで、
初期のtarget出力はzero-fillに相当する。

学習履歴の最初の4stepはbatch lossが0、その後は次のように立ち上がった。

| step | native batch loss |
|---|---:|
| 5 | 4.1966×10^-45 |
| 6 | 3.6139×10^-37 |
| 7 | 2.4790×10^-29 |
| 8 | 1.2875×10^-21 |
| 9 | 5.4892×10^-14 |
| 10 | 3.7454×10^-8 |

float32の二乗誤差が0へunderflowしても、autogradの微分2×誤差は非ゼロになり得る。
今回のAdamWはlr=1e-4、既定eps=1e-8である。勾配の二次momentが十分小さい領域では、
更新の分母がepsに支配される。更新式とepsの定義は
[PyTorch 2.5実装](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.0/torch/optim/adamw.py)で確認した。
[単一parameterのCPU算術例](../runs/study_028_c3_first_results/20260908T230956420876Z_edda0ae6aa06_gnn_scalar_arithmetic_audit/audit.json)でも、
同じoptimizer設定と2.3708×10^-34の教師値で、lossが0でも更新が始まり、
parameterが数stepで10^-5程度へovershootする挙動を再現できた。
これは実モデルの勾配履歴の再生ではなく、初期のloss増加と整合する有力な機構の確認である。
AdamW単独の一般的な欠陥や、全誤差の寄与率を実証したものではない。

[実際の教師露出の監査](../runs/study_028_c3_first_results/20260908T230908953726Z_edda0ae6aa06_gnn_training_exposure_audit/audit.json)では、
seedと既存episode生成器から200step分のquery順を再構成した。
使用した教師は800本すべて異なり、canonical train poolの0.0698%だった。
step87に限り、trace836371の2 samplesがabs > 1e12、うち1 sampleがabs > 1e30だった。
最大値はsample107（0.856秒）の−3.2453×10^32で、正規化後でも−0.0077となる。
通常の教師に比べ桁違いの値で、step87のlossは3.9024×10^-8だった。
一方、step5および最大batch lossのstep12の教師は通常振幅である。
この極端な教師値が、初期のloss上昇を開始させたわけではない。
観測supportとして学習中に読まれた波形全体は、今回の露出監査の対象外である。

小型width32・各関係2近傍・200stepという制約も精度を制限し得る。
ただし、正規化と数値スケールの問題を解消していない状態で、学習回数やモデル容量だけを
増やすことを最初の対処にはしない。800 queryという露出量から、十分な学習収束も主張できない。

学習時と独立推論のSNR差は約0.0009 dBであり、−567.8483 dBという崩壊を説明する
主要因ではない。元の厳密一致検査は不合格のまま保持するが、精度問題の優先順位では
巨大な正規化尺度と入力情報消失を先に扱う。
交互sampleの模様については、保存decoderへ時間一定latentを入力しても偶奇差が生じた。
stride2の転置畳み込みの位相依存が関与する可能性はあるが、実query全体の原因は未確定である。

次の調査順序を提案する。

1. 極端値のtrace／sampleを元SEG-Yまで追い、振幅変換・sample形式・読み出し対応と照合する。
2. 原因に応じてデータQCと異常値の扱いを明文化する。有限値検査だけでは今回の値を検出できない。
3. 通常のtrain波形が適切な数値範囲に入る正規化を検証し、encoderが波形差を保持すること、
   物理単位の誤差が小規模学習で下がることを確認する。robust RMSだけへの置換では、
   残った極端値が別のoverflowを起こす可能性があるため、データQCと合わせて検討する。
4. その後、学習予算・近傍数・モデル幅の比較と、process間の数値再現性を検証する。

既存の固定suiteや初回結果は、その調査・修正の際も追跡できる形で保持する。
