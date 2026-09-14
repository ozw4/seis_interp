# v3 CCNet target-informed tuning

固定v3入力の全Tに対するphysical-amplitude trace SNR平均で14 dB超を目指す。
O-only Global RMS、MSE、Shearなし。Tは学習ラベルに使用しないが、
試行選択に参照するため独立testではない。既存の比較runを上書きしない。

入力はStudy 042のinputs.yamlとv3 condition lockを使用する。
mask10_5k.yamlはCCNetのinner maskを10%とし、他の設定を維持する。
外側の欠損率・配置・評価対象は変更しない。

mask10_20k.yamlは同じモデル・学習率・patchで20,000 updateを実行する。
重みは初期化seed 101から学習し、checkpoint再開は行わない。

| Config | Mean trace SNR (dB) | Global SNR (dB) | RMSE |
|---|---:|---:|---:|
| mask10_5k | 13.1816 | 12.5600 | 2.1824 |
| mask10_20k | 14.4375 | 13.8567 | 1.8797 |

結果run: `runs/study_043_c3_v3_ccnet_target_tuning/20260914T012855Z_03c2fc83df18_mask10_5k/ccnet5d`。

達成run: `runs/study_043_c3_v3_ccnet_target_tuning/20260914T013304Z_03c2fc83df18_mask10_20k/ccnet5d`。
全104,750 target traceのSNR平均は14.437486631276457 dBで、14 dB超を達成。
Global SNRは14 dB未満。O-only Global RMSは9.27911442626805、Shearなし、
全target coverage、Oのexact reinsertionを維持する。保存predictionの再採点はmetrics.jsonと一致。
これはTを参照した探索結果であり、凍結済みv3比較の正本を自動置換しない。
