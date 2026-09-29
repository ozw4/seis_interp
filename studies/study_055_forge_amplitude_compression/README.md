ReGSI-realで、トレース毎RMSを使わない場合の振幅圧縮の効果を調べる。

条件の正本は [config.yaml](config.yaml)、入力は [inputs.yaml](inputs.yaml)。
観測データだけから求めた共通尺度で線形変換する条件と、同じ尺度でasinh圧縮する条件を比較する。
振幅変換の選択は比較結果から判断する。実行結果は各runのmanifestとsummaryに保存する。
判断理由と従来条件からの変更点は [decisions.md](decisions.md) に記録する。

実行入口は `scripts/forge_amplitude_compression.py`。
`--variant linear` と `--variant asinh` をそれぞれ独立した初期重みから実行する。
asinhの尺度比較では、`config.yaml` の `asinh_scale_sweep` に従い、
`--scale-multiplier` で観測データ由来の基準尺度に対する倍率を指定する。
倍率1は既存の実行を比較に使う。

実行記録：

- [linear](../../runs/study_055_forge_amplitude_compression/20260921T084353194259Z_51acdb640a29_linear/manifest.json)
- [asinh](../../runs/study_055_forge_amplitude_compression/20260921T084354787401Z_51acdb640a29_asinh/manifest.json)

尺度比較の投入記録は [plan.json](../../runs/study_055_forge_amplitude_compression/20260921T104002335539Z_51acdb640a29_scale_sweep/plan.json)。
各条件の開始・終了とrunへの対応は
[GPU 0](../../runs/study_055_forge_amplitude_compression/20260921T104002335539Z_51acdb640a29_scale_sweep/status_cuda0.json)・
[GPU 1](../../runs/study_055_forge_amplitude_compression/20260921T104002335539Z_51acdb640a29_scale_sweep/status_cuda1.json)
の実行記録で確認する。
