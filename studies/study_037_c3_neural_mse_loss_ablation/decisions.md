# 実験条件

固定targetでのphysical-amplitude global SNR最大化はtarget SSE最小化と等価。
全手法が同じO-only Global RMSを使うため、正規化振幅MSEと物理振幅SSEは定数倍の関係にある。
trace-energy weightingの効果を調べるため、変更する因子をtraining.lossだけに限定する。

学習率0.001、5000 updates、seed、architecture、sampling、inner mask、推論・評価条件はStage-1と同一。
発散しても同一learning-rate条件での不安定性として記録し、再調整・自動retryしない。
学習lossにはmasked_trace_mseを採用する。先行研究との整合を優先するユーザー判断であり、
今回のSNRによる選択ではない。3手法ともSNRが低下した結果を含め全runを凍結する。
参照済みTでの探索結果を独立な性能検証とはせず、今後の検証にはStage Aのvalidation targetを用いる。
Stage-1 relative-loss baselineは変更・置換しない。
