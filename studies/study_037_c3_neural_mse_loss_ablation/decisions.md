# 実験条件

固定targetでのphysical-amplitude global SNR最大化はtarget SSE最小化と等価。
全手法が同じO-only Global RMSを使うため、正規化振幅MSEと物理振幅SSEは定数倍の関係にある。
trace-energy weightingの効果を調べるため、変更する因子をtraining.lossだけに限定する。

学習率0.001、5000 updates、seed、architecture、sampling、inner mask、推論・評価条件はStage-1と同一。
発散しても同一learning-rate条件での不安定性として記録し、再調整・自動retryしない。
参照済みTでの探索結果だけでは正式lossを採択せず、将来のStage Aのvalidation targetで判断する。
