# 過去のモデル設定を含める探索

ユーザーはstudy_063のReGSI・NeRSIを停止し、過去の良かった設定が25構成に
含まれるよう探索することを指定した。ReGSIはstudy_044の50,000更新run
`20260918T151554Z_5d660f87b9b4_mask10_fourier16_width128_ema999_neighbors6_geometry500_attention_rms_50k`
のRMS attention集約・座標特徴500 m・k=6・時間dilation [1,2]を基準にする。
NeRSIはstudy_046の `20260915T020222Z_684c4e4c659a_base115_50k` の
Fourier40・encoder40・latent28・decoder[29,14,8]・β=1.15を基準にする。
核ノルム正則化を加えないλ=0を候補に含める。

ReGSIはモデル固定値を基準に合わせ、従来の時間dilation×kの25構成を保持する。
NeRSIはλ×βの25構成を保持し、βの5候補を1.05/1.075/1.10/1.15/1.25とする。
固定した候補を途中で拡張せず、評価指標V・checkpoint選択・3 seed割当を維持する。

ユーザーはMSE・lr=1e-3への復帰、CCNet5Dの停止と条件統一を指定した。
ReGSI・NeRSIに加えCCNet5Dも新規25構成とし、3手法のloss・learning rate・
optimizer・更新数・1更新64本・EMA・weight decay・学習seedを統一する。
CCNet5Dは従来のpatch×時間dilationの25候補と53/63 channelを保持する。
POCS・DRRは学習率や訓練lossを持たないため、既存の探索を継続する。

GPUの他ジョブが使用するメモリを踏まえ、ReGSIとCCNet5DはGPU 0で
1構成ずつ排他実行し、NeRSIはGPU 1で実行する。実測済みのピーク割当を
参考に構成ごとの最低空きメモリを定め、条件を満たさない間は待機する。
ReGSIはkごと、CCNet5Dはpatchと時間dilationの大きさごとに必要量を定める。
NeRSIは全構成で同じ必要量とする。候補順や学習条件は変更しない。
待機順とGPUメモリ管理は探索軸に含めず、全25構成と3 seedで同じ規則を使う。

同じ波形割当を維持するため、外側のmask namespaceはstudy_062、検証holdoutの
namespaceはstudy_063を使う。新しい4分割のinput artifactすべてが継続元のhashと
一致しない場合は実行を拒否する。元の中断workerは再開しない。
