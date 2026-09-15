# v3 parameter-matched NeRSI

Status: running.

CCNetの111,969パラメータに近いNeRSIで、固定v3の全Tに対する物理振幅の
平均trace SNRを評価する。実行条件と入力lock参照は[config.yaml](config.yaml)。
既存のNeRSI正本・CCNet結果は変更しない。

NeRSIは111,875パラメータで、CCNetとの差は94個（約0.084%）。
Fourier mapping、kernel、decoder段数、upsample、activationを保持し、
encoder幅・latent幅・decoder幅だけを縮小する。
decoder比4:2:1とlatent=第1decoder幅を保った整数幅候補から、
パラメータ数の差だけで選択した。性能を見た構成探索は行っていない。
元のNeRSI条件は64,127,857パラメータ。

同じ初期化seedから50,000更新する。MSE、O-only Global RMS、EMA、
学習率、sampling、固定入力・評価器は元のNeRSI条件から変更しない。
時間shearは使用しない。checkpoint継続ではない。

比較元の全target平均trace SNRは、NeRSI正本15.5856 dB、
CCNet＋EMA・50,000更新16.0672 dB。
パラメータ数を揃えてもFLOPs・教師trace提示数・探索量の一致は意味しない。
v3はT参照済みの開発ケースであり、独立test性能や一般的手法優位を主張しない。
