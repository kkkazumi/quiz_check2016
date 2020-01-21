### jrm kosatsu nn vs phi
JRM考察のため書いてるプログラム。
NNとPhiの推定性能を比較するため、
１０点、２０点、３０点の学習データでそれぞれ推定精度がどう変わるかを確認しました。

- sukuna_datamake.py

  - prog to 30, 20 10 supervisor data estimation trial!
  - proposed method vs NN
  - but this prog is just for making random supervisor data.
 


- sukuna_nntest.py
  - prog to output estimation of m by nn
- sukuna_phi.py

  -  prog to output correlation estimated m by phi.


%%%%%%%%%%%%%%%%%%%%%%%%
- stdapvscorr.py

  - correlation of {predicted and answered} and stdap of factors

- plot_bar.py

  - bar1: correlation of face vs answered M
  - bar2: correlation of predicted M vs answered M

- cos_sim.py

  - cosine sim vs correlation of {predicted vs answered}

### 2nd trial
* 40_datamake.py
  - データセット作成用プログラム
  - 教師データセット数は5から40までのものをそれぞれ30通り作成する。

* 40_phi.py
  - 基底関数用いた気分推定手法による結果について、誤差を算出するだけのプログラム

* 40_nntest.py
  - NNで気分推定するプログラム

* 40_nngosa.py
  - NN用いた気分推定結果について、誤差を算出するだけのプログラム

* 40_compare_box.py
  - 基底とNNの結果を箱ひげグラフで表示する

* 40_plot.py
  - 基底関数による推定結果を１こずつ表示する
