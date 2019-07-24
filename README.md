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

