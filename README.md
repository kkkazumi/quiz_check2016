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
  - 基底とNNの結果を箱ひげグラフで表示する(corr)

* 40_compare_box_gosa.py
  - 基底とNNの結果を箱ひげグラフで表示する(gosa)

* 40_compare_gosa2.py
  - 教師データ個数を変えたときの誤差を比較するグラフ

* 40_compare_corr2.py
  - 教師データ個数を変えたときの相関を比較するグラフ

* 40_plot.py
  - 基底関数による推定結果を１こずつ表示する

* w_phiaqu_check.py
  - 基底関数の正しさ度合いCと重み係数Wの相関を各表情毎に出力

  - 関数の正しそうどは、/media/kazumi/4b35d6ed-76bb-41f1-9d5d-197a4ff1a6ab/home/kazumi/prog/est_pred/out_new.py

    - out_kitekan_new.py という名前で、このディレクトリに追加。2020/06/09
    - out_kitekan_all.py も追加。３０個全部のデータについて、基底関数の正しそうな度合いを出す。

  - 因子毎の重みの大きさは、/media/kazumi/4b35d6ed-76bb-41f1-9d5d-197a4ff1a6ab/home/kazumi/prog/quiz_anal/out_corr.py


* good_data.py

  - テストデータと訓練データがそれぞれ基底関数に合っているか算出し、書き出す。
  - 推定したのが、テストデータの前半か、後半か、番号を調べる。
  - 各因子との相関とかかな・・

* kitekan_corr_est_corr.py

  - 重みの正しさ度合いと推定結果との関係を調べる
  - 重みの正しさ度合いは、基底関数の正しさと重みとの相関係数のこと。
  - 基底関数の正しさは、各基底関数にFとM(実際に観測したFとアンケートM)を入力した値と、表情との相関係数のこと。
* total_violin.py

  - とりあえず全体の分布を見る。（２５回データの時）

