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

* 40_gosa.py
  - 学習用データの因子Fと表情Sを用いて推定した気分M(TRAIN)のスケールを、アンケートMのスケールに合わせて出した縮尺をもとに、テストデータを用いて推定した気分M(TEST)の値を変換し、アンケートMとの誤差を算出する。

* 40_absgosa.py
  - 上のプログラムの、相対誤差検出バージョン

* 40_nntest.py
  - NNで気分推定器を作るプログラム

* 40_nnest.py
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


* face_check.py

  - 表情データはきてかんの重み付きわで表現可能であるか確認するためのプログラム


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

* train_corr_check.py

  - 訓練データに対する推定精度はどうだったか。

* combine_fig.py

  - グラフをまとめる。読み取るグラフを作るのは、out_kitekan_all.pyで、timeseries_graph/, diff_graph/, diff_graph/以下。
  - どんなグラフを作るかというと、基底関数の正しさを調べるグラフ。
  - それぞれ、

    - timeseries_graph/: 時系列３０点に対し、表情観測データと、各基底関数の出力を比較したもの
    - diff_graph/: 表情と各基底関数の出力の誤差を、各因子の値に対してプロットしたもの
    - factor_graph/: 表情の値と、各規定関数の出力を、各因子の値に対しプロットしたもの。
* mental_vs_happy.py
  - 気分と表情の相関がもともとあるならば、本手法はあまり要らないことになるので、調べておく。

* deltaM.py
  - 気分変化をとりあえず予測してみるプログラム。基底関数を使う意味があまり無いと思ったので、NNで予測してみる。
 

* deltaM_compare.py

  - 気分予測結果と、アンケート結果Mの変化をそれぞれ比較するプログラム。

* gosa_check.py

  - 推定Mが過去に推定したMより高いか低いかを予測した結果を出力する。
* k_mean_sudata.py

  - 教師データのクラスタリングをするよ
  - 因子分析もしてる。
