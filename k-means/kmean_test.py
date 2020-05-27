import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
#教師データのベクトル（因子＋表情）をクラスタして、推定結果と相関するか見てみた。意味不明すぎ。

C_SIZE = 5
 
for set_num in (5,10,15,20,25,30,35,40):
  #set_num = 5

  for name_num in (0,1,2,3,4,5,6,7,8):
    #name_num = 0
    dir_name = "../jrm_test/" + str(name_num+1)
    data = np.zeros((30,set_num,10+4))

    for i in range(30):
      i_csv = str(set_num) + "-" + str(i) + ".csv"
      factor_train = dir_name + "/factor_train" + i_csv
      signal_train = dir_name + "/face_train" + i_csv
      #CAUTIONNNN!! train and test

      estimated = dir_name + "/TESTestimated_phi" + i_csv

      fact = np.loadtxt(factor_train,delimiter=',')
      face = np.loadtxt(signal_train,delimiter=',')
      data[i,:,:]= np.hstack((fact,face))
    corr = dir_name + "/phi_corr-"+str(set_num)+".csv"
    corr_data = np.loadtxt(corr,delimiter=",")
    vec = np.zeros((30,set_num))
    pred = KMeans(n_clusters=C_SIZE).fit_predict(data.reshape(-1,10))
    for i in range(30):
      vec[i,:] = pred[5*i:5*i+set_num]
    print(vec)
    pred = KMeans(n_clusters=C_SIZE).fit_predict(vec)
    print("data crust",pred)
    #print(pearsonr(pred,corr_data))
    plt.scatter(pred,corr_data,alpha=0.2,s=50)
    plt.xlabel("pred")
    plt.ylabel("corr score")
    plt.show()
    #pred = KMeans(n_clusters=10).fit_predict(corr_data.reshape(-1,1))
