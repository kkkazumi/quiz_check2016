import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr

test_len = 29
hito_num = 9

phi_ave = np.zeros((hito_num,test_len))

for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)

  set_num = 29

  phi_corr = np.loadtxt(dir_name + "/phi_diff-" + str(set_num) + ".csv", delimiter=",")

  kitekan_corr = np.loadtxt(dir_name + "/weigt_correct25-24.csv",delimiter=",")

  corr = pearsonr(kitekan_corr[~np.isnan(phi_corr)],phi_corr[~np.isnan(phi_corr)])
  #corr = np.corrcoef(kitekan_corr[~np.isnan(kitekan_corr)],phi_corr[~np.isnan(kitekan_corr)])[0,1]
  print(corr)

  plt.scatter(phi_corr,kitekan_corr,label=str(name_num))
  plt.xlabel("correctness of estimation")
  plt.ylabel("correctness of weighting")
  plt.show()

