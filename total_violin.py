import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr

test_len = 25
hito_num = 9

nn_ave = np.zeros((1,hito_num*test_len))
phi_ave = np.zeros((1,hito_num*test_len))

for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)

  set_num = 25

  nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
  phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")

  nn_ave[0,name_num*test_len:name_num*test_len+test_len]= nn_corr#np.where(nn_corr>0,nn_corr,None)
  phi_ave[0,name_num*test_len:name_num*test_len+test_len]= phi_corr#np.where(phi_corr>0,phi_corr,None)

sns_plot = sns.distplot(nn_ave[~np.isnan(nn_ave)],bins=20,label="neural network")
sns_plot = sns.distplot(phi_ave[~np.isnan(phi_ave)],bins=20,label="proposed method")
plt.legend()
plt.title("histogram of correlation")
plt.ylabel('frequency')
plt.xlabel('correlation between estimated mood and self-assessed mood')
print(np.mean(nn_ave[~np.isnan(nn_ave)]))
print(np.mean(phi_ave[~np.isnan(phi_ave)]))

#plt.ylim([0,0.3])

fig = sns_plot.get_figure()
plt.show()
#plt.savefig('5-40diff_matome.eps')
