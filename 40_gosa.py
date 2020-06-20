import numpy as np
import scipy as sp
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

def min_max(trg_array,std_x, axis=None):
  min = std_x.min(axis=axis, keepdims=True)
  max = std_x.max(axis=axis, keepdims=True)
  result = (trg_array-min)/(max-min)
  print(trg_array.shape)
  return result,1.0/(max-min)

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (5,10,15,20,25):
    #set_num = 35
    diff = np.zeros(25)
    rate = np.zeros(25)
    for i in range(25):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_train_est = np.loadtxt(dir_name + "/TRAINestimated_phi" + i_csv, delimiter=',')

      mood_test_ans = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      mood_est,rate[i] = min_max(np.loadtxt(dir_name + "/TESTestimated_phi" + i_csv, delimiter=','),mood_train_est)

      diff[i] = np.mean(mood_test_ans - mood_est)
      print(diff[i],rate[i])
      plt.plot(mood_test_ans,label="answer")
      plt.plot(mood_est,label="estimated")
      plt.legend()
      plt.show()


    #np.savetxt(dir_name+"/phi_diff-"+str(set_num)+".csv",diff,fmt='%.5f',delimiter=',')

