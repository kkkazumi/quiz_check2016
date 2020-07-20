import numpy as np
from scipy.stats import pearsonr
import sys

import matplotlib.pyplot as plt

def min_max(trg_array,std_x, axis=None):
  _min = std_x.min(axis=axis, keepdims=True)
  _max = std_x.max(axis=axis, keepdims=True)
  mmax = 1.6*_max - 0.3*_min
  mmin = 1.6*_min - 0.3*_min
  result = (trg_array-mmin)/(mmax-mmin)
  return result,1.0/(mmax-mmin)

if sys.argv[1] == 'phi':
  num = 0
elif sys.argv[1] == 'nn':
  num = 1

filename = [['TESTestimated_phi','estimated_nn'],['TRAINestimated_phi','TRAINestimated_nn']]
  
for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (25,26,27,28,29):
    #set_num = 35
    diff = np.zeros(29)
    rate = np.zeros(29)
    for i in range(29):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      _mood_train_ans = np.loadtxt(dir_name + "/mood_train" + i_csv, delimiter=',')
      mood_test_ans,rate[i] = min_max(np.loadtxt(dir_name + "/mood_test"+i_csv, delimiter=','),_mood_train_ans)

      _mood_train_est = np.loadtxt(dir_name + "/"+filename[1][num] + i_csv, delimiter=',')
      mood_est,rate[i] = min_max(np.loadtxt(dir_name + "/"+filename[0][num] + i_csv, delimiter=','),_mood_train_est)
      diff[i] = mood_est - mood_test_ans
      #mood_est,rate[i] = min_max(np.loadtxt(dir_name + "/TESTestimated_phi" + i_csv, delimiter=','),mood_train_est)

      #print(np.count_nonzero(array[array>0]))
    print(name_num,'set_num',set_num,'diff',diff)

    np.savetxt(dir_name+"/"+sys.argv[1]+"_DIFF-"+str(set_num)+".csv",diff,fmt='%.5f',delimiter=',')

