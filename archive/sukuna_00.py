import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import math

import matplotlib.pyplot as plt
from conv_num import * #for id number

TEST_NUM = 10
FACTOR_NUM = 10
r=np.zeros((5,9))
m_diff=np.zeros((5,9))

zero_count = 0
info = np.empty(100)

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (10,15,20,25,30):

    dist_fact = np.zeros(TEST_NUM)
    dist_face = np.zeros(TEST_NUM)
    dist_mood = np.zeros(TEST_NUM)
    distance = np.zeros(TEST_NUM)

    for i in range(10):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      factor_train = np.loadtxt(dir_name + "/factor_train" + i_csv ,delimiter=',')
      factor_test = np.loadtxt(dir_name + "/factor_test" + i_csv ,delimiter=',')

      face_train = np.loadtxt(dir_name + "/face_train" + i_csv ,delimiter=',')
      face_test= np.loadtxt(dir_name + "/face_test" + i_csv ,delimiter=',')

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      mood_train = np.loadtxt(dir_name + "/mood_train" + i_csv ,delimiter=',')
      mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')

      if (np.sum(mood_est)==0):
        info[zero_count]=set_num
        print("estimation is all 0",set_num)
        if(np.sum(mood_test)==10):
          print("ans is all 10",set_num)
        if (zero_count==0):
          zero_factor_train=factor_train
          zero_count = 1
        else:
          zero_factor_train=np.append(zero_factor_train,factor_train,axis=0)
          zero_count += 1

print("zero",zero_factor_train)
print("10",np.count_nonzero(info==10))
print("15",np.count_nonzero(info==15))
print("20",np.count_nonzero(info==20))
print("25",np.count_nonzero(info==25))
print("30",np.count_nonzero(info==30))
