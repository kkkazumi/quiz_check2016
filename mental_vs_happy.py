import numpy as np
import scipy as sp
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (5,10,15,20,25):
    #set_num = 35
    r = np.zeros(25)
    p = np.zeros(25)
    for i in range(25):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      happy_test = np.loadtxt(dir_name + "/face_test" + i_csv, delimiter=',')
      print(happy_test.shape)


      #pearson r
      r[i], p[i] = pearsonr(mood_test, happy_test[:,0])
      fig, ax=plt.subplots()
      ax1 = ax.twinx()
      ax.plot(mood_test,label="mood")
      ax1.plot(happy_test[:,0],linestyle='dashed',label="happy",color="red")
      plt.title('corr:'+str(r[i]))
      plt.legend()
      plt.show()

      print(i,r[i])
