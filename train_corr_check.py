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

      mood_test = np.loadtxt(dir_name + "/mood_train" + i_csv ,delimiter=',')
      mood_est = np.loadtxt(dir_name + "/TRAINestimated_phi" + i_csv, delimiter=',')
      number = np.loadtxt(dir_name + "/selected_num_train" + i_csv, delimiter=',')


      #plt.plot((mood_test-np.min(mood_test))/(np.max(mood_test)-np.min(mood_test)),label="answered")
      #plt.plot((mood_est-np.min(mood_est))/(np.max(mood_est)-np.min(mood_est)),label="estimated")
      #plt.legend()
      #plt.show()


      #pearson r
      r[i], p[i] = pearsonr(mood_test, mood_est)
      #else:
      #  r[i], p[i] = None,None

      #diff[i] =np.mean(mood_test - mood_est/10.0)

    #test output
    #np.savetxt(dir_name+"/dummy_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    np.savetxt(dir_name+"/phi_train_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
    np.savetxt(dir_name+"/phi_train_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    ###np.savetxt(dir_name+"/phi_diff-"+str(set_num)+".csv",diff,fmt='%.5f',delimiter=',')
    #np.savetxt(dir_name+"/dummy_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
    ###np.savetxt(dir_name+"/phi_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
    #np.savetxt(dir_name+"/phi_rimp-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')

