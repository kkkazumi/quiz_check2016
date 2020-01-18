import numpy as np
import scipy as sp
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (5,10,15,20,25,30,35,40):
    #set_num = 35
    diff = np.zeros(30)
    #p = np.zeros(30)
    for i in range(30):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      #mood_est = np.loadtxt(dir_name + "/estimated_dummy" + i_csv, delimiter=',')
      mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')
      #mood_esta = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')
      #mood_estb = np.loadtxt(dir_name + "/estimated_phi_before" + i_csv, delimiter=',')

      #mood_est = np.hstack((mood_esta,mood_estb))

      #m_max = np.max(mood_est)
      #m_min = np.min(mood_est)

      #selected = np.loadtxt(dir_name + "/selected_num_test" + i_csv, delimiter=',')
      #if selected[0] <= 20 or selected[0]==30 or selected[0] == 40:
      #  print selected

      #  #pearson r
      #  r[i], p[i] = pearsonr(mood_test, mood_est)
      #else:
      #  r[i], p[i] = None,None

      """
      print(mood_test,mood_est)
      plt.plot(mood_test*10.0,label="test")
      plt.plot(mood_est,label="estimated")
      plt.legend()
      plt.pause(0.1)
      plt.clf()
      """

      diff[i] =np.mean(mood_test - mood_est/10.0)

    #test output
    #np.savetxt(dir_name+"/dummy_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    ###np.savetxt(dir_name+"/phi_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    np.savetxt(dir_name+"/phi_diff-"+str(set_num)+".csv",diff,fmt='%.5f',delimiter=',')
    print(diff)

    #np.savetxt(dir_name+"/dummy_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
    ###np.savetxt(dir_name+"/phi_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
    #np.savetxt(dir_name+"/phi_rimp-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')

