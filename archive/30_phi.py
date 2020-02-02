import numpy as np
import scipy as sp
from scipy.stats import pearsonr

TEST_NUM = 25

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  set_num = 25
  r = np.zeros(TEST_NUM)
  p = np.zeros(TEST_NUM)
  for i in range(TEST_NUM):#calculate the avelage value of estimated accuracy for datum in this loop
    i_csv = str(set_num) + "-" + str(i) + ".csv"

    mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
    #mood_est = np.loadtxt(dir_name + "/estimated_dummy" + i_csv, delimiter=',')
    mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')
    #selected = np.loadtxt(dir_name + "/selected_num_test" + i_csv, delimiter=',')
    #if selected[0] <= 20 or selected[0]==30 or selected[0] == 40:
    #  print selected

      #pearson r
    r[i], p[i] = pearsonr(mood_test, mood_est)
    #else:
    #  r[i], p[i] = None,None

  #test output
  #np.savetxt(dir_name+"/dummy_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
  ###np.savetxt(dir_name+"/phi_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
  np.savetxt(dir_name+"/phi_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')

  #np.savetxt(dir_name+"/dummy_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
  ###np.savetxt(dir_name+"/phi_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
  np.savetxt(dir_name+"/phi_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')

