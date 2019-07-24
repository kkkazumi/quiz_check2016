import numpy as np
import scipy as sp
from scipy.stats import pearsonr

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (10,20,30):
    r = np.zeros(10)
    p = np.zeros(10)
    for i in range(10):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')

      #pearson r
      r[i], p[i] = pearsonr(mood_test, mood_est)

    #test output
    np.savetxt(dir_name+"/phi_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    np.savetxt(dir_name+"/phi_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')

