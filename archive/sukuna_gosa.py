import numpy as np
import scipy as sp
from scipy.stats import pearsonr

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (5,10,15,20,25,30):
    r = np.zeros(30)
    p = np.zeros(30)
    gosa = np.zeros(30)
    for i in range(30):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')

      #mood_est = np.loadtxt(dir_name + "/estimated_nn" + i_csv, delimiter=',')
      mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')

      
      #pearson r
      #if np.max(mood_est) > 10:
      #  print 'mood test max',np.max(mood_test)
      #  print 'mood test mim',np.min(mood_test)
      #  print 'mood est max',np.max(mood_est)
      #  print 'mood est mim',np.min(mood_est)
      #r[i], p[i] = pearsonr(mood_test, mood_est)
      gosa[i] = np.average((mood_est - mood_test)/mood_test)
      val_max = np.max(mood_est)
      val_min = np.min(mood_est)
      #if phi
      #gosa[i] = gosa[i] / ((float)(val_max - val_min))
      print gosa[i]

    #test output
    #np.savetxt(dir_name+"/nn_gosa-"+str(set_num)+".csv",gosa,fmt='%.5f',delimiter=',')
    np.savetxt(dir_name+"/phi_gosa-"+str(set_num)+".csv",gosa,fmt='%.5f',delimiter=',')

