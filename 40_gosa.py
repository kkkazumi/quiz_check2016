import numpy as np
import scipy as sp
from scipy.stats import pearsonr

#gosa version

def norm(array,a_max,a_min):
  return (array - np.ones_like(array)*a_min)/(a_max-a_min)
  

g = np.zeros(30)

for name_num in range(9):
  dir_name = "./jrm_test/4th_test/" + str(name_num+1)
  for set_num in (5,10,15,20,25,30,35,40):
    gosa = np.zeros(30)
    for i in range(30):#calculate the avelage value of estimated accuracy for datum in this loop
      print i
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      mood_train= np.loadtxt(dir_name + "/mood_train" + i_csv ,delimiter=',')

      #mood_est = np.loadtxt(dir_name + "/estimated_dummy" + i_csv, delimiter=',')
      mood_esta = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')
      mood_estb = np.loadtxt(dir_name + "/estimated_phi_before" + i_csv, delimiter=',')

      mood_est = np.hstack((mood_esta,mood_estb))
      mood_ans = np.hstack((mood_test,mood_train))

      m_max = np.max(mood_est)
      m_min = np.min(mood_est)

      ans_max = np.max(mood_ans)
      ans_min = np.min(mood_ans)

      mood_test = norm(mood_test,ans_max,ans_min)

      mood_est_test = norm(mood_esta,m_max,m_min)
      print mood_est_test
      print mood_test

      if np.sum(mood_est) == 0: 
        g[i] =None
      else:
        g[i] = np.average(mood_est_test - mood_test)

    #test output
    #np.savetxt(dir_name+"/dummy_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    ###np.savetxt(dir_name+"/phi_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    np.savetxt(dir_name+"/phi_gosa-"+str(set_num)+".csv",g,fmt='%.5f',delimiter=',')

