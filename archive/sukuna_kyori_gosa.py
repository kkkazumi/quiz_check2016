import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import math

import matplotlib.pyplot as plt

TEST_NUM = 10
FACTOR_NUM = 10
r=np.zeros((3,9))
m_diff=np.zeros((3,9))


def cos_sim(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

dist_fact = np.zeros((3,10,10,9))
mm = np.zeros((3,10,9))
mm_p = np.zeros((3,10,9))
hh_count=0

mm_gosa = np.zeros((3,10,10,9))

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (10,20,30):
    for i in range(10):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      factor_train = np.loadtxt(dir_name + "/factor_train" + i_csv ,delimiter=',')
      factor_test = np.loadtxt(dir_name + "/factor_test" + i_csv ,delimiter=',')

      face_train = np.loadtxt(dir_name + "/face_train" + i_csv ,delimiter=',')
      face_test= np.loadtxt(dir_name + "/face_test" + i_csv ,delimiter=',')

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      mood_train = np.loadtxt(dir_name + "/mood_train" + i_csv ,delimiter=',')
      mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')

      selected_num_train = np.loadtxt(dir_name + "/selected_num_train" + i_csv, delimiter=',')
      selected_num_test = np.loadtxt(dir_name + "/selected_num_test" + i_csv, delimiter=',')
      #print(np.count_nonzero(selected_num_train>selected_num_test[0]))

      m_diff[set_num/10-1,name_num] = pearsonr(mood_test,mood_est)[0]
      mm[(set_num/10-1),i,name_num] = pearsonr(mood_test,mood_est)[0]
      mm_p[(set_num/10-1),i,name_num] = pearsonr(mood_test,mood_est)[1]

      mm_gosa[(set_num/10-1),i,:,name_num] = mood_test - mood_est

      #mm_gosa[(set_num/10-1),i,:,name_num] = mm_gosa[(set_num/10-1),i,:,name_num]/(np.max(abs(mm_gosa[(set_num/10-1),i,:,name_num])))

      for j in range(set_num):

        dist_fact[(set_num/10-1),i,:,name_num] = cos_sim(np.hstack((factor_test[i,:],mood_test[i],face_test[i,:])),np.hstack((factor_train[j,:],mood_train[j],face_train[j,:])))


color = ('r','b','g')

r=dist_fact.reshape(-1,1)
m=mm_gosa.reshape(-1,1)

for ii in range(3):
#  plt.subplot(2,2,ii+1)
  plt.scatter(dist_fact[ii,:,:,:],mm_gosa[ii,:,:,:],color=color[ii],label=str((ii+1)*10))

ii=0
p = pearsonr(r,m)
#plt.scatter(r,m,color=color[ii],label=str(p))
#plt.xlim(0.6,1)
#plt.ylim(-1,1)
plt.legend()
plt.xlabel("cos sim")
plt.ylabel("mood correlation")

plt.show()
  

#p = pearsonr(rr[~np.isnan(mm)],mm[~np.isnan(mm)])
#
#plt.title(str(p))
#plt.show()
