import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import math

import matplotlib.pyplot as plt

TEST_NUM = 10
FACTOR_NUM = 10
r=np.zeros((3,9))
m_diff=np.zeros((3,9))

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

    dist_fact = np.zeros(TEST_NUM)
    dist_face = np.zeros(TEST_NUM)
    dist_mood = np.zeros(TEST_NUM)
    distance = np.zeros(TEST_NUM)

    m_diff[set_num/10-1,name_num] = pearsonr(mood_test,mood_est)[0]
    #print m_diff[set_num/10-1,name_num]

    for i in range(TEST_NUM):
      #print(factor_test[i,:])
      for j in range(set_num):
        #print(factor_train[j,:])
        dist_fact[i] += np.linalg.norm(factor_test[i,:]-factor_train[j,:],ord=2)
        dist_face[i] += np.linalg.norm(face_test[i,:]-face_train[j,:],ord=2)

      distance[i] = math.sqrt(dist_fact[i]*dist_fact[i]+dist_face[i]*dist_face[i])

    r[set_num/10-1,name_num]=pearsonr(dist_fact,dist_face)[0]
    

    '''
    plt.figure(figsize=(5,5))
    plt.scatter(dist_fact,dist_face,color='b')
    plt.xlabel("factor distance")
    plt.ylabel("mood diff")
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.title(str(r)+"\n"+str(dir_name)+"-"+str(set_num))
    '''

    '''
    plt.figure(figsize=(5,5))
    #plt.scatter(dist_face,dim,color='r')
    plt.scatter(r,abs(mood_test-mood_est),color='r')
    plt.xlabel("supervisor distance")
    plt.ylabel("mood diff")
    plt.ylim(0,20)
    plt.title('err')
    plt.show()
    '''
for ii in range(3):
  plt.scatter(r[ii,:],m_diff[ii,:],color='r')
  plt.show()
  
rr=r.reshape(-1,1)
mm=m_diff.reshape(-1,1)

evalr=pearsonr(rr[~np.isnan(mm)],mm[~np.isnan(mm)])
plt.scatter(rr[~np.isnan(mm)],mm[~np.isnan(mm)],color='r')
plt.title(str(evalr))
plt.xlabel("distance")
plt.ylabel("mood correlation")

plt.show()
