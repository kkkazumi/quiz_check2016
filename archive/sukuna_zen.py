import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import math

import matplotlib.pyplot as plt
from out_new import *

TEST_NUM = 10
FACTOR_NUM = 10
r=np.zeros((3,9))
m_diff=np.zeros((3,9))
phi_train=np.zeros((3,10,9))
phi_test=np.zeros((3,10,9))


def cos_sim(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

dist_fact = np.zeros((3,10,9))
mm = np.zeros((3,10,9))
hh_count=0
for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (10,20,30):
    for i in range(10):#calculate the avelage value of estimated accuracy for datum in this loop
      weight = np.zeros((4,10))
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      factor_train = np.loadtxt(dir_name + "/factor_train" + i_csv ,delimiter=',')
      factor_test = np.loadtxt(dir_name + "/factor_test" + i_csv ,delimiter=',')

      face_train = np.loadtxt(dir_name + "/face_train" + i_csv ,delimiter=',')
      face_test= np.loadtxt(dir_name + "/face_test" + i_csv ,delimiter=',')

      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      mood_train = np.loadtxt(dir_name + "/mood_train" + i_csv ,delimiter=',')
      mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')

      fig ,ax1 = plt.subplots()
      ax1.plot(mood_test,color="orange",label="ans")
      ax2 = ax1.twinx()
      ax2.plot(mood_est,label="estimated")
      ax1.legend()

      weight[0,:] = np.loadtxt(dir_name + "/hap_weight" + i_csv, delimiter=',')
      weight[1,:] = np.loadtxt(dir_name + "/sup_weight" + i_csv, delimiter=',')
      weight[2,:] = np.loadtxt(dir_name + "/ang_weight" + i_csv, delimiter=',')
      weight[3,:] = np.loadtxt(dir_name + "/sad_weight" + i_csv, delimiter=',')


      m_diff[set_num/10-1,name_num] = pearsonr(mood_test,mood_est)[0]
      mm[(set_num/10-1),i,name_num] = pearsonr(mood_test,mood_est)[0]
      plt.title(str(mm[(set_num/10-1,i,name_num)]))
      plt.show()

      phiphi= phi_out(factor_test[:,:],mood_test[:])
      #diff = face_test - phi_out(factor_test[:,:],mood_test[:])
      corr = np.zeros(4)
      for enum in range(4):
        s = 0
        ss = np.zeros(4)
        for fnum in range(10):
          #ss[enum] = phiphi[:,fnum,enum]
          #s += phiphi[:,fnum,enum]*weight[enum,fnum]
        #corr[enum] = pearsonr(face_test[:,enum],s)[0]
          for tt in range(10):
            corr[enum] += pearsonr(weight[enum,:],phiphi[tt,:,enum])[0]
      phi_test[(set_num/10-1),i,name_num] = np.average(corr[~np.isnan(corr)])

      phiphi= phi_out(factor_train[:,:],mood_train[:])

      corr = np.zeros(4)
      for enum in range(4):
        s = 0
        for fnum in range(10):
          for tt in range(set_num):
            #s += phiphi[:,fnum,enum]*weight[enum,fnum]

            corr[enum] += pearsonr(weight[enum,:],phiphi[tt,:,enum])[0]
      phi_train[(set_num/10-1),i,name_num] = np.average(corr[~np.isnan(corr)])


      dist_fact[(set_num/10-1),i,name_num] = phi_test[(set_num/10-1),i,name_num]
      hh_count+=1



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
color = ('r','b','g')

for ii in range(3):
  plt.subplot(2,2,ii+1)
  #print(r[ii,:],np.average(r[ii,:]))
  ##plt.scatter(r[ii,:],m_diff[ii,:],color=color[ii],label=str((ii+1)*10))

  p = pearsonr(dist_fact[ii,~np.isnan(mm[ii,:,:])],mm[ii,~np.isnan(mm[ii,:,:])])
  plabel = "{0:.3f}".format(p[0])
  plt.scatter(dist_fact[ii,:,:],mm[ii,:,:],color=color[ii],label=str(p))
  #plt.ylim(-1,1)
  #plt.xlim(0,15)
  plt.legend()
  plt.xlabel("cos sim")
  plt.ylabel("mood correlation")

rr=dist_fact.reshape(-1,1)
mmm=mm.reshape(-1,1)
print(mmm)
pp = pearsonr(rr[~np.isnan(mmm)],mmm[~np.isnan(mmm)])
plt.title(str(p))
plt.show()
  

#plt.show()
