import numpy as np
import scipy as sp
from scipy.stats import pearsonr

import matplotlib.pyplot as plt


nn_num = np.zeros((3,9))
phi_num = np.zeros((3,9))

nn_ave = np.zeros((3,9))
phi_ave = np.zeros((3,9))



for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)

  for set_num in (10,20,30):
    r = np.zeros(10)
    p = np.zeros(10)

    nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
    nn_p = np.loadtxt(dir_name + "/nn_p-" + str(set_num) + ".csv")

    phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")
    phi_p = np.loadtxt(dir_name + "/phi_p-" + str(set_num) + ".csv")

    print(name_num , "+", set_num)
    print(name_num,set_num,"---nn:",np.average(nn_corr[~np.isnan(nn_corr)]))
    print(name_num,set_num,"--phi:",np.average(phi_corr[~np.isnan(phi_corr)]))

    nn_ave[set_num/10-1,name_num]= np.average(abs(nn_corr[~np.isnan(nn_corr)]))
    phi_ave[set_num/10-1,name_num]= np.average(abs(phi_corr[~np.isnan(phi_corr)]))

    nn_num[set_num/10-1,name_num] = np.count_nonzero(abs(nn_corr)>0.2)
    phi_num[set_num/10-1,name_num] = np.count_nonzero(abs(phi_corr)>0.2)
    print(set_num/10-1,name_num)
    print(nn_num[set_num/10-1,name_num])

    print(np.count_nonzero(nn_corr>0),np.count_nonzero(phi_corr>0))

    vsjudge = 'count' + str(np.count_nonzero(abs(nn_corr)>0.2)) + "vs" + str(np.count_nonzero(abs(phi_corr)>0.2))
    vsave = 'ave'+str(abs(nn_ave[set_num/10-1,name_num])) + "vs" + str(abs(phi_ave[set_num/10-1,name_num]))

    left = np.arange(10)
    width = 0.3

    g1=plt.bar(left,abs(nn_corr),color='b',label='nn_corr',width=width,align='center')
    g2=plt.bar(left+width, abs(phi_corr), color='r',label='phi_corr', width=width, align='center')
    plt.legend(handles=[g1,g2],loc='best',shadow=True)
    plt.xlabel('set num')
    plt.ylabel('correlation')
    plt.ylim(0,1)
    plt.title("no."+str(name_num) +"-"+ str(set_num)+"\n" + vsjudge+"\n" +vsave)
    plt.show()

pave = np.zeros(3)
nave = np.zeros(3)

plt.figure(figsize=(4,8))
for i in range(3):
  val = phi_ave[i,:]
  pave[i] = np.average(val[~np.isnan(val)])
  val = nn_ave[i,:]
  nave[i] = np.average(val[~np.isnan(val)])


g1=plt.plot((10,20,30),pave,color='r',label='proposed method',marker='o',linestyle='dashdot')
g2=plt.plot((10,20,30),nave,color='b',label='nn',marker='o',linestyle='dashed')
plt.xlabel('number of training data',fontsize=15)
plt.ylabel('average of correlation',fontsize=15)
plt.legend()
#plt.show()
plt.savefig('ave_matome.eps')

pcorr = np.zeros(3)
ncorr = np.zeros(3)

plt.figure(figsize=(4,8))

for i in range(3):
  val2 = phi_num[i,:]
  pcorr[i] = np.sum(val2)
  val2 = nn_num[i,:]
  ncorr[i] = np.sum(val2)


g1=plt.plot((10,20,30),pcorr,color='r',marker='o',label='proposed method',linestyle='dashdot')
g2=plt.plot((10,20,30),ncorr,color='b',marker='o',label='nn',linestyle='dashed')
plt.xlabel('number of training data',fontsize=15)
plt.ylabel('number of correlation larger than 0.2',fontsize=15)

plt.legend()
#plt.savefig('corr_matome.eps')
plt.show()

for i in range(3):
  print('nn_num',nn_num[i,:])
  print('phi_num',phi_num[i])

  plt.figure(figsize=(8,4))

  left = np.arange(9)
  width = 0.3

  g1=plt.bar(left,phi_ave[i,:],color='pink',label='proposed method',width=width,align='center',tick_label=range(1,10))
  g2=plt.bar(left+width, nn_ave[i,:], color='none',edgecolor='skyblue',hatch='//////////',label='neural network', width=width, align='center',tick_label=range(1,10))
  plt.legend(handles=[g1,g2],loc='best',shadow=True,fontsize=10)
  plt.xlabel('user id',fontsize=12)
  plt.ylabel('the average of correlation',fontsize=12)
  #plt.ylabel('the number of correlation\n larger than 0.2',fontsize=12)
  #plt.ylim(0,7.3)
  #plt.title(str((i+1)*10))

  #plt.ylabel('count of up/keep')

  #plt.savefig(str(i)+'times_compare.eps')
  plt.show()
