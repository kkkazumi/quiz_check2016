import numpy as np
import scipy as sp
from scipy.stats import pearsonr

import matplotlib.pyplot as plt


nn_num = np.zeros((3,9))
phi_num = np.zeros((3,9))


for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)

  for set_num in (10,20,30):
    r = np.zeros(10)
    p = np.zeros(10)

    nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
    nn_p = np.loadtxt(dir_name + "/nn_p-" + str(set_num) + ".csv")

    phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")
    phi_p = np.loadtxt(dir_name + "/phi_p-" + str(set_num) + ".csv")

    #print(name_num,set_num,"---nn:",np.average(nn_corr[~np.isnan(nn_corr)]))
    #print(name_num,set_num,"--phi:",np.average(phi_corr[~np.isnan(phi_corr)]))

    nn_num[set_num/10-1,name_num] = np.count_nonzero(nn_corr>0.2)
    print(set_num/10-1,name_num)
    print(nn_num[set_num/10-1,name_num])
    phi_num[set_num/10-1,name_num] = np.count_nonzero(phi_corr>0.2)

    print(np.count_nonzero(nn_corr>0),np.count_nonzero(phi_corr>0))
    print(np.count_nonzero(nn_corr>0.2),np.count_nonzero(phi_corr>0.2))

    print('nn_p',np.where(nn_p<0.05))
    print('phi_p',np.where(phi_p<0.05))

    left = np.arange(10)
    width = 0.3

    g1=plt.bar(left,nn_corr,color='r',label='nn_corr',width=width,align='center')
    g2=plt.bar(left+width, phi_corr, color='b',label='phi_corr', width=width, align='center')
    plt.legend(handles=[g1,g2],loc='best',shadow=True)
    plt.xlabel('set num')
    plt.ylabel('correlation')
    plt.title("no."+str(name_num) +"-"+ str(set_num))
    plt.show()

'''
for i in range(3):
  print('nn_num',nn_num[i,:])
  print('phi_num',phi_num[i])

  plt.figure(figsize=(8,4))

  left = np.arange(9)
  width = 0.3

  g1=plt.bar(left,phi_num[i,:],color='r',label='proposed method',width=width,align='center',tick_label=range(1,10))
  g2=plt.bar(left+width, nn_num[i,:], color='none',edgecolor='b',hatch='//////////',label='neural network', width=width, align='center',tick_label=range(1,10))
  plt.legend(handles=[g1,g2],loc='best',shadow=True,fontsize=10)
  plt.xlabel('user id',fontsize=12)
  plt.ylabel('the number of correlation\n larger than 0.2',fontsize=12)
  plt.ylim(0,7.3)
  #plt.title(str((i+1)*10))

#plt.ylabel('count of up/keep')

  plt.savefig(str(i)+'times_compare.eps')
  #plt.show()
'''
