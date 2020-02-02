import numpy as np
import scipy as sp
from scipy.stats import pearsonr

import matplotlib.pyplot as plt


nn_ave= np.zeros((6,9))
phi_ave= np.zeros((6,9))

nn_gosa = np.zeros((6,9))
phi_gosa = np.zeros((6,9))

sum_ave = 0
sum_ave2 = 0
sum_ave3 = 0

for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)

  for set_num in (5,10,15,20,25,30):

    nn_gosa = np.loadtxt(dir_name + "/nn_gosa-" + str(set_num) + ".csv", delimiter=",")

    phi_gosa = np.loadtxt(dir_name + "/phi_gosa-" + str(set_num) + ".csv", delimiter=",")

    nn_ave[set_num/5-1,name_num]= np.average(abs(nn_gosa))
    val = (phi_gosa[~np.isinf(phi_gosa)])
    phi_ave[set_num/5-1,name_num]= np.average(abs(val[~np.isnan(val)]))

    #arr = phi_corr[~np.isnan(phi_corr)]

    #nn_num[set_num/5-1,name_num] = np.count_nonzero((nn_corr)>0.2)
    #phi_num[set_num/5-1,name_num] = np.count_nonzero((phi_corr)>0.2)

    left = np.arange(30)
    width = 0.3

    #g1=plt.bar(left,(nn_corr),color='b',label='nn_corr',width=width,align='center')
    #g2=plt.bar(left+width, (phi_corr), color='r',label='phi_corr', width=width, align='center')
    #plt.legend(handles=[g1,g2],loc='best',shadow=True)
    #plt.xlabel('set num')
    #plt.ylabel('correlation')
    #plt.ylim(0.2,1)
    #plt.title("no."+str(name_num) +"-"+ str(set_num)+"\n" + vsjudge+"\n" +vsave)
    #plt.show()

pave = np.zeros(6)
nave = np.zeros(6)

plt.figure(figsize=(4,8))

for i in range(6):
  val = phi_ave[i,:]
  print "ave",val
  val2=val[~np.isinf(val)]
  pave[i] = np.average(val[~np.isnan(val2)])
  print "i,pave",i,pave[i]
  val = nn_ave[i,:]*10
  val2=val[~np.isinf(val)]
  nave[i] = np.average(val[~np.isnan(val2)])


g1=plt.plot((5,10,15,20,25,30),pave,color='r',label='proposed method',marker='o',linestyle='dashdot')
g2=plt.plot((5,10,15,20,25,30),nave,color='b',label='nn',marker='o',linestyle='dashed')
plt.xlabel('number of training data',fontsize=15)
plt.ylabel('average of correlation',fontsize=15)
plt.legend()
#plt.ylim(0,1)
plt.show()
#plt.savefig('ave_matome.eps')

#pcorr = np.zeros(6)
#ncorr = np.zeros(6)

#plt.figure(figsize=(4,8))

#for i in range(6):

#  phi_num[set_num/5-1,name_num] = np.count_nonzero((phi_corr)>0.2)
#  val2 = phi_num[i,:]
#  print "npcount",np.count_nonzero(val2[~np.isnan(val2)])
#  pcorr[i] = np.sum(val2)
#
#  val2 = nn_num[i,:]
#  ncorr[i] = np.sum(val2)

#g1=plt.plot((5,10,15,20,25,30),pcorr,color='r',marker='o',label='proposed method',linestyle='dash#dot')
#g2=plt.plot((5,10,15,20,25,30),ncorr,color='b',marker='o',label='nn',linestyle='dashed')
#plt.xlabel('number of training data',fontsize=15)
#plt.ylabel('number of correlation larger than 0.2',fontsize=15)

#plt.legend()
#plt.savefig('corr_matome.eps')
#plt.show()

'''
for i in range(6):
  #print('nn_num',nn_num[i,:])
  print('phi_num',phi_ave[i])

  plt.figure(figsize=(8,4))

  left = np.arange(9)
  width = 0.3

  g1=plt.bar(left,phi_ave[i,:],color='pink',label='proposed method',width=width,align='center',tick_label=range(1,10))
  #g2=plt.bar(left+width, nn_ave[i,:], color='none',edgecolor='skyblue',hatch='//////////',label='neural network', width=width, align='center',tick_label=range(1,10))
  #plt.legend(handles=[g1,g2],loc='best',shadow=True,fontsize=10)
  plt.xlabel('user id',fontsize=12)
  plt.ylabel('the average of correlation',fontsize=12)
  #plt.ylabel('the number of correlation\n larger than 0.2',fontsize=12)
  #plt.ylim(0,7.3)
  #plt.title(str((i+1)*10))

  #plt.ylabel('count of up/keep')

  #plt.savefig(str(i)+'times_compare.eps')
  plt.show()
'''
