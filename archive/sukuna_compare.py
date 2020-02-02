import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


hito_num = 9

set_list = (5, 10, 15, 20, 25, 30)

nn_num = np.zeros((len(set_list),hito_num))
phi_num = np.zeros((len(set_list),hito_num))

nn_ave = np.zeros((len(set_list),hito_num))
phi_ave = np.zeros((len(set_list),hito_num))

phi_minmax = np.zeros((2,len(set_list),hito_num))
print phi_minmax

sum_ave = 0
sum_ave2 = 0
sum_ave3 = 0

for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)

  for set_num in set_list:

    nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
    nn_p = np.loadtxt(dir_name + "/nn_p-" + str(set_num) + ".csv")

    phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")
    phi_p = np.loadtxt(dir_name + "/phi_p-" + str(set_num) + ".csv")

    #print(name_num,set_num,"---nn:",np.average(nn_corr[~np.isnan(nn_corr)]))
    #print(name_num,set_num,"--phi:",np.average(phi_corr[~np.isnan(phi_corr)]))

    #nn_ave[set_num/5-1,name_num]= np.average((nn_corr[nn_corr>0]))
    #phi_ave[set_num/5-1,name_num]= np.average((phi_corr[phi_corr>0]))
    nn_ave[set_num/5-2,name_num]= np.average(abs(nn_corr[~np.isnan(nn_corr)]))
    phi_ave[set_num/5-1,name_num]= np.average(abs(phi_corr[~np.isnan(phi_corr)]))

    #nn_ave[set_num/5-1,name_num]= np.var((nn_corr[nn_corr>0]))
    #phi_ave[set_num/5-1,name_num]= np.var((phi_corr[phi_corr>0]))

    arr = phi_corr[~np.isnan(phi_corr)]

    nn_num[set_num/5-1,name_num] = np.count_nonzero((nn_corr)>0.5)
    phi_num[set_num/5-1,name_num] = np.count_nonzero((phi_corr)>0.5)
    #print(nn_num[set_num/5-2,name_num])

    #print(np.count_nonzero(nn_corr>0),np.count_nonzero(phi_corr>0))

    vsjudge = 'count' + str(np.count_nonzero(abs(nn_corr)>0.2)) + "vs" + str(np.count_nonzero(abs(phi_corr)>0.2))
    vsave = 'ave'+str((nn_ave[set_num/5-2,name_num])) + "vs" + str((phi_ave[set_num/5-2,name_num]))
    #print set_num/5-1, phi_ave[set_num/5-1,name_num],phi_num[set_num/5-1,name_num]

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

pave = np.zeros(len(set_list))
perr = np.zeros((2,len(set_list)))
nave = np.zeros(len(set_list))
nerr = np.zeros((2,len(set_list)))

plt.figure(figsize=(4,8))

for i in range(len(set_list)):
  val = phi_ave[i,:]
  pave[i] = np.average(val[~np.isnan(val)])
  perr[1,i] = abs(np.max(val[~np.isnan(val)])-pave[i])
  perr[0,i] = abs(pave[i]-np.min(val[~np.isnan(val)]))

  val = nn_ave[i,:]
  nave[i] = np.average(val[~np.isnan(val)])
  #nerr[1,i] = abs(np.max(val[~np.isnan(val)])-nave[i])
  #nerr[0,i] = abs(nave[i]-np.min(val[~np.isnan(val)]))
  nerr[1,i] = abs(np.max(nn_ave[i,:])-nave[i])
  nerr[0,i] = abs(nave[i]-np.min(nn_ave[i,:]))
  print "val",val
  print "ave",nave[i]


#plt.errorbar(x, y, yerr = y_err, xerr = x_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')

#g1=plt.plot(set_list,pave,color='r',label='proposed method',marker='o',linestyle='dashdot')
#g1 = plt.boxplot(phi_ave.T)
#g2 = plt.boxplot(nn_ave.T)

phi_arr=(phi_ave[0,~np.isnan(phi_ave[0,:])],
  phi_ave[1,~np.isnan(phi_ave[1,:])],
  phi_ave[2,~np.isnan(phi_ave[2,:])],
  phi_ave[3,~np.isnan(phi_ave[3,:])],
  phi_ave[4,~np.isnan(phi_ave[4,:])],
  phi_ave[5,~np.isnan(phi_ave[5,:])])


nn_arr=(nn_ave[0,~np.isnan(nn_ave[0,:])],
  nn_ave[1,~np.isnan(nn_ave[1,:])],
  nn_ave[2,~np.isnan(nn_ave[2,:])],
  nn_ave[3,~np.isnan(nn_ave[3,:])],
  nn_ave[4,~np.isnan(nn_ave[4,:])],
  nn_ave[5,~np.isnan(nn_ave[5,:])])

columns = ['phi','nn']
index = np.zeros((6,1))
index[:,0] = np.linspace(0,5,6)
phi_ways = np.ones((6,1))*99
data_phi = np.hstack((index,phi_ways,phi_ave))


nn_ways = np.ones((6,1))*88
data_nn = np.hstack((index,nn_ways,nn_ave))

df_phi = pd.DataFrame(data=data_phi,columns=["number","way","0","1","2","3","4","5","6","7","8"],dtype='float')
df_nn = pd.DataFrame(data=data_nn,columns=["number","way","0","1","2","3","4","5","6","7","8"],dtype='float')

phi_melt = pd.melt(df_phi, id_vars=['number','way'],var_name='hito')
nn_melt = pd.melt(df_nn, id_vars=['number','way'],var_name='hito')

data = pd.concat([phi_melt,nn_melt])

data['way']=data['way'].astype(str)

data['way'].str.replace('99.0','phi')
data['way'].str.replace('88.0','nn')

print data['way']
print data['way'].dtype


#sns.violinplot(x='number',y='value',hue='way',data=data,split=True,inner="stick",scale_hue=False,bw=.5)
#sns.despine(offset=10,trim=True)
sns.boxplot(x='number',y='value',hue='way',data=data)

posi = np.linspace(0,6,6)
posi2 = np.linspace(0,6,6) + np.ones(6)*0.5

#g1 = plt.violinplot(phi_arr,posi,showmeans=True)
#g2 = plt.violinplot(nn_arr,posi2,showmeans=True)
#g1 = plt.boxplot(arr1)
#g2 = plt.boxplot(arr2)

#box
#g1=plt.errorbar(set_list,pave,yerr = 0.1)
plt.figure(figsize=(4,8))
g1=plt.plot(set_list,pave,color='r',label='phi',marker='o',linestyle='dashed')
g2=plt.plot(set_list,nave,color='b',label='nn',marker='o',linestyle='dashed')


#error bar
#g1=plt.errorbar(set_list,pave,yerr = perr, color='r',label='proposed method',marker='o',elinewidth=1,linestyle='dashdot',capsize=4)
#g2=plt.errorbar(set_list,nave,yerr=nerr,color='b',label='nn',marker='o',elinewidth=1,linestyle='dashed',capsize=4)
plt.xlabel('number of training data',fontsize=15)
plt.ylabel('average of correlation',fontsize=15)
plt.legend()
plt.show()
#plt.savefig('ave_matome.eps')

pcorr = np.zeros(len(set_list))
ncorr = np.zeros(len(set_list))

plt.figure(figsize=(4,8))

for i in range(len(set_list)):

  phi_num[set_num/5-1,name_num] = np.count_nonzero((phi_corr)>0.2)
  val2 = phi_num[i,:]
  print "npcount",np.count_nonzero(val2[~np.isnan(val2)])
  pcorr[i] = np.sum(val2)

  val2 = nn_num[i,:]
  ncorr[i] = np.sum(val2)

g1=plt.plot(set_list,pcorr,color='r',marker='o',label='proposed method',linestyle='dashdot')
g2=plt.plot(set_list,ncorr,color='b',marker='o',label='nn',linestyle='dashed')
plt.xlabel('number of training data',fontsize=15)
plt.ylabel('number of correlation larger than 0.2',fontsize=15)

plt.legend()
#plt.savefig('corr_matome.eps')
plt.show()

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
