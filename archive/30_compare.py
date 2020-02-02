import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr

hito_num = 9

set_list = (25)

#nn_num = np.zeros((len(set_list),hito_num))
#phi_num = np.zeros((len(set_list),hito_num))

nn_ave = np.zeros(hito_num)
phi_ave = np.zeros(hito_num)

nn_max= np.zeros(hito_num)
phi_max = np.zeros(hito_num)

nn_min= np.zeros(hito_num)
phi_min = np.zeros(hito_num)


for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)

  #for set_num in set_list:
  set_num = 25

  #nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
  #nn_p = np.loadtxt(dir_name + "/nn_p-" + str(set_num) + ".csv")

  phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")
  phi_p = np.loadtxt(dir_name + "/phi_p-" + str(set_num) + ".csv")

  #nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
  #nn_p = np.loadtxt(dir_name + "/nn_p-" + str(set_num) + ".csv")

  #phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")
  #phi_p = np.loadtxt(dir_name + "/phi_p-" + str(set_num) + ".csv")
  
  #nn_corr[np.isnan(phi_corr)]=None

  #nn_ave[name_num]= np.average((nn_corr[~np.isnan(nn_corr)]))

  #nn_ave[name_num]= np.average((nn_corr[nn_corr>0]))
  phi_ave[name_num]= np.average((phi_corr[phi_corr>0]))

  #nn_ave[name_num]= np.average(abs(nn_corr[~np.isnan(phi_corr)]))
  phi_ave[name_num]= np.average(abs(phi_corr[~np.isnan(phi_corr)]))

  phi_ave[name_num]= np.average((phi_corr[~np.isnan(phi_corr)]))
  #nn_max[name_num]= np.max(abs(nn_corr[~np.isnan(nn_corr)]))
  #phi_max[name_num]= np.max(abs(phi_corr[~np.isnan(phi_corr)]))

  #nn_min[name_num]= np.min(abs(nn_corr[~np.isnan(nn_corr)]))
  phi_min[name_num]= np.min(abs(phi_corr[~np.isnan(phi_corr)]))

  left = np.arange(30)
  width = 0.3

  #g1=plt.bar(left,(nn_corr),color='b',label='nn_corr',width=width,align='center')
  #g2=plt.bar(left+width, (phi_corr), color='r',label='phi_corr', width=width, align='center')
  #plt.legend(handles=[g1,g2],loc='best',shadow=True)
  #plt.xlabel('set num')
  #plt.ylabel('correlation')
  #plt.ylim(0.2,1)
  #plt.title("no."+str(name_num) +"-"+ str(set_num)+"\n" + vsjudge+"\n" +vsave)
  #plt.title("no."+str(name_num) +"-"+ str(max(phi_corr)))
  #plt.show()

phi_err2=phi_max-phi_ave
phi_err1=phi_ave - phi_min

print np.average(phi_ave)
print np.average(nn_ave)

#err1=iqr(phi_ave)
#err2=iqr(nn_ave)
left = np.arange(9)
width = 0.3

fsize=15

plt.figure(figsize=(8,4.5))

g1=plt.bar(left,phi_ave,color='r',label='proposed method',width=width,align='center',tick_label=range(1,10))
#g2=plt.bar(left+width, nn_ave,capsize=5, color='none',edgecolor='blue',hatch='///////',label='neural network', width=width, align='center',tick_label=range(1,10))
#g3=plt.scatter(left,phi_max,color='r',marker='o')
#g4=plt.scatter(left+width,nn_max,color='b',marker='x')
#plt.legend(handles=[g1,g2],loc='best',shadow=True,fontsize=10)
plt.xlabel('user id',fontsize=fsize)
plt.ylabel('average of correlation',fontsize=fsize)
#plt.ylim(0,0.45)

#plt.savefig('vsnn.eps')
plt.show()

#for i in range(len(set_list)):
  #val = phi_ave[i,:]
  #pave[i] = np.average(val[~np.isnan(val)])
  #perr[1,i] = abs(np.max(val[~np.isnan(val)])-pave[i])
  #perr[0,i] = abs(pave[i]-np.min(val[~np.isnan(val)]))

  #val = nn_ave[i,:]
  #nave[i] = np.average(val[~np.isnan(val)])
  #nerr[1,i] = abs(np.max(val[~np.isnan(val)])-nave[i])
  #nerr[0,i] = abs(nave[i]-np.min(val[~np.isnan(val)]))
  #nerr[1,i] = abs(np.max(nn_ave[i,:])-nave[i])
  #nerr[0,i] = abs(nave[i]-np.min(nn_ave[i,:]))
  #print "val",val
  #print "ave",nave[i]


#plt.errorbar(x, y, yerr = y_err, xerr = x_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')

#g1=plt.plot(set_list,pave,color='r',label='proposed method',marker='o',linestyle='dashdot')
#g1 = plt.boxplot(phi_ave.T)
#g2 = plt.boxplot(nn_ave.T)

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
