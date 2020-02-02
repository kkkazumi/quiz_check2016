import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr


test_len = 30

hito_num = 9

set_list = (5, 10, 15, 20, 25, 30,35,40)

nn_num = np.zeros((len(set_list),hito_num))
phi_num = np.zeros((len(set_list),hito_num))

nn_ave = np.zeros((len(set_list),hito_num))
phi_ave = np.zeros((len(set_list),hito_num))

nn_all = np.zeros((len(set_list),hito_num,test_len))
phi_all = np.zeros((len(set_list),hito_num,test_len))

phi_minmax = np.zeros((2,len(set_list),hito_num))
print phi_minmax

sum_ave = 0
sum_ave2 = 0
sum_ave3 = 0

for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)
  #dir_name2 = "./jrm_test/third_test/" + str(name_num+1)

  for set_num in set_list:

    nn_corr = np.loadtxt(dir_name + "/nn_diff-" + str(set_num) + ".csv", delimiter=",")
    #nn_p = np.loadtxt(dir_name + "/nn_rimp-" + str(set_num) + ".csv")

    phi_corr = np.loadtxt(dir_name + "/phi_diff-" + str(set_num) + ".csv", delimiter=",")
    #phi_p = np.loadtxt(dir_name + "/phi_rimp-" + str(set_num) + ".csv")

    if len(phi_corr[~np.isnan(phi_corr)])>20:
      print len(phi_corr[~np.isnan(phi_corr)]),name_num,"-",set_num,len(phi_corr[phi_corr>0])

    nn_num[set_num/5-1,name_num] = np.count_nonzero(nn_corr<=0)

    phi_num[set_num/5-1,name_num] = len(phi_corr[np.isnan(phi_corr)])

    #phi_corr[np.isnan(phi_corr)] = 0 
    phi_corr[np.isinf(phi_corr)] = None 

    nn_corr[np.isnan(phi_corr)] = None 
    nn_corr[np.isinf(phi_corr)] = None 

    nn_all[set_num/5-1,name_num,:]= nn_corr
    phi_all[set_num/5-1,name_num,:]= phi_corr

    arr = phi_corr[~np.isnan(phi_corr)]

    nn_ave[set_num/5-1,name_num]= np.average((nn_corr[nn_corr>0]))
    phi_ave[set_num/5-1,name_num]= np.average((phi_corr[phi_corr>0]))

    #nn_ave[set_num/5-1,name_num]= np.average(nn_corr[~np.isnan(nn_corr)])
    #phi_ave[set_num/5-1,name_num]= np.average(phi_corr[~np.isnan(phi_corr)])

    vsjudge = 'count' + str(np.count_nonzero(abs(nn_corr)>0.2)) + "vs" + str(np.count_nonzero(abs(phi_corr)>0.2))
    vsave = 'ave'+str((nn_ave[set_num/5-1,name_num])) + "vs" + str((phi_ave[set_num/5-1,name_num]))

    left = np.arange(30)
    width = 0.3

pave = np.zeros(len(set_list))
perr = np.zeros((2,len(set_list)))
nave = np.zeros(len(set_list))
nerr = np.zeros((2,len(set_list)))


for i in range(len(set_list)):
  val = phi_all[i,:,:]
  #pave[i] = np.var(val[~np.isnan(val)])
  pave[i] = iqr(val[~np.isnan(val)])
  perr[1,i] = abs(np.max(val[~np.isnan(val)])-pave[i])
  perr[0,i] = abs(pave[i]-np.min(val[~np.isnan(val)]))

  val = nn_all[i,:,:]
  #nave[i] = np.var(val[~np.isnan(val)])
  nave[i] = iqr(val[~np.isnan(val)])
  #nerr[1,i] = abs(np.max(val[~np.isnan(val)])-nave[i])
  #nerr[0,i] = abs(nave[i]-np.min(val[~np.isnan(val)]))

  nerr[1,i] = abs(np.max(nn_ave[i,:])-nave[i])
  nerr[0,i] = abs(nave[i]-np.min(nn_ave[i,:]))
  print "val",val
  print "ave",nave[i]


columns = ['phi','nn']
index = np.zeros((len(set_list),1))
index[:,0] = np.linspace(5,40,len(set_list),dtype='int')

df_phi = pd.concat([df_phi_in,df_phi_way,df_phi_data],axis=1)
print df_phi

df_phi_in = pd.DataFrame(data=index,columns=["the number of supervisor data"],dtype='int')
df_nn_way = pd.DataFrame({'way':['neural network','neural network','neural network','neural network','neural network','neural network','neural network','neural network']},columns=["way"],dtype='object',index=[0,1,2,3,4,5,6,7])
df_nn_data = pd.DataFrame(data=nn_ave,columns=["0","1","2","3","4","5","6","7","8"],dtype='float',index=[0,1,2,3,4,5,6,7])

df_nn = pd.concat([df_phi_in,df_nn_way,df_nn_data],axis=1)
print df_nn


phi_melt = pd.melt(df_phi, id_vars=['the number of supervisor data','way'],var_name='hito')
nn_melt = pd.melt(df_nn, id_vars=['the number of supervisor data','way'],var_name='hito')

data = pd.concat([nn_melt,phi_melt])
data_new = data.rename(columns={'value':'correlation between estimated mood and self-assessed mood'})


plt.figure(figsize=(8,6))
#sns.violinplot(x='number',y='value',hue='way',data=data,split=True,inner="stick",scale_hue=False,bw=.5)
#sns.swarmplot(x='number',y='value',hue='way',data=data,split=True)
#sns.despine(offset=10,trim=True)
sns.set(style='whitegrid',palette='bright')
sns.set_context(font_scale=10)
sns_plot = sns.boxplot(x='the number of supervisor data',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new)
#sns_plot = sns.pointplot(x='the number of supervisor data',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new,
#  markers=['o','x'],linestyles=["-",'--'],capsize=.2,dodge=True,ci="sd")
fig = sns_plot.get_figure()
#fig.legend()
#plt.show()
fig.savefig('var_ave_box.eps')

plt.figure(figsize=(8,6))
#box
#g1=plt.errorbar(set_list,pave,yerr = 0.1)
g1=plt.plot(set_list,pave,color='darkorange',label='proposed method',marker='x',linestyle='dashed',linewidth=3,markeredgewidth=3,markersize=10)
g2=plt.plot(set_list,nave,color='b',label='neural network',marker='o',linewidth=3,markersize=10)

print('pave',pave)
print('nave',nave)

#error bar
#g1=plt.errorbar(set_list,pave,yerr = perr, color='r',label='proposed method',marker='o',elinewidth=1,linestyle='dashdot',capsize=4)
#g2=plt.errorbar(set_list,nave,yerr=nerr,color='b',label='nn',marker='o',elinewidth=1,linestyle='dashed',capsize=4)

plt.xlabel('the number of training data',fontsize=15)
plt.ylabel('interquartile range of correlation',fontsize=15)
plt.legend()
plt.show()
#plt.savefig('ave_matome.eps')

pcorr = np.zeros(len(set_list))
ncorr = np.zeros(len(set_list))

fig

#plt.figure(figsize=(4,8))

for i in range(len(set_list)):

  #phi_num[set_num/5-1,name_num] = np.count_nonzero((phi_corr)>0.2)
  val2 = phi_num[i,:]
  print "npcount",np.count_nonzero(val2[~np.isnan(val2)])
  pcorr[i] = np.average(val2)

  val2 = nn_num[i,:]
  ncorr[i] = np.average(val2)

g1=plt.plot(set_list,pcorr,color='r',marker='o',label='proposed method',linestyle='dashdot')
g2=plt.plot(set_list,ncorr,color='b',marker='o',label='nn',linestyle='dashed')
plt.xlabel('the number of supervisor data',fontsize=15)
plt.ylabel('average of correlation (positive)',fontsize=15)

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
