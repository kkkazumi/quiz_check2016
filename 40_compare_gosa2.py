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

#nn_num = np.zeros((len(set_list),hito_num))
#phi_num = np.zeros((len(set_list),hito_num))

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

  for set_num in set_list:

    nn_corr = np.loadtxt(dir_name + "/nn_diff-" + str(set_num) + ".csv", delimiter=",")

    phi_corr = np.loadtxt(dir_name + "/phi_diff-" + str(set_num) + ".csv", delimiter=",")

    nn_all[set_num/5-1,name_num,:]= nn_corr
    phi_all[set_num/5-1,name_num,:]= phi_corr

    nn_ave[set_num/5-1,name_num]= np.average((abs(nn_corr)))
    phi_ave[set_num/5-1,name_num]= np.average((abs(phi_corr)))

    left = np.arange(30)
    width = 0.3



for i in range(len(set_list)):
  val = phi_all[i,:,:]
  #pave[i] = np.var(val[~np.isnan(val)])
  #pave[i] = iqr(val)

  val = nn_all[i,:,:]
  #nave[i] = np.var(val[~np.isnan(val)])
  #nave[i] = iqr(val)

columns = ['phi','nn']
index = np.zeros((len(set_list),1))
index[:,0] = np.linspace(5,40,len(set_list),dtype='int')

df_phi_in = pd.DataFrame(data=index,columns=["the number of supervisor data"],dtype='int')
df_nn_way = pd.DataFrame({'way':['neural network','neural network','neural network','neural network','neural network','neural network','neural network','neural network']},columns=["way"],dtype='object',index=[0,1,2,3,4,5,6,7])
df_phi_way = pd.DataFrame({'way':['proposed method','proposed method','proposed method','proposed method','proposed method','proposed method','proposed method','proposed method']},columns=["way"],dtype='object',index=[0,1,2,3,4,5,6,7])
df_nn_data = pd.DataFrame(data=nn_ave,columns=["0","1","2","3","4","5","6","7","8"],dtype='float',index=[0,1,2,3,4,5,6,7])
df_phi_data = pd.DataFrame(data=phi_ave,columns=["0","1","2","3","4","5","6","7","8"],dtype='float',index=[0,1,2,3,4,5,6,7])

df_nn = pd.concat([df_phi_in,df_nn_way,df_nn_data],axis=1)
print df_nn

df_phi = pd.concat([df_phi_in,df_phi_way,df_phi_data],axis=1)
print df_phi


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
plt.show()
#fig.savefig('var_ave_box.eps')

#plt.figure(figsize=(8,6))
#box
#g1=plt.errorbar(set_list,pave,yerr = 0.1)
#g1=plt.plot(set_list,pave,color='darkorange',label='proposed method',marker='x',linestyle='dashed',linewidth=3,markeredgewidth=3,markersize=10)
#g2=plt.plot(set_list,nave,color='b',label='neural network',marker='o',linewidth=3,markersize=10)

#print('pave',pave)
#print('nave',nave)

#error bar

#plt.xlabel('the number of training data',fontsize=15)
#plt.ylabel('interquartile range of correlation',fontsize=15)
#plt.legend()