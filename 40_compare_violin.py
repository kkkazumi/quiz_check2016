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

nn_ave = np.zeros((len(set_list),hito_num*test_len))
phi_ave = np.zeros((len(set_list),hito_num*test_len))

for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)

  for set_num in set_list:

    nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
    phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")

    #abs
    ave_index = name_num*test_len
    nn_ave[set_num/5-1,ave_index:ave_index+test_len]= nn_corr
    phi_ave[set_num/5-1,ave_index:ave_index+test_len]= phi_corr
    #plus
    #nn_ave[set_num/5-1,name_num]= nn_corr.flatten()
    #phi_ave[set_num/5-1,name_num]= phi_corr.flatten()
    #print("ave",phi_ave[set_num/5-1,name_num])
    #raw_input()


columns = ['phi','nn']
index = np.zeros((len(set_list),1))
index[:,0] = np.linspace(5,40,len(set_list),dtype='int')

df_phi_in = pd.DataFrame(data=index,columns=["the number of supervisor data"],dtype='int')
nnlabel_list = ['neural network']*8
philabel_list = ['proposed method']*8
df_nn_way = pd.DataFrame({'way':nnlabel_list},columns=["way"],dtype='object',index=[0,1,2,3,4,5,6,7])
df_phi_way = pd.DataFrame({'way':philabel_list},columns=["way"],dtype='object',index=[0,1,2,3,4,5,6,7])

datacolumns= [str(n) for n in range(int(hito_num*test_len))]
df_nn_data = pd.DataFrame(data=nn_ave,columns=datacolumns,dtype='float',index=[0,1,2,3,4,5,6,7])
df_phi_data = pd.DataFrame(data=phi_ave,columns=datacolumns,dtype='float',index=[0,1,2,3,4,5,6,7])

df_nn = pd.concat([df_phi_in,df_nn_way,df_nn_data],axis=1)
df_phi = pd.concat([df_phi_in,df_phi_way,df_phi_data],axis=1)

phi_melt = pd.melt(df_phi, id_vars=['the number of supervisor data','way'],var_name='hito')
nn_melt = pd.melt(df_nn, id_vars=['the number of supervisor data','way'],var_name='hito')

data = pd.concat([nn_melt,phi_melt])
data_new = data.rename(columns={'value':'correlation between estimated mood and self-assessed mood'})

plt.figure(figsize=(8,6))
sns.set(style='whitegrid',palette='bright')
sns.set_context(font_scale=10)
#sns_plot = sns.boxplot(x='the number of supervisor data',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new)
sns_plot = sns.violinplot(x='the number of supervisor data',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new,capsize=.2)

#plt.ylim([0,0.3])

fig = sns_plot.get_figure()
plt.show()
#plt.savefig('5-40diff_matome.eps')
