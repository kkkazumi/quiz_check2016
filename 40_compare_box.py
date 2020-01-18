import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr

flatui = ["#9b59b6","#3498db"]

test_len = 30

hito_num = 9

set_list = (5, 10, 15, 20, 25, 30,35,40)

nn_all = np.zeros((test_len,hito_num))
phi_all = np.zeros((test_len,hito_num))

for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)
  #dir_name2 = "./jrm_test/third_test/" + str(name_num+1)

  set_num = 40

  nn_corr = np.loadtxt(dir_name + "/nn_diff-" + str(set_num) + ".csv", delimiter=",")
  phi_corr = np.loadtxt(dir_name + "/phi_diff-" + str(set_num) + ".csv", delimiter=",")

  nn_all[:,name_num]= abs(nn_corr)
  phi_all[:,name_num]= abs(phi_corr)

columns = ['phi','nn']
index = np.zeros((30,1))
index[:,0] = np.linspace(0,29,30,dtype='int')

df_phi_in = pd.DataFrame(data=index,columns=["test no"],dtype='int')
df_phi_way = pd.DataFrame({'way':"proposed method"},columns=["way"],dtype='object',index=np.linspace(0,29,30,dtype='int'))
df_phi_data = pd.DataFrame(data=phi_all,dtype='float',columns=["1","2","3","4","5","6","7","8","9"])
df_phi = pd.concat([df_phi_way,df_phi_data],axis=1)

df_nn_in = pd.DataFrame(data=index,columns=["test no"],dtype='int')
df_nn_way = pd.DataFrame({'way':"neural network"},columns=["way"],dtype='object',index=np.linspace(0,29,30,dtype='int'))
df_nn_data = pd.DataFrame(data=nn_all,dtype='float',columns=["1","2","3","4","5","6","7","8","9"])
df_nn = pd.concat([df_nn_way,df_nn_data],axis=1)

phi_melt = pd.melt(df_phi, id_vars=['way'])
nn_melt = pd.melt(df_nn, id_vars=['way'])

data= pd.concat([nn_melt,phi_melt])
data2= data.rename(columns={'variable':'user id'})
data_new = data2.rename(columns={'value':'correlation between self-assessed mood and estimated mood'})

plt.figure(figsize=(8,6))
#sns.violinplot(x='number',y='value',hue='way',data=data,split=True,inner="stick",scale_hue=False,bw=.5)
#sns.swarmplot(x='number',y='value',hue='way',data=data,split=True)
#sns.despine(offset=10,trim=True)
#sns.palplot(sns.color_palette(flatui))
sns.set_context(font_scale=10)
sns_plot = sns.boxplot(x='user id',y='correlation between self-assessed mood and estimated mood',hue='way',data=data_new)
#sns_plot = sns.pointplot(x='the number of supervisor data',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new,
#  markers=['o','x'],linestyles=["-",'--'],capsize=.2,dodge=True,ci="sd")
fig = sns_plot.get_figure()
#fig.legend()

plt.show()
#fig.savefig('vsnn_box.eps')
