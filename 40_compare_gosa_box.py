import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

flatui = ["#9b59b6","#3498db"]
graph_label = "estimation method"

test_len = 30

hito_num = 9

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
df_phi_way = pd.DataFrame({graph_label:"proposed method"},columns=[graph_label],dtype='object',index=np.linspace(0,29,30,dtype='int'))
df_phi_data = pd.DataFrame(data=phi_all,dtype='float',columns=["1","2","3","4","5","6","7","8","9"])
df_phi = pd.concat([df_phi_way,df_phi_data],axis=1)

df_nn_in = pd.DataFrame(data=index,columns=["test no"],dtype='int')
df_nn_way = pd.DataFrame({graph_label:"neural network"},columns=[graph_label],dtype='object',index=np.linspace(0,29,30,dtype='int'))
df_nn_data = pd.DataFrame(data=nn_all,dtype='float',columns=["1","2","3","4","5","6","7","8","9"])
df_nn = pd.concat([df_nn_way,df_nn_data],axis=1)

phi_melt = pd.melt(df_phi, id_vars=[graph_label])
nn_melt = pd.melt(df_nn, id_vars=[graph_label])



data= pd.concat([nn_melt,phi_melt])
data2= data.rename(columns={'variable':'user id'})
data_new = data2.rename(columns={'value':'absolute value of errors between self-assessed mood and estimated mood'})

plt.figure(figsize=(8,6))
sns.set_context(font_scale=10)
sns_plot = sns.barplot(x='user id',y='absolute value of errors between self-assessed mood and estimated mood',hue=graph_label,data=data_new,capsize=.2)
#sns_plot = sns.boxplot(x='user id',y='absolute value of errors between self-assessed mood and estimated mood',hue=graph_label,data=data_new)
fig = sns_plot.get_figure()

#plt.show()
fig.savefig('vsnn_gosa.eps')
