import numpy as np
import scipy as sp
from scipy.stats import pearsonr
from scipy.stats import wilcoxon
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

flatui = ["#9b59b6","#3498db"]
graph_label = "estimation method"

test_len = 25

hito_num = 9

nn_all = np.zeros((test_len,hito_num))
phi_all = np.zeros((test_len,hito_num))

for name_num in range(hito_num):
  dir_name = "./jrm_test/jul_test2/" + str(name_num+1)
  #dir_name = "./jrm_test/third_test/" + str(name_num+1)

  set_num = 25

  nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
  phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")

  phi_corr[np.isnan(phi_corr)] = None #zero to get corr larger than 0.
  nn_corr[np.isnan(nn_corr)] = None

  #print('nn-average: ',np.average(nn_corr[nn_corr>0]))
  #print('phi-average: ',np.average(phi_corr[phi_corr>0]))

  #print("only positive-nn",nn_corr[nn_corr>0])
  #print("only positive-phi",phi_corr[phi_corr>0])

  #print("np.where positive-nn",np.where(nn_corr>0))
  #factors = np.loadtxt(dir_name + "/factor_test" + str(set_num) + ".csv", delimiter=",")
  #print("np.where positive-phi",np.where(phi_corr>0))
  #raw_input()


  #set all
  nn_all[:,name_num]= nn_corr
  phi_all[:,name_num]= phi_corr

  #plut
  #nn_all[:,name_num]= np.average(nn_corr[nn_corr>0])
  #phi_all[:,name_num]= np.average(phi_corr[phi_corr>0])

  #output corr
  #print('wilcoxon check',wilcoxon(nn_corr,phi_corr))

  #abs
  #nn_all[:,name_num]= np.average(abs(nn_corr))
  #phi_all[:,name_num]= np.average(abs(phi_corr))

  #all
  #nn_all[:,name_num]= np.average(nn_corr)
  #phi_all[:,name_num]= np.average(phi_corr)

#print(nn_all)

columns = ['phi','nn']
index = np.zeros((int(test_len),1))
index[:,0] = np.linspace(0,test_len-1,int(test_len),dtype='int')

df_phi_in = pd.DataFrame(data=index,columns=["test no"],dtype='int')
df_phi_way = pd.DataFrame({graph_label:"proposed method"},columns=[graph_label],dtype='object',index=np.linspace(0,test_len-1,int(test_len),dtype='int'))
df_phi_data = pd.DataFrame(data=phi_all,dtype='float',columns=["1","2","3","4","5","6","7","8","9"])
df_phi = pd.concat([df_phi_way,df_phi_data],axis=1)

df_nn_in = pd.DataFrame(data=index,columns=["test no"],dtype='int')
df_nn_way = pd.DataFrame({graph_label:"neural network"},columns=[graph_label],dtype='object',index=np.linspace(0,test_len-1,int(test_len),dtype='int'))
df_nn_data = pd.DataFrame(data=nn_all,dtype='float',columns=["1","2","3","4","5","6","7","8","9"])
df_nn = pd.concat([df_nn_way,df_nn_data],axis=1)

for i in ["1","2","3","4","5","6","7","8","9"]:
  a=df_nn[i]
  b=df_phi[i]
  if((a.isnull().any())or(b.isnull().any())):
    a2=a[a.notnull()]
    b2=b[b.notnull()]
    print(i,stats.ttest_ind(a2,b2),len(a2),len(b2))
  if(a.isnull().any()):
    a2=a[a.notnull()]
    b2=b[a.notnull()]
    if(b2.isnull().any()):
      a3=a2[b2.notnull()]
      b3=b2[b2.notnull()]
      a=a3
      b=b3
    else:
      a=a2
      b=b2
  print(i,stats.ttest_rel(a,b),len(a))
  input()

print("abs average of corr")
print("nn",df_nn_data.abs().mean())
print("phi",df_phi_data.abs().mean())

phi_melt = pd.melt(df_phi, id_vars=[graph_label])
nn_melt = pd.melt(df_nn, id_vars=[graph_label])


data= pd.concat([nn_melt,phi_melt])
data2= data.rename(columns={'variable':'user id'})
data_new = data2.rename(columns={'value':'correlation between self-assessed mood and estimated mood'})

plt.figure(figsize=(16,12))
#sns.set_context(font_scale=20)
plt.rcParams["font.size"] = 18

#sns_plot = sns.barplot(x='user id',y='correlation between self-assessed mood and estimated mood',hue=graph_label,data=data_new)
plt.grid(color = "gray", linestyle="--",alpha=0.5)


sns_plot = sns.boxplot(x='user id',y='correlation between self-assessed mood and estimated mood',hue=graph_label,data=data_new)

#sns_plot = sns.barplot(x='user id',y='absolute value of errors between self-assessed mood and estimated mood',hue=graph_label,data=data_new,capsize=.2)
#sns_plot = sns.boxplot(x='user id',y='absolute value of errors between self-assessed mood and estimated mood',hue=graph_label,data=data_new)

#plt.legend()
plt.legend(bbox_to_anchor=(.75, 1.14), loc='upper left', borderaxespad=0, fontsize=18)
fig = sns_plot.get_figure()

plt.show()
#fig.savefig('userid30_compare-corrall.eps')
