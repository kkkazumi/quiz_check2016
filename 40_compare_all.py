import numpy as np
import scipy as sp
from scipy.stats import pearsonr
from scipy.stats import wilcoxon
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import stats

print("input abs or plus")
flag = raw_input()

test_len = 25

hito_num = 9

set_list = (5, 10, 15, 20, 25)

nn_ave = np.zeros((len(set_list)*test_len,hito_num))
phi_ave = np.zeros((len(set_list)*test_len,hito_num))

nn_all = np.zeros((len(set_list),hito_num,test_len))
phi_all = np.zeros((len(set_list),hito_num,test_len))

phi_minmax = np.zeros((2,len(set_list),hito_num))

sum_ave = 0
sum_ave2 = 0
sum_ave3 = 0

for name_num in range(hito_num):
  dir_name = "./jrm_test/jul_test2/" + str(name_num+1)

  for set_num in set_list:

    nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
    phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")

    #print(phi_corr[np.isinf(nn_corr)])
    #print(nn_corr[np.isinf(nn_corr)])
    phi_corr[np.isinf(nn_corr)] = 0
    nn_corr[np.isnan(nn_corr)] = 0

    nn_all[set_num/5-1,name_num,:]= nn_corr
    phi_all[set_num/5-1,name_num,:]= phi_corr

    if flag == "abs":

      for t in range(test_len):
        nn_ave[(set_num/5-1)*test_len:(set_num/5-1)*test_len+test_len,name_num]= np.abs(nn_corr)
        phi_ave[(set_num/5-1)*test_len:(set_num/5-1)*test_len+test_len,name_num]= np.abs(phi_corr)

      #abs
      #nn_ave[set_num/5-1,name_num]= np.average(abs(nn_corr))
      #phi_ave[set_num/5-1,name_num]= np.average(abs(phi_corr))
      str_label = 'the average of the absolute value of the correlation \n between estimated mood and self-assessed mood'

    elif flag == "plus":
      #plus

      for t in range(test_len):
        nn_ave[(set_num/5-1)*test_len:(set_num/5-1)*test_len+test_len,name_num]= nn_corr
        phi_ave[(set_num/5-1)*test_len:(set_num/5-1)*test_len+test_len,name_num]= phi_corr

      #nn_ave[set_num/5-1,name_num]= np.average(nn_corr[nn_corr>0])
      #phi_ave[set_num/5-1,name_num]= np.average(phi_corr[phi_corr>0])
      str_label = 'the average of the positive correlation \n between estimated mood and self-assessed mood'

    elif flag == "num":
      #plus
      nn_ave[set_num/5-1,name_num]= nn_corr[nn_corr>0].shape[0]
      phi_ave[set_num/5-1,name_num]= phi_corr[phi_corr>0].shape[0]
      str_label = 'the number of results in positive correlation'

    elif flag == "normal":
      #plus
      nn_ave[set_num/5-1,name_num]= np.average(nn_corr)
      phi_ave[set_num/5-1,name_num]= np.average(phi_corr)
      str_label = 'the average of the correlation \n between estimated mood and self-assessed mood'

    left = np.arange(test_len)
    width = 0.3

zero_wil   = nn_all[1,:,:]
for set_num in (5,15,20,25):
  phi_wil  = nn_all[set_num/5-1,:,:]
  print('set_num',set_num)
  print('wilcoxon check',wilcoxon(zero_wil.reshape(-1),phi_wil.reshape(-1)))

for i in range(len(set_list)):
  val = phi_all[i,:,:]
  val = nn_all[i,:,:]

columns = ['phi','nn']
index = np.zeros((len(set_list),1))
index[:,0] = np.linspace(5,25,len(set_list),dtype='int')

for setnum in range(4):
  for hitonum in range(9):
    nn_pdall=pd.DataFrame(nn_all[setnum,:,hitonum])
    phi_pdall=pd.DataFrame(phi_all[setnum,:,hitonum])
    #print(nn_pdall)
    #raw_input()

df_phi_in = pd.DataFrame(data=index,columns=["the number of supervisor data"],dtype='int')
df_nn_way = pd.DataFrame({'way':['neural network','neural network','neural network','neural network','neural network']},columns=["way"],dtype='object',index=[0,1,2,3,4])
df_phi_way = pd.DataFrame({'way':['proposed method','proposed method','proposed method','proposed method','proposed method']},columns=["way"],dtype='object',index=[0,1,2,3,4])

index_list = []
for i in range(5):
  for t in range(test_len):
    index_list.append(i)
print(index_list)

df_nn_data = pd.DataFrame(data=nn_ave,columns=["0","1","2","3","4","5","6","7","8"],dtype='float',index=index_list)
df_phi_data = pd.DataFrame(data=phi_ave,columns=["0","1","2","3","4","5","6","7","8"],dtype='float',index=index_list)

df_nn = pd.concat([df_phi_in,df_nn_way,df_nn_data],axis=1)
print("df_nn",df_nn)

df_phi = pd.concat([df_phi_in,df_phi_way,df_phi_data],axis=1)
print("df_phi",df_phi)

for t in range(5):
  _a=df_nn[df_nn['the number of supervisor data']==(t+1)*5]
  _b=df_phi[df_phi['the number of supervisor data']==(t+1)*5]

  a= pd.concat(_a[str(i)] for i in range(9))
  b= pd.concat(_b[str(i)] for i in range(9))

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

phi_melt = pd.melt(df_phi, id_vars=['the number of supervisor data','way'],var_name='hito')
nn_melt = pd.melt(df_nn, id_vars=['the number of supervisor data','way'],var_name='hito')

data = pd.concat([nn_melt,phi_melt])
data_new = data.rename(columns={'value':str_label})
#data_new = data.rename(columns={'value':'correlation between estimated mood and self-assessed mood'})

plt.figure(figsize=(8,6))
sns.set(style='whitegrid',palette='bright')
sns.set_context(font_scale=10)
#sns_plot = sns.boxplot(x='the number of supervisor data',y=str_label,hue='way',data=data_new)
sns_plot = sns.pointplot(x='the number of supervisor data',y=str_label,hue='way',data=data_new,capsize=.2)

#sns_plot = sns.pointplot(x='the number of supervisor data',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new,capsize=.2)

#plt.ylim([0,0.3])

fig = sns_plot.get_figure()
plt.show()
#plt.savefig('5-30_compare_all_'+flag+'.eps')
