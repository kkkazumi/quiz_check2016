import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr

print("input abs or plus")
flag = raw_input()



test_len = 25

hito_num = 9

set_list = (5, 10, 15, 20, 25)

nn_ave = np.zeros((len(set_list),hito_num))
phi_ave = np.zeros((len(set_list),hito_num))

nn_all = np.zeros((len(set_list),hito_num,test_len))
phi_all = np.zeros((len(set_list),hito_num,test_len))

phi_minmax = np.zeros((2,len(set_list),hito_num))

sum_ave = 0
sum_ave2 = 0
sum_ave3 = 0

for name_num in range(hito_num):
  dir_name = "./jrm_test/" + str(name_num+1)

  for set_num in set_list:

    nn_corr = np.loadtxt(dir_name + "/nn_sign-" + str(set_num) + ".csv", delimiter=",")
    phi_corr = np.loadtxt(dir_name + "/phi_sign-" + str(set_num) + ".csv", delimiter=",")
    #nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
    #phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")


    phi_corr[np.isinf(nn_corr)] = 0
    nn_corr[np.isnan(nn_corr)] = 0

    nn_all[set_num/5-1,name_num,:]= nn_corr
    phi_all[set_num/5-1,name_num,:]= phi_corr
    print('nn',nn_corr[nn_corr>0].shape)
    print('phi',phi_corr[phi_corr>0].shape)

    if flag == "abs":
      #abs
      nn_ave[set_num/5-1,name_num]= np.average(abs(nn_corr))
      phi_ave[set_num/5-1,name_num]= np.average(abs(phi_corr))
      str_label = 'the average of the absolute value of the correlation \n between estimated mood and self-assessed mood'

    elif flag == "plus":
      #plus
      nn_ave[set_num/5-1,name_num]= np.average(nn_corr[nn_corr>0])
      phi_ave[set_num/5-1,name_num]= np.average(phi_corr[phi_corr>0])
      str_label = 'the average of the positive correlation \n between estimated mood and self-assessed mood'

    elif flag == "num":
      #plus
      nn_ave[set_num/5-1,name_num]= nn_corr[nn_corr>0].shape[0]
      phi_ave[set_num/5-1,name_num]= phi_corr[phi_corr>0].shape[0]
      str_label = 'the number of results in positive correlation'



    left = np.arange(test_len)
    width = 0.3


for i in range(len(set_list)):
  val = phi_all[i,:,:]

  val = nn_all[i,:,:]

columns = ['phi','nn']
index = np.zeros((len(set_list),1))
index[:,0] = np.linspace(5,25,len(set_list),dtype='int')

df_phi_in = pd.DataFrame(data=index,columns=["the number of supervisor data"],dtype='int')
df_nn_way = pd.DataFrame({'way':['neural network','neural network','neural network','neural network','neural network']},columns=["way"],dtype='object',index=[0,1,2,3,4])
df_phi_way = pd.DataFrame({'way':['proposed method','proposed method','proposed method','proposed method','proposed method']},columns=["way"],dtype='object',index=[0,1,2,3,4])
df_nn_data = pd.DataFrame(data=nn_ave,columns=["0","1","2","3","4","5","6","7","8"],dtype='float',index=[0,1,2,3,4])
df_phi_data = pd.DataFrame(data=phi_ave,columns=["0","1","2","3","4","5","6","7","8"],dtype='float',index=[0,1,2,3,4])

df_nn = pd.concat([df_phi_in,df_nn_way,df_nn_data],axis=1)
print "df_nn",df_nn

df_phi = pd.concat([df_phi_in,df_phi_way,df_phi_data],axis=1)
print "df_phi",df_phi



phi_melt = pd.melt(df_phi, id_vars=['the number of supervisor data','way'],var_name='hito')
nn_melt = pd.melt(df_nn, id_vars=['the number of supervisor data','way'],var_name='hito')

data = pd.concat([nn_melt,phi_melt])
data_new = data.rename(columns={'value':str_label})
#data_new = data.rename(columns={'value':'correlation between estimated mood and self-assessed mood'})

plt.figure(figsize=(8,6))
sns.set(style='whitegrid',palette='bright')
sns.set_context(font_scale=10)
sns_plot = sns.pointplot(x='the number of supervisor data',y=str_label,hue='way',data=data_new,capsize=.2)
#sns_plot = sns.pointplot(x='the number of supervisor data',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new,capsize=.2)

#plt.ylim([0,0.3])

fig = sns_plot.get_figure()
plt.show()
#plt.savefig('5-40diff_matome_abs.eps')
