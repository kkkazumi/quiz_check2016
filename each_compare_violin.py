import numpy as np
import scipy as sp
from scipy.stats import pearsonr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr

test_len = 25
hito_num = 9

nn_ave = np.zeros((hito_num,test_len))
phi_ave = np.zeros((hito_num,test_len))

for set_num in (5,10,15,20,25):
  print('set_num',set_num)
  for name_num in range(hito_num):
    dir_name = "./jrm_test/" + str(name_num+1)


    nn_corr = np.loadtxt(dir_name + "/nn_sign-" + str(set_num) + ".csv", delimiter=",")
    phi_corr = np.loadtxt(dir_name + "/phi_sign-" + str(set_num) + ".csv", delimiter=",")
    #nn_corr = np.loadtxt(dir_name + "/nn_corr-" + str(set_num) + ".csv", delimiter=",")
    #phi_corr = np.loadtxt(dir_name + "/phi_corr-" + str(set_num) + ".csv", delimiter=",")


    nn_ave[name_num,:]= nn_corr#np.where(nn_corr>0,nn_corr,None)
    phi_ave[name_num,:]= phi_corr#np.where(phi_corr>0,phi_corr,None)
    plt.scatter(nn_corr,phi_corr,label=str(name_num),alpha=.1,color="blue")
  print(pearsonr(phi_ave.reshape(-1),nn_ave.reshape(-1)))
  plt.xlabel("nn_sign")
  plt.ylabel("phi_sign")
  plt.title(str(name_num))
  plt.show()
  plt.clf()

  columns = ['phi','nn']
#columns = ['phi','nn']
  index = np.zeros((hito_num,1))
  index[:,0] = np.linspace(0,hito_num,hito_num,dtype='int')

  df_phi_in = pd.DataFrame(data=index,columns=["user id"],dtype='int')

  nnlabel_list = ['neural network']*9
  philabel_list = ['proposed method']*9
#ranlabel_list = ['random']*5

  df_nn_way = pd.DataFrame({'way':nnlabel_list},columns=["way"],dtype='object',index=[0,1,2,3,4,5,6,7,8])
  df_phi_way = pd.DataFrame({'way':philabel_list},columns=["way"],dtype='object',index=[0,1,2,3,4,5,6,7,8])
#df_ran_way = pd.DataFrame({'way':ranlabel_list},columns=["way"],dtype='object',index=[0,1,2,3,4])

  datacolumns= [str(n) for n in range(test_len)]
  df_nn_data = pd.DataFrame(data=nn_ave,columns=datacolumns,dtype='float',index=[0,1,2,3,4,5,6,7,8])
  df_phi_data = pd.DataFrame(data=phi_ave,columns=datacolumns,dtype='float',index=[0,1,2,3,4,5,6,7,8])

  df_nn = pd.concat([df_phi_in,df_nn_way,df_nn_data],axis=1)
  df_phi = pd.concat([df_phi_in,df_phi_way,df_phi_data],axis=1)

  phi_melt = pd.melt(df_phi, id_vars=['user id','way'],var_name='hito')
  nn_melt = pd.melt(df_nn, id_vars=['user id','way'],var_name='hito')

  data = pd.concat([nn_melt,phi_melt])
  data_new = data.rename(columns={'value':'correlation between estimated mood and self-assessed mood'})

  plt.figure(figsize=(8,6))
  sns.set(style='whitegrid',palette='bright')
  sns.set_context(font_scale=10)
#sns_plot = sns.boxplot(x='user id',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new)
  sns_plot = sns.violinplot(x='user id',y='correlation between estimated mood and self-assessed mood',hue='way',data=data_new,capsize=.2)

#plt.ylim([0,0.3])

  fig = sns_plot.get_figure()
  plt.show()
#plt.savefig('5-40diff_matome.eps')
