import numpy as np
import scipy as sp
from scipy.stats import pearsonr
from scipy.stats import wilcoxon
import pandas as pd

from conv_num import *

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def out_figure(arr,ylabel):
  plt.xticks([1,2,3,4,5,6,7,8,9])
  width=0.2
  array=np.array([1,2,3,4,5,6,7,8,9])
  plt.bar(array-width,arr[:,0],align="edge",width=-width,color="white", edgecolor='black',label="happy")
  plt.bar(array,arr[:,1],align="edge",width=-width,color="white", edgecolor='black',hatch="////",label="surprise")
  plt.bar(array,arr[:,2],align="edge",width=width,color="black",label="angry")
  plt.bar(array+width,arr[:,3],align="edge",width=width,color="white", edgecolor='black',hatch="-----",label="sad")
  plt.xlabel("User ID")
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

flatui = ["#9b59b6","#3498db"]
graph_label = "estimation method"

test_len = 25

hito_num = 9

nn_all = np.zeros((test_len,hito_num))
phi_all = np.zeros((test_len,hito_num))


NUM_MAX = 30
TEST_NUM = 30 #the number of test data is 10
face_num=4

var=np.zeros((hito_num,face_num))
mean=np.zeros((hito_num,face_num))

for i_name in username:
  #print(i_name)
  tag_number = conv_num(i_name)

  #set input, output data to train

  xdata_b = "./jiken/" + i_name + "/factor_before.csv"
  x2data_b = "./jiken/" + i_name + "/kibun_before.csv"
  ydata_b = "./jiken/" + i_name + "/signal_before.csv"

  xdata_a = "./jiken/" + i_name + "/factor_after.csv"
  x2data_a = "./jiken/" + i_name + "/kibun_after.csv"
  ydata_a = "./jiken/" + i_name + "/signal_after.csv"

  xdata_c = "./jiken/" + i_name + "/factor_after2.csv"
  x2data_c = "./jiken/" + i_name + "/kibun_after2.csv"
  ydata_c = "./jiken/" + i_name + "/signal_after2.csv"

  factor_b= np.loadtxt(xdata_b,delimiter='\t')
  mood_b = np.loadtxt(x2data_b,delimiter='\t')
  face_b= np.loadtxt(ydata_b,delimiter='\t')
  #x_train_b = np.hstack((x1_train_b,x2_train_b.reshape(-1,1)))

  factor_a = np.loadtxt(xdata_a,delimiter='\t')
  mood_a = np.loadtxt(x2data_a,delimiter='\t')
  face_a = np.loadtxt(ydata_a,delimiter='\t')
  #x_train_a = np.hstack((x1_train_a,x2_train_a.reshape(-1,1)))

  factor_c = np.loadtxt(xdata_c,delimiter='\t')
  mood_c = np.loadtxt(x2data_c,delimiter='\t')
  face_c = np.loadtxt(ydata_c,delimiter='\t')

  factor = np.vstack((factor_b,factor_a,factor_c))
  mood = np.hstack((mood_b,mood_a,mood_c))
  face = np.vstack((face_b,face_a,face_c))


  var[int(tag_number)-1,:]=np.var(face,axis=0)
  mean[int(tag_number)-1,:]=np.mean(face,axis=0)

  np.set_printoptions(precision=6, suppress=True)
  #print(tag_number,"var,",var[int(tag_number)-1,:],"mean,",mean[int(tag_number)-1,:])

emolabel=["hap","sup","ang","sad"]
data=pd.read_csv("nn_corr_25_absave_all.csv")
print(data['nn'],)
for emo in range(4):
  r_nn = pearsonr(data['nn'],var[:,emo])
  r_phi = pearsonr(data['phi'],var[:,emo])
  print(emolabel[emo],": var vs r_nn",r_nn,"r_phi",r_phi)
  r_nn = pearsonr(data['nn'],mean[:,emo])
  r_phi = pearsonr(data['phi'],mean[:,emo])
  print(emolabel[emo],": mean vs r_nn",r_nn,"r_phi",r_phi)


#out_figure(var,"var of facial expression")
#out_figure(mean,"average of facial expression")
