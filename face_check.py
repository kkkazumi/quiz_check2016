import matplotlib
import numpy as np
from func_new import *
import matplotlib.pyplot as plt
from conv_num import *

EMO_NUM = 4
SIT_NUM = 10
USR_NUM = 9

xlabel = ["hap","sup","ang","sad"]

func_num = [[0,1,2,3],[4,5,6,7],[8,9,10,11],
  [12,13,14,15],[16,17,18,19],[20,21,22,23],
  [24,25,26,27],[28,29,30,31],[32,33,34,35],[36,37,38,39]]

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

j=0
count_array=np.zeros((USR_NUM,4))
graph_label = ["0.1<r<0.2","0.2<r<0.3","0.3<r<0.4","0.4<r"]

def out_kitekan_correct(i_name, set_num, test_num):
  tag_number = inv_num(i_name)
  factor = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+str(tag_number)+"/factor_before.csv",delimiter="\t")
  kibun = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+str(tag_number)+"/kibun_before.csv",delimiter="\t")

  phi_out = np.zeros((kibun.shape[0],SIT_NUM,EMO_NUM))
  for data_num in range(kibun.shape[0]):
    for emo_num in range(EMO_NUM):
      for sit_num in range(SIT_NUM):
        ret = func(inv_norm(func_num[sit_num][emo_num],factor[data_num][sit_num]/100.0),kibun[data_num]*10.0+0.001,func_num[sit_num][emo_num])
        phi_out[data_num][sit_num][emo_num] = ret

  return phi_out

for i_name in range(USR_NUM):
  set_num = 29
  for test_num in range(29):
    tag_number = inv_num(i_name)
    #phi_out = np.zeros((set_num,SIT_NUM,EMO_NUM))
    corr_out = np.zeros((SIT_NUM,4))
    phi_out = out_kitekan_correct(i_name,set_num,test_num)
    signal = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+str(tag_number)+"/signal_before.csv",delimiter="\t")

    for emo_num in range(EMO_NUM):
      #array = signal - phi_out[:,sit_num,:]
      for sit_num in range(SIT_NUM):

        #print("i_name",str(i_name),"sit_num",str(sit_num),"emo_num",str(emo_num))

        #fig, ax1= plt.subplots()
        #ax2=ax1.twinx()

        corr = np.corrcoef(signal[:,emo_num],phi_out[:,sit_num,emo_num])
        corr_out[sit_num,emo_num]=corr[0,1]
        #print(corr[0,1])

    #input()
    count_array[i_name,0] = (np.sum(corr_out>=.1)- np.sum(corr_out>.2))/float(SIT_NUM*EMO_NUM) 
    count_array[i_name,1] = (np.sum(corr_out>=.2)- np.sum(corr_out>.3))/float(SIT_NUM*EMO_NUM) 
    count_array[i_name,2] = (np.sum(corr_out>=.3)- np.sum(corr_out>.4))/float(SIT_NUM*EMO_NUM) 
    count_array[i_name,3] = np.sum(corr_out>=.4)/float(SIT_NUM*EMO_NUM) 
    j+=1
    #plt.title(i_name)

    #np.savetxt('/home/kazumi/prog/quiz_check2016/jrm_test/'+str(i_name+1)+'/kitekan_correct'+str(set_num)+"-"+str(test_num)+'.csv',corr_out,delimiter=",")

bottom = 0
plt.xlabel("user id")
plt.ylabel("percentage of correct basis functions [%]")
for i in range(4):
  plt.bar(range(USR_NUM),count_array[:,i],bottom=bottom,label=graph_label[i])
  bottom = bottom+count_array[:,i]
  print count_array[:,i]
plt.legend()
plt.show()
#plt.savefig("correct_phi.eps")
