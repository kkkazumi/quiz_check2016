import matplotlib
import numpy as np
from func_new import *
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

EMO_NUM = 4
SIT_NUM = 10
USR_NUM = 9

xlabel = ["hap","sup","ang","sad"]

func_num = [[0,1,2,3],[4,5,6,7],[8,9,10,11],
  [12,13,14,15],[16,17,18,19],[20,21,22,23],
  [24,25,26,27],[28,29,30,31],[32,33,34,35],[36,37,38,39]]

str_func = ["time of trial","rate of wins","encourage","symp","teasing","unrelated","no actions","total points","consecutive wins","consecutive losses"]
str_emo = ["hap","sup","ang","sad"]

j=0
count_array=np.zeros((USR_NUM,4))
graph_label = ["0.1<r<0.2","0.2<r<0.3","0.3<r<0.4","0.4<r"]

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

def out_kitekan_correct(i_name, set_num):
  factor = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+i_name+"/factor_before.csv",delimiter="\t")
  print(factor.shape)
  kibun = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+i_name+"/kibun_before.csv",delimiter="\t")
  #factor = np.tile(np.linspace(0,1,30),(10,1)).T
  print(factor.shape)
  #kibun = np.ones(30)

  phi_out = np.ones((30,SIT_NUM,EMO_NUM))
  #for tt in range(26):
  tt=0
  for data_num in range(30):
    for emo_num in range(EMO_NUM):
      for sit_num in range(SIT_NUM):
        ret = func(inv_norm(func_num[sit_num][emo_num],factor[tt+data_num][sit_num]),kibun[tt+data_num]*10.0+0.001,func_num[sit_num][emo_num])
        phi_out[data_num][sit_num][emo_num] = ret

  return phi_out

fig,ax=plt.subplots(figsize=(16,12))

color_label = ["red","darkorange","gold","lime","green","deepskyblue","blue","purple","hotpink"]

#for tt in range(26):
  #array = signal - phi_out[:,sit_num,:]
for sit_num in range(SIT_NUM):
  for emo_num in range(EMO_NUM):

    for i_name in username:
      weight_out = np.zeros((10,4))
      set_num = 30
      corr_out = np.zeros((SIT_NUM,4))
      phi_out = out_kitekan_correct(i_name,set_num)
      signal = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+i_name+"/signal_before.csv",delimiter="\t")
      factor = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+i_name+"/factor_before.csv",delimiter="\t")
      kibun = np.loadtxt("/home/kazumi/prog/quiz_check2016/jiken/"+i_name+"/kibun_before.csv",delimiter="\t")

      #plt.subplots_adjust(wspace=0.4,hspace=0.6)
      for test_num in range(25):
        hap_w= np.loadtxt("./jrm_test/"+str(username.index(i_name)+1)+"/hap_weight25-"+str(test_num)+".csv",delimiter="\t")
        sup_w= np.loadtxt("./jrm_test/"+str(username.index(i_name)+1)+"/sup_weight25-"+str(test_num)+".csv",delimiter="\t")
        ang_w= np.loadtxt("./jrm_test/"+str(username.index(i_name)+1)+"/ang_weight25-"+str(test_num)+".csv",delimiter="\t")
        sad_w= np.loadtxt("./jrm_test/"+str(username.index(i_name)+1)+"/sad_weight25-"+str(test_num)+".csv",delimiter="\t")

        weight_out[:,0]=+hap_w
        weight_out[:,1]=+sup_w
        weight_out[:,2]=+ang_w
        weight_out[:,3]=+sad_w


      plt.subplot(3,3,username.index(i_name)+1)
      tt = 0
      corr = signal[tt:tt+30,emo_num]-phi_out[:,sit_num,emo_num]

      #plt.plot(signal[:,emo_num],linewidth=1)
      #plt.plot(phi_out[:,sit_num,emo_num],linewidth=1)
      #plt.scatter(np.linspace(0,29,30),signal[:,emo_num],marker="x",c=kibun,label="signal")
      #plt.scatter(np.linspace(0,29,30),phi_out[:,sit_num,emo_num],c=kibun,label="phi out")

      #plt.legend()
      #plt.show()

      plt.scatter(factor[:,sit_num],phi_out[:,sit_num,emo_num],alpha=0.5,c=kibun,label="phi out")
      plt.scatter(factor[:,sit_num],signal[:,emo_num],alpha=0.5,marker="x",c=kibun,label="")


      plt.title("sit:"+str_func[sit_num]+"-emo:"+str_emo[emo_num]+",w:"+str(np.mean(weight_out[sit_num,emo_num])))
      plt.ylabel("val of signal/phi_out")
      plt.xlabel("factor value")
      plt.ylim(-0.2,1.2)
      plt.xlim(-0.2,1.2)
    plt.legend()
    plt.show()
    #plt.savefig("./factor_graph/sit-"+str_func[sit_num]+"_emo-"+str_emo[emo_num]+".png")
    plt.clf()
