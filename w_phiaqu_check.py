import matplotlib
import numpy as np
from func_new import *
import seaborn as sns
import matplotlib.pyplot as plt

from conv_num import *

EMO_NUM = 4
SIT_NUM = 10
USR_NUM = 9

name_list = ["inusan","kumasan","nekosan","test119","test120","test121","tomato","torisan","usagisan"]

xlabel = ["hap","sup","ang","sad"]
class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)



def cos_srangeim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


func_num = [[0,1,2,3],[4,5,6,7],[8,9,10,11],
  [12,13,14,15],[16,17,18,19],[20,21,22,23],
  [24,25,26,27],[28,29,30,31],[32,33,34,35],[36,37,38,39]]

def phi(factor,kibun,emo_num):
  ret = np.zeros(SIT_NUM)

  for sit_num in range(SIT_NUM):
    #func(factor[sit_num],kibun,func_num[sit_num][emo_num])
    ret[sit_num] = func(inv_norm(func_num[sit_num][emo_num],factor[sit_num]/100.0),kibun*10.0+0.001,func_num[sit_num][emo_num])

  return ret

def phi_4(factor,kibun):
  ret = np.zeros((SIT_NUM,4))
  for emo_num in range(EMO_NUM):
    ret[:,emo_num] = phi(factor,kibun,emo_num)

  return ret


def phi_out(factor,kibun):
  phi_out_arr = np.zeros((len(factor),10,4))


  for data_num in range(len(factor)):
      ret = phi_4(factor[data_num,:],kibun[data_num])
      phi_out_arr[data_num,:,:] = ret

  return phi_out_arr


if __name__ == '__main__':
  fig = plt.figure()

  j=0

  set_num = 25
  str_func = ["time of trial","rate of wins","encourage","symp","teasing","unrelated","no actions","total points","consecutive wins","consecutive losses"]

  for i_name in range(USR_NUM):
    for test_num in range(25):
      kitekan_correct = np.loadtxt('/home/kazumi/prog/quiz_check2016/jrm_test/'+str(i_name+1)+'/kitekan_correct'+str(set_num)+"-"+str(test_num)+'.csv',delimiter=",")

      #phi_out_arr = kitekan_correct
      weight_out = np.zeros((SIT_NUM,4))
      
      #plt.subplots_adjust(wspace=0.4,hspace=0.6)
      hap_w= np.loadtxt("./jrm_test/"+str(i_name+1)+"/hap_weight"+str(set_num)+"-"+str(test_num)+".csv",delimiter="\t")
      sup_w= np.loadtxt("./jrm_test/"+str(i_name+1)+"/sup_weight"+str(set_num)+"-"+str(test_num)+".csv",delimiter="\t")
      ang_w= np.loadtxt("./jrm_test/"+str(i_name+1)+"/ang_weight"+str(set_num)+"-"+str(test_num)+".csv",delimiter="\t")
      sad_w= np.loadtxt("./jrm_test/"+str(i_name+1)+"/sad_weight"+str(set_num)+"-"+str(test_num)+".csv",delimiter="\t")


      weight_out[:,0]=hap_w
      weight_out[:,1]=sup_w
      weight_out[:,2]=ang_w
      weight_out[:,3]=sad_w

      print(str_func)
      print(hap_w)
      print(sup_w)
      print(ang_w)
      print(sad_w)
      x=np.linspace(1,10,10)
      plt.bar(x-0.25,hap_w,color="pink",width=0.25,label="happy")
      plt.bar(x,sup_w,color="orange",width=0.25,label="surprise")
      plt.bar(x+0.25,ang_w,color="red",width=0.25,label="angry")
      plt.bar(x+0.5,sad_w,color="blue",width=0.25,label="sad")
      #plt.legend()
      #plt.xlabel("usr id")
      #plt.ylabel("correlation between function accuracy C and weight W")
      #plt.savefig("weight_vs_acu.eps")
      plt.show()
