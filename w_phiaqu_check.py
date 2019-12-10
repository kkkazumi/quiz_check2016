import matplotlib
import numpy as np
from func import *
import seaborn as sns
import matplotlib.pyplot as plt

from conv_num import *

DAT_NUM = 50
EMO_NUM = 4
SIT_NUM = 10

name_list = ["inusan","kumasan","nekosan","test119","test120","test121","tomato","torisan","usagisan"]
#name_list = ["inusan","kubosan","kumasan","nekosan","test119","test120","test121","tomato","torisan","usagisan"]
#name_list = ["inusan","kubosan","kumasan","nekosan","test119","test120","test121","tomato","torisan","usagisan","sarada"]

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



def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


func_num = [[0,1,2,3],[4,5,6,7],[8,9,10,11],
  [12,13,14,15],[16,17,18,19],[20,21,22,23],
  [24,25,26,27],[28,29,30,31],[32,33,34,35],[36,37,38,39]]

def phi(factor,kibun,emo_num):
  ret = np.zeros(SIT_NUM)

  for sit_num in range(SIT_NUM):
    ret[sit_num] = func(factor[sit_num],kibun,func_num[sit_num][emo_num])

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
  plt.subplots_adjust(wspace=0.4,hspace=0.6)

  j=0

  corr_out_all = np.zeros((len(name_list),SIT_NUM,4))

  for i_name in name_list:
    plt.subplot(3,4,j+1)
    factor1 = np.loadtxt("./jiken/"+i_name+"/factor_after.csv",delimiter="\t")
    factor2 = np.loadtxt("./jiken/"+i_name+"/factor_after2.csv",delimiter="\t")
    factor3 = np.loadtxt("./jiken/"+i_name+"/factor_before.csv",delimiter="\t")
    factor=np.vstack((factor1,factor2,factor3))
    kibun1 = np.loadtxt("./jiken/"+i_name+"/kibun_after.csv",delimiter="\t")
    kibun2 = np.loadtxt("./jiken/"+i_name+"/kibun_after2.csv",delimiter="\t")
    kibun3 = np.loadtxt("./jiken/"+i_name+"/kibun_before.csv",delimiter="\t")
    _kibun=np.append(kibun1,kibun2)
    kibun=np.append(_kibun,kibun3)
    signal1 = np.loadtxt("./jiken/"+i_name+"/signal_after.csv",delimiter="\t")
    signal2 = np.loadtxt("./jiken/"+i_name+"/signal_after2.csv",delimiter="\t")
    signal3 = np.loadtxt("./jiken/"+i_name+"/signal_before.csv",delimiter="\t")
    signal=np.vstack((signal1,signal2,signal3))

    phi_out_arr = phi_out(factor,kibun)
    corr_out = np.zeros((SIT_NUM,4))
    weight_out = np.zeros((SIT_NUM,4))
    
    for sit_num in range(SIT_NUM):
      #array = signal - phi_out[:,sit_num,:]
      for emo_num in range(EMO_NUM):
        corr = np.corrcoef(signal[:,emo_num],phi_out_arr[:,sit_num,emo_num])
        corr_out[sit_num,emo_num]=corr[0,1]

    corr_out_all[j,:]=corr_out
    sim = np.zeros((1,4))

    num=0

    plt.subplots_adjust(wspace=0.4,hspace=0.6)
    hap_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/hap_weight50-"+str(num)+".csv",delimiter="\t")
    sup_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/sup_weight50-"+str(num)+".csv",delimiter="\t")
    ang_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/ang_weight50-"+str(num)+".csv",delimiter="\t")
    sad_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/sad_weight50-"+str(num)+".csv",delimiter="\t")

    weight_out[:,0]=hap_w
    weight_out[:,1]=sup_w
    weight_out[:,2]=ang_w
    weight_out[:,3]=sad_w


    for i in range(EMO_NUM):
      #sim[num,i]=cos_sim(corr_out[:,i], weight_out[:,i])
      sim[num,i]=np.corrcoef(corr_out[:,i], weight_out[:,i])[0,1]

    print sim[0,0],",", sim[0,1],",", sim[0,2],",", sim[0,3]
    #print np.mean((sim),axis=0)
    #print np.count_nonzero(sim>0.1), sim.size


    sns.heatmap(corr_out,cmap="seismic",norm=SqueezedNorm(vmin=-1, vmax=1, mid=0))
    plt.xticks([0.5,1.5,2.5,3.5],xlabel)
    j+=1
    plt.title(i_name)
    #print np.mean(abs(corr_out_all),axis=0)
    #print conv_num(i_name),float(np.count_nonzero(corr_out>0.5))/float(corr_out.size)
    #print conv_num(i_name),np.count_nonzero(corr_out>0.1)
  plt.show()

#np.savetxt("phi_corr_"+i_name+".csv",corr_out,delimiter=",")
#output of phi function
#sprintf(out_name,"./jiken/%s/phi_out.csv",argv[1]);
#sprintf(phi_data,"./jiken/%s/phi_stdata.csv",argv[1]);//statistical data
