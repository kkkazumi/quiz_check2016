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



func_num = [[0,1,2,3],[4,5,6,7],[8,9,10,11],
  [12,13,14,15],[16,17,18,19],[20,21,22,23],
  [24,25,26,27],[28,29,30,31],[32,33,34,35],[36,37,38,39]]


if __name__ == '__main__':

  fig = plt.figure()
  j=0

  corr_out_all = np.zeros((len(name_list),1,SIT_NUM,EMO_NUM))
  array=np.zeros((len(name_list),SIT_NUM))
  for i_name in name_list:
    #plt.subplot(3,4,j+1)
    corr_out = np.zeros((SIT_NUM,4))

    num = 0
    plt.subplots_adjust(wspace=0.4,hspace=0.6)
    hap_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/hap_weight50-"+str(num)+".csv",delimiter="\t")
    sup_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/sup_weight50-"+str(num)+".csv",delimiter="\t")
    ang_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/ang_weight50-"+str(num)+".csv",delimiter="\t")
    sad_w= np.loadtxt("./jrm_test/"+str(conv_num(i_name))+"/sad_weight50-"+str(num)+".csv",delimiter="\t")

    corr_out[:,0]=hap_w
    corr_out[:,1]=sup_w
    corr_out[:,2]=ang_w
    corr_out[:,3]=sad_w

    corr_out_all[j,num,:,:] = corr_out

    """
    sns.heatmap((corr_out),cmap="seismic",norm=SqueezedNorm(vmin=-1, vmax=1, mid=0),cbar=False)
    #plt.plot(range(10),hap_w)
    plt.xticks([0.5,1.5,2.5,3.5],xlabel)
    plt.title(i_name)
    #print np.mean(abs(corr_out_all),axis=0)
    #print conv_num(i_name),float(np.count_nonzero(corr_out>0.5))/float(corr_out.size)
    plt.pause(.001)
    plt.cla()
    """
    array[j,:]=np.mean(np.mean(abs(corr_out_all[j,:,:,:]),axis=0),axis=1)
    print array[j,:]
    j+=1
  print"las", np.mean(array,axis=0)

#np.savetxt("phi_corr_"+i_name+".csv",corr_out,delimiter=",")
#output of phi function
#sprintf(out_name,"./jiken/%s/phi_out.csv",argv[1]);
#sprintf(phi_data,"./jiken/%s/phi_stdata.csv",argv[1]);//statistical data
