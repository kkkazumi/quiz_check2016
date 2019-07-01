import numpy as np
import matplotlib.pyplot as plt
from conv_num import * #for id number

f_size = 13

#username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

#inusan,kubosan,kumasan,nekosan,sarada,test119,test120,test121,tomato,torisan,usagisan

def out_est(name_list):
  corr_ms = np.empty(9)
  corr_mm = np.empty(9)
  var = np.empty(9)
  i=0
  phi = np.loadtxt("./jiken/phi_predicted.csv",delimiter=",",skiprows=1,dtype='float16')
  phi_list = np.genfromtxt("./jiken/phi_predicted.csv",delimiter=",",dtype=None)
  #print phi

  for name in name_list:
    #predicted = phi[:,i]
    index = np.where(phi_list[0]==name)
    data = phi[:,index]
    predicted = data.flatten()
    #print 'data',data,'predicted',predicted
    answered = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
    faced = np.loadtxt("./jiken/"+name+"/signal_after.csv",delimiter="\t")
    #plt.title(name)
    #plt.show()

    corr_ms[i] = np.corrcoef(answered,faced[:,0])[1,0]
    corr_mm[i] = np.corrcoef(answered,predicted)[1,0]

    #var[i] = np.sqrt(np.var(before_face[:,0]))
    #print('ms,mm',corr_ms[1,0],corr_mm[1,0])
    x = range(len(username))
    #print name, var[i]
    i=i+1
  return corr_mm, corr_ms

if __name__ == '__main__':
  username = ['inusan','kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']
  corr_mm, corr_ms = out_est(username)
  left = np.arange(len(username))
  width = 0.3

  name_list = conv_list(username)

  g1=plt.bar(left,corr_ms,color='r',label='observed happy face',width=width,align='center')
  g2=plt.bar(left+width, corr_mm, color='b',label='estimated mental state', tick_label=name_list,width=width, align='center')
#g3=plt.bar(left+width+width, var, color='g',label='var', tick_label=username,width=width, align='center')
#plt.bar(left+width+width, var, color='g',label='var', tick_label=username,width=width, align='center')
  plt.legend(handles=[g1,g2],loc='best',shadow=True,fontsize=f_size)
  plt.xlabel('user id',fontsize=f_size)
  plt.ylabel('correlation of answered M vs face/predicted M',fontsize=f_size)

  plt.show()
#plt.savefig('correlation_new.eps')
