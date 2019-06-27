import numpy as np
import matplotlib.pyplot as plt
from conv_num import * #for id number

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']
#username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

corr_ms = np.empty(9)
corr_mm = np.empty(9)
var = np.empty(9)
print(corr_ms.shape)
i=0

for name in username:
  predicted = np.loadtxt("./jiken/"+name+"/m_pred.csv",delimiter=",")
  answered = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
  faced = np.loadtxt("./jiken/"+name+"/signal_after.csv",delimiter="\t")
  before_face = np.loadtxt("./jiken/"+name+"/signal_before.csv",delimiter="\t")
  #plt.plot(before_face[:,0])
  #plt.title(name)
  #plt.show()

  corr_ms[i] = np.corrcoef(answered,faced[:,0])[1,0]
  corr_mm[i] = np.corrcoef(answered,predicted)[1,0]

  var[i] = np.sqrt(np.var(before_face[:,0]))
  #print('ms,mm',corr_ms[1,0],corr_mm[1,0])
  x = range(10)
  print name, var[i]
  i=i+1

left = np.arange(len(username))
width = 0.3

name_list = conv_list(username)

g1=plt.bar(left,corr_ms,color='r',label='observed happy face',width=width,align='center')
g2=plt.bar(left+width, corr_mm, color='b',label='estimated mental state', tick_label=name_list,width=width, align='center')
#g3=plt.bar(left+width+width, var, color='g',label='var', tick_label=username,width=width, align='center')
#plt.bar(left+width+width, var, color='g',label='var', tick_label=username,width=width, align='center')
plt.legend(handles=[g1,g2],loc='best',shadow=True)
plt.xlabel('user id')
plt.ylabel('correlation of answered M vs face/predicted M')

plt.show()
