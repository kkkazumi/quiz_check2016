import numpy as np
import matplotlib.pyplot as plt
from conv_num import * #for id number

def ret_count(A, val):
  if val == 1:
    return len(np.where(A==1)[0])+len(np.where(A==0)[0])
  elif val == -1:
    return len(np.where(A==-1)[0])
  

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']
#username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

def ave_count(username):
  face_sum= np.empty(9)
  mental_sum= np.empty(9)
  i=0

  for name in username:
    kibun_mental = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
    kibun_face = np.loadtxt("./jiken/"+name+"/kibun_after2.csv",delimiter=",")

    mental_sum[i] = np.sum(kibun_mental)
    face_sum[i] = np.sum(kibun_face)

    #plt.plot(kibun_face,label='face')
    #plt.plot(kibun_mental,label='mental')
    #plt.legend()
    #plt.title(name)
    #plt.show()
    i=i+1
  return face_sum, mental_sum


def keep_count(username):
  face_up= np.empty(9)
  mental_up= np.empty(9)
  i=0

  for name in username:
    kibun_mental = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
    kibun_face = np.loadtxt("./jiken/"+name+"/kibun_after2.csv",delimiter=",")


    sign_mental= np.sign(np.diff(kibun_mental))
    sign_face= np.sign(np.diff(kibun_face))
    print 'mental: (', ret_count(sign_mental,1),',',ret_count(sign_mental,-1)
    print 'face: (', ret_count(sign_face,1),',',ret_count(sign_face,-1)
    face_up[i]= ret_count(sign_face,1)
    mental_up[i]= ret_count(sign_mental,1)

    #plt.plot(kibun_face,label='face')
    #plt.plot(kibun_mental,label='mental')
    #plt.legend()
    #plt.title(name)
    #plt.show()
    i=i+1
  return face_up, mental_up

if __name__ == '__main__':
  face_up, mental_up = keep_count(username)
  face_sum, mental_sum= ave_count(username)

  left = np.arange(len(username))
  width = 0.3
  username = conv_list(username)

  g1=plt.bar(left,mental_sum,color='r',label='mental_sum',width=width,align='center')
  g2=plt.bar(left+width, face_sum, color='b',label='face_sum', tick_label=username,width=width, align='center')
  #g3=plt.bar(left+width*0, mental_up, color='blue',label='mental', tick_label=username,width=width, align='center')
  #g4=plt.bar(left+width*1, face_up, color='red',label='face', tick_label=username,width=width, align='center')
  plt.legend(handles=[g1,g2],loc='best',shadow=True)
  plt.xlabel('user id')
  plt.ylabel('average of mood')
  #plt.ylabel('count of up/keep')

  #plt.savefig('compare_keep.eps')
  plt.show()
  #plt.savefig('compare_ave.eps')

