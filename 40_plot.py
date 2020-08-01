import numpy as np
import matplotlib.pyplot as plt

#ok_list = np.array([[2,9],[6,18],[7,6]])#2nd, 3rd, 1st
ok_list = np.array([[7,6]])#2nd, 3rd, 1st

diff = np.zeros((9,30))

for user_name in range(9):
  dir_name = "./jrm_test/" + str(user_name+1)
  #for set_num in (5,10,15,20,25,30):
  set_num = 40
  for i in range(30):
    i_csv = str(set_num) + "-" + str(i) + ".csv"
    mood_test= np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
    #mood_est = np.loadtxt(dir_name + "/estimated_dummy" + i_csv, delimiter=',')
    mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')
    diff[user_name,i] = np.mean(abs(mood_test -mood_est/10.0))

sort_diff = np.argsort(diff)
print(diff)
print(sort_diff[0])
print np.unravel_index(np.argmin(diff),diff.shape) ,np.min(diff)

j=0

for user_name in range(9):
  dir_name = "./jrm_test/" + str(user_name+1)
  #for set_num in (5,10,15,20,25,30):
  set_num = 40
  for i in range(30):


    if ok_list[j,0] == user_name:
      i = ok_list[j,1]

      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_test= np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      #mood_est = np.loadtxt(dir_name + "/estimated_dummy" + i_csv, delimiter=',')
      mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')

      #d=np.mean(abs(mood_test -mood_est/10.0))
      #if d<0.2:
      #  print(user_name,i,d)

      #print pearsonr(mood_test,mood_est)

      fz=12
      label=np.linspace(18,27,10)

      fig,ax = plt.subplots()
      plt.plot(label,mood_test,color='limegreen',marker='o',label='self-asessed mood')
      plt.plot(label,mood_est/10.0,color='green',linestyle='dashed',marker='o',markerfacecolor='white',label='estimated mood')
      plt.ylabel('estimated/self-assessed mood',fontsize=fz)
      plt.ylim(-0.02,1.2)
      plt.legend()

      #plt.show()
      plt.savefig('1st_hi.eps')
      #j=j+1
