import sys
import numpy as np

import matplotlib.pyplot as plt

def min_max(trg_array,std_x, axis=None):
  _min = std_x.min(axis=axis, keepdims=True)
  _max = std_x.max(axis=axis, keepdims=True)
  mmax = _max
  mmin = _min
  result = (trg_array-mmin)/(mmax-mmin)
  return result,1.0/(mmax-mmin)

if sys.argv[1] == 'phi':
  num = 0
elif sys.argv[1] == 'nn':
  num = 1

filename = [['TESTestimated_phi','estimated_nn'],['TRAINestimated_phi','TRAINestimated_nn']]
  
for name_num in range(9):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (25,26,27,28,29):
    #set_num = 35
    diff = np.zeros(29)
    point = np.zeros(29)
    rate = np.zeros(29)
    for i in range(29):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      mood_train_ans = np.loadtxt(dir_name + "/mood_train" + i_csv, delimiter=',')
      mood_test_ans = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')

      mood_test_ans_scaled,rate[i] = min_max(mood_test_ans,np.append(mood_train_ans,mood_test_ans))
      mood_train_ans_scaled,rate[i] = min_max(mood_train_ans,np.append(mood_train_ans,mood_test_ans))

      ans_ave_distance = mood_test_ans_scaled-np.mean(mood_train_ans_scaled)
      ans_array = mood_test_ans - np.mean(mood_train_ans)


      #33
      mood_train_est = np.loadtxt(dir_name + "/"+filename[1][num] + i_csv, delimiter=',')
      mood_test_est = np.loadtxt(dir_name + "/"+filename[0][num] + i_csv, delimiter=',')


      mood_test_est_scaled,rate[i] = min_max(mood_test_est,np.append(mood_train_est,mood_test_est))
      mood_train_est_scaled,rate[i] = min_max(mood_train_est,np.append(mood_train_est,mood_test_est))

      est_ave_distance = mood_test_est_scaled-np.mean(mood_train_est_scaled)

      print('est_ave_distance',est_ave_distance,'ans_ave_distance',ans_ave_distance)
      point[i]=(est_ave_distance+1)/(ans_ave_distance+1)*100

      #array=ans_array * est_array
      #diff[i]=np.count_nonzero(array[array>0])

      #print(np.count_nonzero(array[array>0]))

    #print(name_num,set_num,diff)
    np.savetxt(dir_name+"/"+sys.argv[1]+"_point-"+str(set_num)+".csv",point,fmt='%.5f',delimiter=',')

