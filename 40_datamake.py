import numpy as np
from conv_num import *

INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 4
EPOCH = 50000

NUM_MAX = 30
TEST_NUM = 30 #the number of test data is 10

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

for i_name in username:
  print(i_name)
  tag_number = conv_num(i_name)

  #set input, output data to train

  xdata_b = "./jiken/" + i_name + "/factor_before.csv"
  x2data_b = "./jiken/" + i_name + "/kibun_before.csv"
  ydata_b = "./jiken/" + i_name + "/signal_before.csv"

  xdata_a = "./jiken/" + i_name + "/factor_after.csv"
  x2data_a = "./jiken/" + i_name + "/kibun_after.csv"
  ydata_a = "./jiken/" + i_name + "/signal_after.csv"

  xdata_c = "./jiken/" + i_name + "/factor_after2.csv"
  x2data_c = "./jiken/" + i_name + "/kibun_after2.csv"
  ydata_c = "./jiken/" + i_name + "/signal_after2.csv"

  factor_b= np.loadtxt(xdata_b,delimiter='\t')
  mood_b = np.loadtxt(x2data_b,delimiter='\t')
  face_b= np.loadtxt(ydata_b,delimiter='\t')
  #x_train_b = np.hstack((x1_train_b,x2_train_b.reshape(-1,1)))

  factor_a = np.loadtxt(xdata_a,delimiter='\t')
  mood_a = np.loadtxt(x2data_a,delimiter='\t')
  face_a = np.loadtxt(ydata_a,delimiter='\t')
  #x_train_a = np.hstack((x1_train_a,x2_train_a.reshape(-1,1)))

  factor_c = np.loadtxt(xdata_c,delimiter='\t')
  mood_c = np.loadtxt(x2data_c,delimiter='\t')
  face_c = np.loadtxt(ydata_c,delimiter='\t')
  #x_train_a = np.hstack((x1_train_a,x2_train_a.reshape(-1,1)))
  print(mood_a,mood_b)

  factor = np.vstack((factor_b,factor_a,factor_c))
  mood = np.hstack((mood_b,mood_a,mood_c))
  face = np.vstack((face_b,face_a,face_c))

  test_index = np.zeros(1,dtype='int')

  for set_num in (25,26,27,28,29):
    test_index0 = np.random.choice(range(0, NUM_MAX-1),29,replace=False) #the first index to start 30 test data sequence.

    for i in range(29):
      test_index[0] = test_index0[i]
      #for t in range(1,5):
      #test_index[t] = test_index[t-1]+1
      
      numbers = np.linspace(0,29,30,dtype='int')
      not_test_index = np.ones(len(numbers),dtype=bool)
      not_test_index[test_index] = False

      last_index = numbers[not_test_index]
      train_index = last_index[np.random.randint(0,NUM_MAX-5,set_num)]

      print('test',test_index, 'train',train_index)
      #raw_input()

      ####################

      factor_train= factor[train_index,:]
      mood_train= mood[train_index]
      face_train= face[train_index,:]

      factor_test= factor[test_index,:]
      mood_test= mood[test_index]
      face_test= face[test_index,:]

      save_factor_train = "/home/kazumi/prog/quiz_check2016/jrm_test/"+str(tag_number)+"/factor_train"+str(set_num) + "-" +str(i)+".csv"
      save_mood_train = "/home/kazumi/prog/quiz_check2016/jrm_test/"+str(tag_number)+"/mood_train"+str(set_num) + "-" +str(i)+".csv"
      save_face_train = "/home/kazumi/prog/quiz_check2016/jrm_test/"+str(tag_number)+"/face_train"+str(set_num) + "-" +str(i)+".csv"
      np.savetxt(save_factor_train,factor_train,fmt='%.4f',delimiter=",")
      np.savetxt(save_mood_train,mood_train,fmt='%.4f',delimiter=",")
      np.savetxt(save_face_train,face_train,fmt='%.4f',delimiter=",")

      save_factor_test = "/home/kazumi/prog/quiz_check2016/jrm_test/"+str(tag_number)+"/factor_test"+str(set_num) + "-" +str(i)+".csv"
      save_mood_test = "/home/kazumi/prog/quiz_check2016/jrm_test/"+str(tag_number)+"/mood_test"+str(set_num) + "-" +str(i)+".csv"
      save_face_test = "/home/kazumi/prog/quiz_check2016/jrm_test/"+str(tag_number)+"/face_test"+str(set_num) + "-" +str(i)+".csv"
      np.savetxt(save_factor_test,factor_test,fmt='%.4f',delimiter=",")
      np.savetxt(save_mood_test,mood_test,fmt='%.4f',delimiter=",")
      np.savetxt(save_face_test,face_test,fmt='%.4f',delimiter=",")

      selected_test = "/home/kazumi/prog/quiz_check2016/jrm_test/" +str(tag_number) + "/selected_num_test"+str(set_num) + "-"+ str(i) + ".csv"
      selected_train = "/home/kazumi/prog/quiz_check2016/jrm_test/" +str(tag_number) + "/selected_num_train"+str(set_num) + "-"+ str(i) + ".csv"
      np.savetxt(selected_test,test_index,fmt='%2d',delimiter=",") 
      np.savetxt(selected_train,train_index,fmt='%2d',delimiter=",") 
