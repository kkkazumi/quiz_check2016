from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import scipy as sp
from scipy.stats import pearsonr

INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 4
EPOCH = 50000

#define NN framework
model = Sequential()
model.add(Dense(MID_UNIT, input_dim = INPUT_DIM, activation='relu'))
model.add(Dense(OUTPUT_DIM, activation='relu'))
model.compile(optimizer='Adam', loss='mean_squared_error')

#for name_num in (3,4,5,6,7,8,9):
#for name_num in (3,4,5,6,7,8,9):
name_num = 2
dir_name = "./jrm_test/" + str(name_num+1)
#for set_num in (5,10,15,20,25,30):
set_num = 40

r = np.zeros(30)
p = np.zeros(30)
for i in range(30):#calculate the avelage value of estimated accuracy for datum in this loop
  i_csv = str(set_num) + "-" + str(i) + ".csv"

  #set input, output data to train
  factor_train = dir_name + "/factor_train" + i_csv
  mood_train = dir_name + "/mood_train" + i_csv
  face_train = dir_name + "/face_train" + i_csv

  x1_train = np.loadtxt(factor_train,delimiter=',')
  x2_train = np.loadtxt(mood_train,delimiter=',')
  x_train = np.hstack((x1_train,x2_train.reshape(-1,1)))

  y_train = np.loadtxt(face_train,delimiter=',')

  train=model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)

  #test to output
  factor_test = np.loadtxt(dir_name + "/factor_test" + i_csv ,delimiter=',')
  mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
  face_test = np.loadtxt(dir_name + "/face_test" + i_csv ,delimiter=',')
  #m_ans = np.loadtxt("./jiken/" + i_name + "/kibun_after.csv",delimiter='\t')
  #y_sig = np.loadtxt("./jiken/" + i_name + "/signal_after.csv",delimiter='\t')

  m_pred = np.zeros_like(mood_test)

  #set data to test
  for m_t in range(len(mood_test)):
    x_tes = np.tile(factor_test[m_t],(100,1))
    m_candidate = np.linspace(0,1,100)
    y_ans = np.tile(face_test[m_t],(100,1))

    xtest = np.hstack((x_tes,m_candidate.reshape(-1,1)))
    ytest = model.predict(xtest)
    err = ytest - y_ans
    err_array = np.sum(err,1)
    m_est = np.argmin(abs(err_array))
    m_pred[m_t] = m_est/100.0

  #pearson r
  r[i], p[i] = pearsonr(mood_test, m_pred)

  #test output
  outname = dir_name + "/estimated_nn" + i_csv
  np.savetxt(outname,m_pred,fmt="%.3f")
  lossname = dir_name + "/nn_loss" + i_csv
  np.savetxt(lossname,train.history['loss'])
np.savetxt(dir_name+"/nn_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
np.savetxt(dir_name+"/nn_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')

