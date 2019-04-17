from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 4
EPOCH = 50000

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']
#username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

#define NN framework
model = Sequential()
model.add(Dense(MID_UNIT, input_dim = INPUT_DIM, activation='relu'))
model.add(Dense(OUTPUT_DIM, activation='relu'))
model.compile(optimizer='Adam', loss='mean_squared_error')

for i_name in username:
  print(i_name)

  #set input, output data to train
  xdata = "./jiken/" + i_name + "/factor_before.csv"
  x2data = "./jiken/" + i_name + "/kibun_before.csv"
  ydata = "./jiken/" + i_name + "/signal_before.csv"
  x1_train = np.loadtxt(xdata,delimiter='\t')
  x2_train = np.loadtxt(x2data,delimiter='\t')
  x_train = np.hstack((x1_train,x2_train.reshape(-1,1)))

  y_train = np.loadtxt(ydata,delimiter='\t')

  train=model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)

  #test to output
  x_fac = np.loadtxt("./jiken/" + i_name + "/factor_after.csv",delimiter='\t')
  m_ans = np.loadtxt("./jiken/" + i_name + "/kibun_after.csv",delimiter='\t')
  y_sig = np.loadtxt("./jiken/" + i_name + "/signal_after.csv",delimiter='\t')

  m_pred = np.zeros_like(m_ans)

  #set data to test
  for m_t in range(len(m_ans)):
    x_tes = np.tile(x_fac[m_t],(100,1))
    m_candidate = np.linspace(0,1,100)
    y_ans = np.tile(y_sig[m_t],(100,1))

    xtest = np.hstack((x_tes,m_candidate.reshape(-1,1)))
    ytest = model.predict(xtest)
    err = ytest - y_ans
    err_array = np.sum(err,1)
    m_est = np.argmin(abs(err_array))
    m_pred[m_t] = m_est/100.0

  print(m_pred)
  print(m_ans)
  #test output
  outname = "./jiken/" + i_name + "/m_pred.csv"
  np.savetxt(outname,m_pred)
  lossname = "./jiken/" + i_name + "/nnloss_hist.csv"
  np.savetxt(lossname,train.history['loss'])
