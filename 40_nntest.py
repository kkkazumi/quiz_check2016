import optuna

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import scipy as sp
from scipy.stats import pearsonr

INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 4
EPOCH = 50000


def create_model(num_layer, activation, mid_units, num_filters):
  """
  num_layer : 畳込み層の数
  activation : 活性化関数
  mid_units : FC層のユニット数
  num_filters : 各畳込み層のフィルタ数
  """
  inputs = Input(shape=(INPUT_DIM,))
  x = Convolution2D(filters=num_filters[0], kernel_size=(3,3), padding="same", activation=activation)(inputs)
  for i in range(1,num_layer):
      x = Convolution2D(filters=num_filters[i], kernel_size=(3,3), padding="same", activation=activation)(x)

  x = GlobalAveragePooling2D()(x)
  x = Dense(units=mid_units, activation=activation)(x)
  x = Dense(units=10, activation="softmax")(x)

  model = Model(inputs=inputs, outputs=x)
  return model

def objective(trial):
  #セッションのクリア
  K.clear_session()

  #最適化するパラメータの設定
  #中間層のユニット数
  mid_units = int(trial.suggest_discrete_uniform("mid_units", 10, 20, 30))

  #活性化関数
  activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

  #optimizer
  optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])

  #define NN framework
  model = Sequential()
  model.add(Dense(mid_units, input_dim = INPUT_DIM, activation=activation))
  model.add(Dense(OUTPUT_DIM, activation='relu'))

  model.compile(optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["accuracy"])

  #検証用データに対する正答率が最大となるハイパーパラメータを求める

study = optuna.create_study()
study.optimize(objective, n_trial=100)


for name_num in (0,1,2,3,4,5,6,7,8):
  dir_name = "./jrm_test/" + str(name_num+1)
  for set_num in (5,10,15,20,25,30,35,40):

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
      #np.savetxt(outname,m_pred,fmt="%.3f")
      lossname = dir_name + "/nn_loss" + i_csv
      #np.savetxt(lossname,train.history['loss'])
    #np.savetxt(dir_name+"/nn_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    #np.savetxt(dir_name+"/nn_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
