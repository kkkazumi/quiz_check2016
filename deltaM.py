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
OUTPUT_DIM = 1
EPOCH = 50000

def m_conf(mental):
  return mental/40.0+0.5

for name_num in (0,1,2,3,4,5,6,7,8):
  dir_name = "./jrm_test/" + str(name_num+1)
  #for set_num in (5,10,15,20,25,30,35,40):
  #for set_num in (5,10,15,20,25):

  set_num = 25
  r = np.zeros(25)
  p = np.zeros(25)
  for i in range(25):#calculate the avelage value of estimated accuracy for datum in this loop
    i_csv = str(set_num) + "-" + str(i) + ".csv"

    #set input, output data to train
    factor_train = dir_name + "/factor_train" + i_csv
    mood_train = dir_name + "/TRAINestimated_phi" + i_csv
    #face_train = dir_name + "/face_train" + i_csv

    x1_train = np.loadtxt(factor_train,delimiter=',')
    x2_train = m_conf(np.loadtxt(mood_train,delimiter=','))

    _x_train = np.hstack((x1_train,x2_train.reshape(-1,1)))

    x_train = _x_train[1:,:]
    delta_mood = m_conf(np.diff(x2_train))
    y_train = delta_mood

    def objective(trial):
      #セッションのクリア
      K.clear_session()

      #最適化するパラメータの設定
      #中間層のユニット数
      mid_units = trial.suggest_int("mid_units", 5, 100)

      #活性化関数
      activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

      #optimizer
      optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])

      #define NN framework
      model = Sequential()
      model.add(Dense(mid_units, input_dim = INPUT_DIM, activation=activation))
      model.add(Dense(OUTPUT_DIM, activation=activation))

      model.compile(optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["accuracy"])

      history = model.fit(x_train, y_train, verbose=0, epochs=200, batch_size=128, validation_split=0.1)
      return 1 - history.history["val_acc"][-1]

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    mid_units=study.best_params["mid_units"]
    activation=study.best_params["activation"]
    optimizer=study.best_params["optimizer"]

    #define NN framework
    model = Sequential()
    model.add(Dense(mid_units, input_dim = INPUT_DIM, activation=activation))
    model.add(Dense(OUTPUT_DIM, activation=activation))

    model.compile(optimizer=optimizer,
          loss="mean_squared_error",
          metrics=["accuracy"])

    train=model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)

    #test to output
    _factor_test = np.loadtxt(dir_name + "/factor_test" + i_csv ,delimiter=',')
    _mood_test = np.loadtxt(dir_name + "/TESTestimated_phi" + i_csv ,delimiter=',')
    factor_test = _factor_test[1:,:]
    mood_test= m_conf(_mood_test[1:])

    m_pred = np.diff(m_conf(_mood_test))

    x_test = np.hstack((factor_test,mood_test.reshape(-1,1)))

    y_test = model.predict(x_test)

    #test output
    outname = dir_name + "/estimated_dM" + i_csv
    np.savetxt(outname,y_test,fmt="%.3f")
    lossname = dir_name + "/dM_loss" + i_csv
    np.savetxt(lossname,train.history['loss'])
