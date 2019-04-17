from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 4
EPOCH = 5000

username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

#define NN framework
model = Sequential()
model.add(Dense(MID_UNIT, input_dim = INPUT_DIM, activation='relu'))
model.add(Dense(OUTPUT_DIM, activation='relu'))
model.compile(optimizer='Adam', loss='mean_squared_error')

for i_name in username:

	#set input, output data to train
	xdata = "./jiken/" + i_name + "/factor_before.csv"
	x2data = "./jiken/" + i_name + "/kibun_before.csv"
	ydata = "./jiken/" + i_name + "/signal_before.csv"
	x1_train = np.loadtxt(xdata,delimiter='\t')
	x2_train = np.loadtxt(x2data,delimiter='\t')
	x_train = np.hstack((x1_train,x2_train))

	y_train = np.loadtxt(ydata,delimiter='\t')

	train=model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)

	#set data to test
	for i in range():
	x1test = np.loadtxt("./jiken/" + i_name + "/factor_after.csv",delimiter='\t')
	x2test = np.loadtxt("./jiken/" + i_name + "/kibun_after.csv",delimiter='\t')
	xtest = np.hstack((x1test,x2test))
	yans = np.loadtxt("./jiken/" + i_name + "/signal_after.csv",delimiter='\t')

	ycheck = model.predict(x_train)
	print("ch-factor","ch-predicted","ch-self-assessed")
	print(ycheck[:,0], y_train)
	print(np.mean(ycheck[:,0]-y_train))

	#test output
	ytest = model.predict(xtest)
	print("factor","predicted","self-assessed")
	print(xtest, ytest, yans)
	outname = "./jiken/" + i_name + "/kibun_pre.csv"
	np.savetxt(outname,ytest[:,0])
	lossname = "./jiken/" + i_name + "/loss_hist.csv"
	np.savetxt(lossname,train.history['loss'])
