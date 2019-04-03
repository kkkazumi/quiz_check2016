from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

INPUT_DIM = 10
MID_UNIT = 15
OUTPUT_DIM = 1
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
	ydata = "./jiken/" + i_name + "/kibun_before.csv"
	x_train = np.loadtxt(xdata,delimiter='\t')
	y_train = np.loadtxt(ydata,delimiter='\t')

	train=model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)

	#set data to test
	xtest = np.loadtxt("./jiken/" + i_name + "/factor_after.csv",delimiter='\t')
	yans = np.loadtxt("./jiken/" + i_name + "/kibun_after.csv",delimiter='\t')

	ycheck = model.predict(x_train)
	print("ch-factor","ch-predicted","ch-self-assessed")
	print(ycheck[:,0], y_train)
	print(np.mean(ycheck[:,0]-y_train))

	#test output
	ytest = model.predict(xtest)
	print("factor","predicted","self-assessed")
	print(xtest, ytest, yans)
	outname = "./jiken/" + i_name + "/kibun_predicted50k.csv"
	np.savetxt(outname,ytest[:,0])
	lossname = "./jiken/" + i_name + "/loss_history50k.csv"
	np.savetxt(lossname,train.history['loss'])
