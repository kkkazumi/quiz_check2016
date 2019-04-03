from keras.models import Sequential
from keras.layers import Dense, Activation

INPUT_DIM = 10
MID_UNIT = 15
OUTPUT_DIM = 1
EPOCH = 2000

username = [inusan, kubosan, kumasan, nekosan, sarada, test119, test120, test121, tomato, torisan, usagisan]

#define NN framework
model = Sequential()
model.add(Dense(MID_UNIT, input_dim = INPUT_DIM, activation='sigmoid'))
model.add(Dense(OUTPUT_DIM, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy')

for i_name in username:
	#set input, output data to train
	xdata = "~/data/jiken/" + i_name + "/factor_before.csv"
	ydata = "~/data/jiken/" + i_name + "/kibun_before.csv"
	x_train = np.loadtxt(xdata,delimiter=',')
	y_train = np.loadtxt(ydata,delimiter=',')

	#train
	model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)

	#set data to test
	xtest = "~/data/jiken/" + i_name + "/factor_after.csv"
	yans = "~/data/jiken/" + i_name + "/kibun_after.csv"

	#test output
	ytest = model.predict(xtest)

	print("factor","predicted","self-assessed")
	print(x_test, y_test, yans)
