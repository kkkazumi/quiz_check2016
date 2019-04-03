import numpy as np
import matplotlib.pyplot as plt

username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

for name in username:
  predicted = np.loadtxt("./jiken/"+name+"/kibun_predicted.csv",delimiter=",")
  answered = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
  x = range(10)
  plt.title(name)
  plt.plot(x,predicted,label="predicted")
  plt.plot(x,answered,label="answered")
  plt.show()
