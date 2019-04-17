import numpy as np
import matplotlib.pyplot as plt

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']
#username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

for name in username:
  predicted = np.loadtxt("./jiken/"+name+"/m_pred.csv",delimiter=",")
  answered = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
  x = range(10)
  fig, ax1 = plt.subplots()
  ln1 = ax1.plot(x,predicted,'C0',label="predicted")
  ax2 = ax1.twinx()
  ln2 = ax2.plot(x,answered,'C1',label="answered")

  h1, l1 = ax1.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax1.legend(h1+h2,l1+l2,loc="lower right")

  ax1.set_xlabel('time transition')
  ax1.set_ylabel('predicted M')
  ax2.set_ylabel('answered M')
  plt.title(name)

  plt.show()
