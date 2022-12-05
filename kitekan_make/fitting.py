import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def get_spl(x_data,y_data):
  f_sci = interpolate.interp1d(x_data,y_data,kind="cubic")

USER_NUM = 9

FACTOR_NUM = 10
FACE_TYPE = 4

DIR_PATH = "../../est_pred_pc/data/"

def show_graph(factor_data,signal_data):
  for factor_type in range(FACTOR_NUM):
    for signal_type in range(FACE_TYPE):
      for m in range(10):
        x_data = factor_data[:,factor_type]
        y_data = signal_data[:,signal_type]
        f_sci = interpolate.interp1d(x_data,y_data,kind="cubic")
        plt.plot(x_data,y_data,'o')
        plt.plot(x_data,f_sci(x_data),'-')
        plt.show()


for username in range(1,USER_NUM):
  factor_file = DIR_PATH+str(username)+"/factor_before.csv"
  signal_file = DIR_PATH+str(username)+"/signal_before.csv"

  factor_data = np.loadtxt(factor_file,delimiter="\t")
  signal_data = np.loadtxt(signal_file,delimiter="\t")

  show_graph(factor_data,signal_data)
